import modal
import os
import time
from pydantic import BaseModel

class GenerationRequest(BaseModel):
    image: str  # URL to the source image or video
    audio1: str # URL to the first audio file
    prompt: str | None = None # (Optional) text prompt

# Use the new App class instead of Stub
app = modal.App("infinitetalk-api")

# Define persistent volumes for models and outputs
model_volume = modal.Volume.from_name(
    "infinitetalk-models", create_if_missing=True
)
output_volume = modal.Volume.from_name(
    "infinitetalk-outputs", create_if_missing=True
)
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

# Define the custom image with all dependencies
image = (
    # Use the official PyTorch development image which includes nvcc for compiling flash-attn
    modal.Image.from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel")
    # Set environment variable to prevent download timeouts
    .env({"HF_HUB_ETAG_TIMEOUT": "60"})
    # Mount the local InfiniteTalk directory into the container.
    # copy=True is required because we run pip install from this directory later.
    .add_local_dir("infinitetalk", "/root/infinitetalk", copy=True)
    .apt_install("git", "ffmpeg", "git-lfs", "libmagic1")
    # Fix Python 3.11 compatibility issue - remove deprecated ArgSpec import
    .run_commands("sed -i 's/from inspect import ArgSpec/# from inspect import ArgSpec  # Removed for Python 3.11 compatibility/' /root/infinitetalk/wan/multitalk.py")
    .pip_install(
        # Install flash-attn
        "misaki[en]",
        "ninja", 
        "psutil", 
        "packaging",
        "flash_attn==2.7.4.post1",
        # Install other core dependencies
        "pydantic",
        "python-magic",
        "huggingface_hub",
        # Add missing audio dependencies that aren't in requirements.txt
        "soundfile",
        "librosa",  # Common audio processing library often used with soundfile
        # Add missing xformers dependency
        "xformers==0.0.28"
        # "sageattention==1.0.6"
    )
    # Install all other dependencies from the original requirements file.
    .pip_install_from_requirements("infinitetalk/requirements.txt")
)

# --- CPU-only API Class for w polling ---
@app.cls(
    cpu=1.0,  # Explicitly use CPU-only containers
    image=image.pip_install("python-magic"),  # Lightweight image for API endpoints
    volumes={OUTPUT_DIR: output_volume},  # Only need output volume for reading results
)
class API:
    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
    def result(self, call_id: str):
        """
        Poll for video generation results using call_id.
        Returns 202 if still processing, 200 with video if complete.
        """
        import modal
        from fastapi.responses import Response
        import fastapi.responses
        
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            # Try to get result with no timeout
            output_filename = function_call.get(timeout=0)
            
            # Read the file from the volume
            video_bytes = b"".join(output_volume.read_file(output_filename))
            
            # Return the video bytes
            return Response(
                content=video_bytes,
                media_type="video/mp4",
                headers={"Content-Disposition": f"attachment; filename={output_filename}"}
            )
        except TimeoutError:
            # Still processing - return HTTP 202 Accepted with no body
            return fastapi.responses.Response(status_code=202)

    @modal.fastapi_endpoint(method="HEAD", requires_proxy_auth=True)
    def result_head(self, call_id: str):
        """
        HEAD request for polling status without downloading video body.
        Returns 202 if still processing, 200 if ready.
        """
        import modal
        import fastapi.responses
        
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            # Try to get result with no timeout
            function_call.get(timeout=0)
            # If successful, return 200 with video headers but no body
            return fastapi.responses.Response(
                status_code=200,
                media_type="video/mp4"
            )
        except TimeoutError:
            # Still processing - return HTTP 202 Accepted with no body
            return fastapi.responses.Response(status_code=202)

# --- GPU Model Class ---
@app.cls(
    gpu="L40S",
    enable_memory_snapshot=True, # new gpu snapshot feature: https://modal.com/blog/gpu-mem-snapshots
    experimental_options={"enable_gpu_snapshot": True},
    image=image,
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    scaledown_window=2, #scale down after 2 seconds. default is 60 seconds. for testing, just scale down for now
    timeout=2700,  # 45 minutes timeout for large model downloads and initialization
)
class Model:
    def _download_and_validate(self, url: str, expected_types: list[str]) -> bytes:
        """Download content from URL and validate file type."""
        import magic
        from fastapi import HTTPException
        import urllib.request
        
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download from URL {url}: {e}")
        
        # Validate file type
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_buffer(content)
        if detected_mime not in expected_types:
            expected_str = ", ".join(expected_types)
            raise HTTPException(status_code=400, detail=f"Invalid file type. Expected {expected_str}, but got {detected_mime}.")
        
        return content

    @modal.enter()  # Modal handles long initialization appropriately
    def initialize_model(self):
        """Initialize the model and audio components when container starts."""
        # Add module paths for imports
        import sys
        from pathlib import Path
        sys.path.extend(["/root", "/root/infinitetalk"])
        
        from huggingface_hub import snapshot_download

        print("--- Container starting. Initializing model... ---")

        try:
            # --- Download models if not present using huggingface_hub ---
            model_root = Path(MODEL_DIR)
            
            from huggingface_hub import hf_hub_download
            
            
            # Helper function to download files with proper error handling
            def download_file(
                repo_id: str,
                filename: str,
                local_path: Path,
                revision: str = None,
                description: str = None,
                subfolder: str | None = None,
            ) -> None:
                """Download a single file with error handling and logging."""
                relative_path = Path(filename)
                if subfolder:
                    relative_path = Path(subfolder) / relative_path
                download_path = local_path.parent / relative_path

                if download_path.exists():
                    print(f"--- {description or filename} already present ---")
                    return
                
                download_path.parent.mkdir(parents=True, exist_ok=True)

                print(f"--- Downloading {description or filename}... ---")
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                        local_dir=local_path.parent,
                        subfolder=subfolder,
                    )
                    print(f"--- {description or filename} downloaded successfully ---")
                except Exception as e:
                    raise RuntimeError(f"Failed to download {description or filename} from {repo_id}: {e}")
            
            def download_repo(repo_id: str, local_dir: Path, check_file: str, description: str) -> None:
                """Download entire repository with error handling and logging."""
                check_path = local_dir / check_file
                if check_path.exists():
                    print(f"--- {description} already present ---")
                    return
                
                print(f"--- Downloading {description}... ---")
                try:
                    snapshot_download(repo_id=repo_id, local_dir=local_dir)
                    print(f"--- {description} downloaded successfully ---")
                except Exception as e:
                    raise RuntimeError(f"Failed to download {description} from {repo_id}: {e}")

            try:
                
                # Create necessary directories
                # (model_root / "quant_models").mkdir(parents=True, exist_ok=True)
                
                # Download full Wan model for non-quantized operation with LoRA support
                wan_model_dir = model_root / "Wan2.1-I2V-14B-480P"
                wan_model_dir.mkdir(exist_ok=True)
                
                # Essential Wan model files (config and encoders)
                wan_base_files = [
                    ("config.json", "Wan model config"),
                    ("models_t5_umt5-xxl-enc-bf16.pth", "T5 text encoder weights"),
                    ("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", "CLIP vision encoder weights"),
                    ("Wan2.1_VAE.pth", "VAE weights")
                ]
                
                for filename, description in wan_base_files:
                    download_file(
                        repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                        filename=filename,
                        local_path=wan_model_dir / filename,
                        description=description
                    )
                
                # Download full diffusion model (7 shards) - required for non-quantized operation
                wan_diffusion_files = [
                    ("diffusion_pytorch_model-00001-of-00007.safetensors", "Wan diffusion model shard 1/7"),
                    ("diffusion_pytorch_model-00002-of-00007.safetensors", "Wan diffusion model shard 2/7"),
                    ("diffusion_pytorch_model-00003-of-00007.safetensors", "Wan diffusion model shard 3/7"),
                    ("diffusion_pytorch_model-00004-of-00007.safetensors", "Wan diffusion model shard 4/7"),
                    ("diffusion_pytorch_model-00005-of-00007.safetensors", "Wan diffusion model shard 5/7"),
                    ("diffusion_pytorch_model-00006-of-00007.safetensors", "Wan diffusion model shard 6/7"),
                    ("diffusion_pytorch_model-00007-of-00007.safetensors", "Wan diffusion model shard 7/7")
                ]
                
                for filename, description in wan_diffusion_files:
                    download_file(
                        repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                        filename=filename,
                        local_path=wan_model_dir / filename,
                        description=description
                    )
                
                # Download tokenizer directories (need full structure)
                tokenizer_dirs = [
                    ("google/umt5-xxl", "T5 tokenizer"),
                    ("xlm-roberta-large", "CLIP tokenizer")
                ]
                
                for subdir, description in tokenizer_dirs:
                    tokenizer_path = wan_model_dir / subdir
                    if not (tokenizer_path / "tokenizer_config.json").exists():
                        print(f"--- Downloading {description}... ---")
                        try:
                            snapshot_download(
                                repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
                                allow_patterns=[f"{subdir}/*"],
                                local_dir=wan_model_dir
                            )
                            print(f"--- {description} downloaded successfully ---")
                        except Exception as e:
                            raise RuntimeError(f"Failed to download {description}: {e}")
                    else:
                        print(f"--- {description} already present ---")
                
                # Download chinese wav2vec2 model (need full structure for from_pretrained)
                wav2vec_model_dir = model_root / "chinese-wav2vec2-base"
                download_repo(
                    repo_id="TencentGameMate/chinese-wav2vec2-base",
                    local_dir=wav2vec_model_dir,
                    check_file="config.json",
                    description="Chinese wav2vec2-base model"
                )
                
                # Download specific wav2vec safetensors file from PR revision
                download_file(
                    repo_id="TencentGameMate/chinese-wav2vec2-base",
                    filename="model.safetensors",
                    local_path=wav2vec_model_dir / "model.safetensors",
                    revision="refs/pr/1",
                    description="wav2vec safetensors file"
                )
                
                # Download InfiniteTalk weights
                infinitetalk_dir = model_root / "InfiniteTalk" / "single"
                infinitetalk_dir.mkdir(parents=True, exist_ok=True)
                download_file(
                    repo_id="MeiGen-AI/InfiniteTalk",
                    filename="single/infinitetalk.safetensors",
                    local_path=infinitetalk_dir / "infinitetalk.safetensors",
                    description="InfiniteTalk weights file",
                )

                # Skip quantized model downloads since we're using non-quantized models
                # quant_files = [
                #     ("quant_models/infinitetalk_single_fp8.safetensors", "fp8 quantized model"),
                #     ("quant_models/infinitetalk_single_fp8.json", "quantization mapping for fp8 model"),
                #     ("quant_models/t5_fp8.safetensors", "T5 fp8 quantized model"),
                #     ("quant_models/t5_map_fp8.json", "T5 quantization mapping for fp8 model"),
                # ]

                # for filename, description in quant_files:
                #     download_file(
                #         repo_id="MeiGen-AI/InfiniteTalk",
                #         filename=filename,
                #         local_path=model_root / filename,
                #         description=description,
                #     )

                # Download FusioniX LoRA weights (will create FusionX_LoRa directory)
                download_file(
                    repo_id="vrgamedevgirl84/Wan14BT2VFusioniX",
                    filename="Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                    local_path=model_root / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
                    subfolder="FusionX_LoRa",
                    description="FusioniX LoRA weights",
                )
                
                print("--- All required files present. Committing to volume. ---")
                model_volume.commit()
                print("--- Volume committed. ---")
                
            except Exception as download_error:
                print(f"--- Failed to download models: {download_error} ---")
                print("--- This repository may be private/gated or require authentication ---")
                raise RuntimeError(f"Cannot access required models: {download_error}")

            print("--- Model downloads completed successfully. ---")
            print("--- Will initialize models when generate() is called. ---")

        except Exception as e:
            print(f"--- Error during initialization: {e} ---")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()  
    def _generate_video(self, image: bytes, audio1: bytes, prompt: str | None = None) -> str:
        """
        Internal method to generate video from image/video input and save it to the output volume.
        Returns the filename of the generated video.
        """
        import sys
        # Add the required directories to the Python path at runtime.
        # This is needed in every method that imports from the local InfiniteTalk dir.
        sys.path.extend(["/root", "/root/infinitetalk"])

        from PIL import Image as PILImage
        import io
        import tempfile
        import time
        from types import SimpleNamespace
        import uuid

        t0 = time.time()
        
        # --- Prepare Inputs ---
        # Determine if input is image or video based on content
        import magic
        mime = magic.Magic(mime=True)
        detected_mime = mime.from_buffer(image)
        
        if detected_mime.startswith('video/'):
            # Handle video input
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_file.write(image)
                image_path = tmp_file.name
        else:
            # Handle image input
            source_image = PILImage.open(io.BytesIO(image)).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_image:
                source_image.save(tmp_image.name, "JPEG")
                image_path = tmp_image.name

        # --- Save audio files directly - let pipeline handle processing ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio1:
            tmp_audio1.write(audio1)
            audio1_path = tmp_audio1.name
        
        # Create audio dictionary with file paths (not embeddings)
        cond_audio_dict = {"person1": audio1_path}

        # --- Create Input Data Structure ---
        input_data = {
            "cond_video": image_path,  # Pass the file path (accepts both images and videos)
            "cond_audio": cond_audio_dict,
            "prompt": prompt or "a person is talking", # Use provided prompt or a default
        }

        print("--- Audio files prepared, using generate_infinitetalk.py directly ---")

        import json
        import os
        import shutil
        from pathlib import Path
        from infinitetalk.generate_infinitetalk import generate
        
        # Create input JSON in the format expected by generate_infinitetalk.py
        input_json_data = {
            "prompt": input_data["prompt"],
            "cond_video": input_data["cond_video"],
            "cond_audio": input_data["cond_audio"]
        }
        
        # Add audio_type for multi-speaker
        if len(input_data["cond_audio"]) > 1:
            input_json_data["audio_type"] = "add"
        
        # Save input JSON to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp_json:
            json.dump(input_json_data, tmp_json)
            input_json_path = tmp_json.name
        
        # Calculate appropriate frame_num based on audio duration(s)
        import librosa
        total_audio_duration = librosa.get_duration(path=audio1_path)
        print(f"--- Single audio duration: {total_audio_duration:.2f}s ---")
        
        # Convert to frames: 25 fps, embedding_length must be > frame_num
        # Audio embedding is exactly 25 frames per second
        audio_embedding_frames = int(total_audio_duration * 25)
        # Leave some buffer to ensure we don't exceed embedding length
        max_possible_frames = max(5, audio_embedding_frames - 5)  # 5 frame safety buffer
        # Use minimum of pipeline max (1000) and what audio can support
        calculated_frame_num = min(1000, max_possible_frames)
        # Ensure it follows the 4n+1 pattern required by the model
        n = (calculated_frame_num - 1) // 4
        frame_num = 4 * n + 1
        
        # Final safety check: ensure frame_num doesn't exceed audio embedding length
        if frame_num >= audio_embedding_frames:
            # Recalculate with more conservative approach
            safe_frames = audio_embedding_frames - 10  # 10 frame safety buffer
            n = max(1, (safe_frames - 1) // 4)  # Ensure at least n=1 
            frame_num = 4 * n + 1
        
        # Determine mode and frame settings based on total length needed
        if calculated_frame_num > 81:
            # Long video: use streaming mode
            mode = "streaming"
            chunk_frame_num = 81  # Standard chunk size for streaming
            max_frame_num = frame_num  # Total length we want to generate
        else:
            # Short video: use clip mode  
            mode = "clip"
            chunk_frame_num = frame_num  # Generate exactly what we need in one go
            max_frame_num = frame_num  # Same as chunk for clip mode
        
        print(f"--- Audio duration: {total_audio_duration:.2f}s, embedding frames: {audio_embedding_frames} ---")
        print(f"--- Total frames needed: {frame_num}, chunk size: {chunk_frame_num}, max: {max_frame_num}, mode: {mode} ---")
        
        # Create output directory and filename
        output_filename = f"{uuid.uuid4()}"
        output_dir = Path(OUTPUT_DIR)
        model_root = Path(MODEL_DIR)
        
        # Create args object that mimics command line arguments  
        args = SimpleNamespace(
            task="infinitetalk-14B",
            size="infinitetalk-480",
            frame_num=chunk_frame_num,  # Chunk size for each iteration
            max_frame_num=max_frame_num,  # Total target length
            ckpt_dir=str(model_root / "Wan2.1-I2V-14B-480P"),
            infinitetalk_dir=str(model_root / "InfiniteTalk" / "single" / "single" / "infinitetalk.safetensors"),
            quant_dir=None,  # Using non-quantized model for LoRA support
            wav2vec_dir=str(model_root / "chinese-wav2vec2-base"),
            dit_path=None,
            lora_dir=[str(model_root / "FusionX_LoRa" / "FusionX_LoRa" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors")],
            lora_scale=[1.0],
            offload_model=False,
            ulysses_size=1,
            ring_size=1,
            t5_fsdp=False,
            t5_cpu=False,
            dit_fsdp=False,
            save_file=str(output_dir / output_filename),
            audio_save_dir=str(output_dir / "temp_audio"),
            base_seed=42,
            input_json=input_json_path,
            motion_frame=25,
            mode=mode,
            sample_steps=8,
            sample_shift=3.0,
            sample_text_guide_scale=1.0,
            sample_audio_guide_scale=6.0, # under 6 we lose some lip sync but as we go higher image gets unstable.
            num_persistent_param_in_dit=500000000,
            audio_mode="localfile",
            use_teacache=True,
            teacache_thresh=0.3,
            use_apg=True,
            apg_momentum=-0.75,
            apg_norm_threshold=55,
            color_correction_strength=0.2,
            scene_seg=False,
            quant=None,  # Using non-quantized model for LoRA support
        )
        
        # Set environment variables for single GPU setup
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        
        # Ensure audio save directory exists
        audio_save_dir = Path(args.audio_save_dir)
        audio_save_dir.mkdir(parents=True, exist_ok=True)
        
        print("--- Generating video using original generate_infinitetalk.py logic ---")
        print(f"--- Input JSON: {input_json_data} ---")
        print(f"--- Audio save dir: {audio_save_dir} ---")
        
        # Call the original generate function
        generate(args)
        
        # The generate function saves the video with .mp4 extension
        generated_file = f"{args.save_file}.mp4"
        final_output_path = output_dir / f"{output_filename}.mp4"
        
        # Move the generated file to our expected location
        if os.path.exists(generated_file):
            os.rename(generated_file, final_output_path)
        
        output_volume.commit()
        
        # Clean up input JSON and temp audio directory
        os.unlink(input_json_path)
        temp_audio_dir = output_dir / "temp_audio"
        if temp_audio_dir.exists():
            shutil.rmtree(temp_audio_dir)
        
        print(f"--- Generation complete in {time.time() - t0:.2f}s ---")
        
        # --- Cleanup temporary files ---
        os.unlink(audio1_path)
        os.unlink(image_path) # Clean up the temporary image/video file

        return output_filename + ".mp4"  # Return the final filename with .mp4 extension

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def submit(self, request: "GenerationRequest"):
        """
        Submit a video generation job and return call_id for polling.
        Following Modal's recommended polling pattern for long-running tasks.
        """
        # Download and validate inputs
        image_bytes = self._download_and_validate(request.image, [
            # Image formats
            "image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff",
            # Video formats
            "video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", 
            "video/webm", "video/x-ms-wmv", "video/x-flv"
        ])
        audio1_bytes = self._download_and_validate(request.audio1, ["audio/mpeg", "audio/wav", "audio/x-wav"])

        # Spawn the generation job and return call_id
        call = self._generate_video.spawn(
            image_bytes, audio1_bytes, request.prompt
        )
        
        return {"call_id": call.object_id}

# --- Local Testing CLI ---
@app.local_entrypoint()
def main(
    image_path: str,
    audio1_path: str,
    prompt: str = None,
    output_path: str = "outputs/test.mp4",
):
    """
    A local CLI to generate an InfiniteTalk video from local files or URLs.

    Example:
    modal run app.py --image-path "url/to/image.png" --audio1-path "url/to/audio1.wav"
    """
    import base64
    import urllib.request

    print(f"--- Starting generation for {image_path} ---")
    print(f"--- Current working directory: {os.getcwd()} ---")
    print(f"--- Output path: {output_path} ---")
    
    def _read_input(path: str) -> bytes:
        if not path:
            return None
        if path.startswith(("http://", "https://")):
            return urllib.request.urlopen(path).read()
        else:
            with open(path, "rb") as f:
                return f.read()

    # --- Read inputs (validation only happens on remote Modal containers) ---
    image_bytes = _read_input(image_path)
    audio1_bytes = _read_input(audio1_path)
    
    # --- Run model ---
    # We call the internal _generate_video method remotely like the FastAPI endpoint.
    model = Model()
    output_filename = model._generate_video.remote(
        image_bytes, audio1_bytes, prompt
    )

    # --- Save output ---
    print(f"--- Reading '{output_filename}' from volume... ---")
    video_bytes = b"".join(output_volume.read_file(output_filename))
    
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    
    print(f"🎉 --- Video saved to {output_path} ---") 
