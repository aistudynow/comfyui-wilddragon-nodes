import os
import numpy as np
import torch
from PIL import Image
from typing import Optional


class FaithDiffRestorer:
    """
    Wrapper for FaithDiff SDXL-based face restoration model.
    Model: https://github.com/JyChen9811/FaithDiff
    Downloads the 10GB FaithDiff.bin model automatically.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize FaithDiff face restoration model.

        Args:
            model_path: Path to FaithDiff.bin model file. If None, uses default cache location.
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.pipe = None

        if model_path is None:
            cache_dir = os.path.expanduser("~/.cache/faithdiff")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, "FaithDiff.bin")

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the FaithDiff SDXL pipeline and model weights."""
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            import torch

            print(f"[FaithDiff] Loading model on {self.device}...")

            if not os.path.exists(self.model_path):
                print(f"[FaithDiff] Model not found at {self.model_path}")
                print("[FaithDiff] Downloading FaithDiff.bin (10GB) from HuggingFace...")
                self._download_model()

            self.pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                use_safetensors=False,
            )

            if self.device == 'cuda':
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe = self.pipe.to(self.device)

            self.pipe.enable_attention_slicing()

            print("[FaithDiff] Model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "FaithDiff requires: pip install diffusers transformers accelerate safetensors"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load FaithDiff model: {e}") from e

    def _download_model(self):
        """Download FaithDiff.bin from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download

            print("[FaithDiff] Downloading from HuggingFace (this may take a while)...")
            downloaded_path = hf_hub_download(
                repo_id="jychen9811/FaithDiff",
                filename="FaithDiff.bin",
                cache_dir=os.path.dirname(self.model_path),
                resume_download=True,
            )

            import shutil
            shutil.copy(downloaded_path, self.model_path)
            print(f"[FaithDiff] Model downloaded to {self.model_path}")

        except ImportError:
            raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Failed to download FaithDiff model: {e}") from e

    def restore(self, face_image: Image.Image, strength: float = 0.8) -> np.ndarray:
        """
        Restore a blurry face image using FaithDiff.

        Args:
            face_image: PIL Image of the face to restore
            strength: Denoising strength (0.0-1.0), higher = more restoration

        Returns:
            Restored face as numpy array (RGB)
        """
        if self.pipe is None:
            raise RuntimeError("FaithDiff model not loaded")

        try:
            width, height = face_image.size

            if width > 1024 or height > 1024:
                scale = min(1024 / width, 1024 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_image = face_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            prompt = "high quality, detailed face, sharp focus, professional photo"
            negative_prompt = "blurry, low quality, distorted, disfigured, bad anatomy"

            denoising_strength = max(0.3, min(0.95, strength * 0.7 + 0.25))

            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=face_image,
                    strength=denoising_strength,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                ).images[0]

            if result.size != (width, height):
                result = result.resize((width, height), Image.Resampling.LANCZOS)

            return np.array(result)

        except Exception as e:
            print(f"[FaithDiff] Restoration error: {e}")
            return np.array(face_image)

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if self.pipe is not None:
            del self.pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
