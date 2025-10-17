import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths  # provided by ComfyUI
except ImportError as exc:  # pragma: no cover - ComfyUI runtime provides this
    raise ImportError(
        "The FaithDiff restorer must run inside ComfyUI (folder_paths module missing)."
    ) from exc


@dataclass
class _FaithDiffPaths:
    """Container for resolved FaithDiff asset paths."""

    root: str
    checkpoint: str
    base_model: Optional[str]
    vae: Optional[str]


@dataclass
class _FaithDiffComponents:
    """Resolved objects needed to build the FaithDiff pipeline."""

    text_encoder: Optional[Any] = None
    text_encoder_2: Optional[Any] = None
    tokenizer: Optional[Any] = None
    tokenizer_2: Optional[Any] = None
    vae: Optional[Any] = None
    unet: Optional[Any] = None

    def missing(self) -> Iterable[str]:
        for field_name in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
            if getattr(self, field_name) is None:
                yield field_name

    def merge_missing(self, other: "_FaithDiffComponents") -> "_FaithDiffComponents":
        for field_name in (
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "vae",
            "unet",
        ):
            if getattr(self, field_name) is None:
                setattr(self, field_name, getattr(other, field_name))
        return self


def _find_first_existing(paths: Tuple[str, ...]) -> Optional[str]:
    for candidate in paths:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _normalize_vae(vae: Optional[Any]) -> Optional[Any]:
    if vae is None:
        return None
    if hasattr(vae, "vae"):
        return vae.vae
    return vae


def _normalize_unet(model: Optional[Any]) -> Optional[Any]:
    if model is None:
        return None
    # ComfyUI wraps the UNet inside "model" or "diffusion_model" attributes.
    for attr in ("model", "diffusion_model", "inner_model"):
        if hasattr(model, attr):
            model = getattr(model, attr)
    return model


def _extract_clip_components(clip_obj: Any) -> _FaithDiffComponents:
    """Attempt to pull diffusers-compatible components from a ComfyUI CLIP object."""

    def _extract(candidate: Any) -> _FaithDiffComponents:
        if candidate is None:
            return _FaithDiffComponents()

        if isinstance(candidate, dict):
            return _FaithDiffComponents(
                text_encoder=candidate.get("text_encoder"),
                text_encoder_2=candidate.get("text_encoder_2"),
                tokenizer=candidate.get("tokenizer"),
                tokenizer_2=candidate.get("tokenizer_2"),
            )

        return _FaithDiffComponents(
            text_encoder=getattr(candidate, "text_encoder", None),
            text_encoder_2=getattr(candidate, "text_encoder_2", None),
            tokenizer=getattr(candidate, "tokenizer", None),
            tokenizer_2=getattr(candidate, "tokenizer_2", None),
        )

    components = _FaithDiffComponents()
    for candidate in (
        clip_obj,
        getattr(clip_obj, "clip", None),
        getattr(clip_obj, "cond_stage_model", None),
        getattr(clip_obj, "model", None),
    ):
        components.merge_missing(_extract(candidate))

    # Some CLIP wrappers expose nested "model" attributes with the actual encoders.
    if components.text_encoder is None and hasattr(clip_obj, "text_encoder"):  # pragma: no cover
        components.text_encoder = clip_obj.text_encoder
    if components.text_encoder_2 is None and hasattr(clip_obj, "text_encoder_2"):  # pragma: no cover
        components.text_encoder_2 = clip_obj.text_encoder_2

    return components


def _resolve_model_paths(
    explicit_model_path: Optional[str],
    explicit_base_model_path: Optional[str],
    explicit_vae_path: Optional[str],
) -> _FaithDiffPaths:
    """Resolve FaithDiff asset locations inside ComfyUI's model directory."""

    models_root = os.path.join(folder_paths.models_dir, "widdragon", "face")
    os.makedirs(models_root, exist_ok=True)

    checkpoint = (
        explicit_model_path
        if explicit_model_path and os.path.exists(explicit_model_path)
        else os.path.join(models_root, "FaithDiff.bin")
    )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"FaithDiff checkpoint not found. Expected it at '{checkpoint}'.\n"
            "Download FaithDiff.bin from https://huggingface.co/jychen9811/FaithDiff "
            "and place it in ComfyUI/models/widdragon/face."
        )

    base_model: Optional[str] = None
    if explicit_base_model_path:
        if os.path.isdir(explicit_base_model_path):
            base_model = explicit_base_model_path
        else:
            raise FileNotFoundError(
                f"Provided FaithDiff base model path '{explicit_base_model_path}' does not exist or is not a directory."
            )

    if base_model is None:
        # Search for an SDXL base directory with text encoders + tokenizers
        def _looks_like_sdxl_base(path: str) -> bool:
            return all(
                os.path.isdir(os.path.join(path, sub))
                for sub in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2")
            )

        candidates = []
        for name in (
            "Real_4_SDXL",
            "sdxl_base",
            "SDXL",
            "stable-diffusion-xl-base-1.0",
        ):
            candidates.append(os.path.join(models_root, name))

        # Also scan direct children of models_root
        for entry in os.listdir(models_root):
            path = os.path.join(models_root, entry)
            if os.path.isdir(path):
                candidates.append(path)

        for path in candidates:
            if path and os.path.isdir(path) and _looks_like_sdxl_base(path):
                base_model = path
                break

    # Optional dedicated VAE folder. Fall back to base model's VAE subfolder if it exists.
    vae_path: Optional[str] = None
    if explicit_vae_path:
        if os.path.isdir(explicit_vae_path):
            vae_path = explicit_vae_path
        else:
            raise FileNotFoundError(
                f"Provided FaithDiff VAE path '{explicit_vae_path}' does not exist or is not a directory."
            )

    if vae_path is None and base_model is not None:
        vae_candidates = (
            os.path.join(models_root, "VAE_FP16"),
            os.path.join(base_model, "vae"),
        )
        vae_path = _find_first_existing(vae_candidates)

    return _FaithDiffPaths(root=models_root, checkpoint=checkpoint, base_model=base_model, vae=vae_path)


class FaithDiffRestorer:
    """
    Wrapper for FaithDiff SDXL-based face restoration model.

    The model assets are expected inside ``ComfyUI/models/widdragon/face``::

        FaithDiff.bin                # main FaithDiff weights
        Real_4_SDXL/                 # SDXL base model (text encoders, tokenizers, etc.)
        VAE_FP16/ (optional)         # dedicated VAE checkpoint (falls back to base VAE)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        base_model_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        basic_pipe: Optional[Tuple[Any, Any, Any, Any, Any]] = None,
        model: Optional[Any] = None,
        clip: Optional[Any] = None,
        vae: Optional[Any] = None,
        text_encoder: Optional[Any] = None,
        text_encoder_2: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        tokenizer_2: Optional[Any] = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.paths: Optional[_FaithDiffPaths] = None
        self._checkpoint_path: Optional[str] = None

        component_overrides = _FaithDiffComponents(
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            vae=_normalize_vae(vae),
            unet=_normalize_unet(model),
        )

        if basic_pipe is not None:
            try:
                basic_model, basic_clip, basic_vae, _, _ = basic_pipe
            except Exception as exc:  # pragma: no cover - defensive against unexpected shapes
                raise ValueError(f"Invalid FaithDiff basic pipe tuple: {exc}") from exc

            component_overrides.merge_missing(
                _FaithDiffComponents(
                    vae=_normalize_vae(basic_vae),
                    unet=_normalize_unet(basic_model),
                )
            )
            clip = clip or basic_clip

        clip_components = _FaithDiffComponents()
        if clip is not None:
            clip_components = _extract_clip_components(clip)

        component_overrides.merge_missing(clip_components)

        needs_paths = any(component_overrides.missing())
        if needs_paths:
            self.paths = _resolve_model_paths(model_path, base_model_path, vae_path)
            if self.paths.base_model is None:
                missing = ", ".join(component_overrides.missing())
                raise FileNotFoundError(
                    "Could not resolve SDXL base model to supply missing FaithDiff components: "
                    f"{missing}. Provide a base model directory via the node inputs or ensure "
                    "ComfyUI/models/widdragon/face contains an SDXL diffusers folder."
                )
        else:
            # Even if all components are provided we still need a checkpoint path.
            self.paths = _resolve_model_paths(model_path, base_model_path, vae_path)

        if self.paths:
            self._checkpoint_path = self.paths.checkpoint
        else:
            self._checkpoint_path = model_path

        self._load_model(component_overrides)

    def _load_model(self, overrides: _FaithDiffComponents):
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
            from transformers import (
                CLIPTextModel,
                CLIPTextModelWithProjection,
                CLIPTokenizer,
            )

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            components = overrides

            if self.paths and self.paths.base_model:
                print(f"[FaithDiff] Loading SDXL components from {self.paths.base_model}")

                base_components = _FaithDiffComponents(
                    text_encoder=CLIPTextModel.from_pretrained(
                        self.paths.base_model,
                        subfolder="text_encoder",
                        torch_dtype=dtype,
                    ),
                    text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
                        self.paths.base_model,
                        subfolder="text_encoder_2",
                        torch_dtype=dtype,
                    ),
                    tokenizer=CLIPTokenizer.from_pretrained(
                        self.paths.base_model,
                        subfolder="tokenizer",
                    ),
                    tokenizer_2=CLIPTokenizer.from_pretrained(
                        self.paths.base_model,
                        subfolder="tokenizer_2",
                    ),
                )

                if self.paths.vae is None:
                    base_components.vae = AutoencoderKL.from_pretrained(
                        self.paths.base_model,
                        subfolder="vae",
                        torch_dtype=dtype,
                    )
                elif os.path.isdir(self.paths.vae):
                    base_components.vae = AutoencoderKL.from_pretrained(
                        self.paths.vae,
                        torch_dtype=dtype,
                    )
                else:
                    raise FileNotFoundError(
                        f"VAE path '{self.paths.vae}' does not exist."
                    )

                components.merge_missing(base_components)

            missing = list(components.missing())
            if missing:
                raise RuntimeError(
                    "FaithDiff requires the following components but they could not be resolved: "
                    + ", ".join(missing)
                )

            checkpoint_path = self._checkpoint_path or (self.paths.checkpoint if self.paths else None)
            if not checkpoint_path:
                raise RuntimeError("FaithDiff checkpoint path could not be resolved")

            print(f"[FaithDiff] Loading checkpoint {checkpoint_path}")
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                checkpoint_path,
                torch_dtype=dtype,
                text_encoder=components.text_encoder,
                text_encoder_2=components.text_encoder_2,
                tokenizer=components.tokenizer,
                tokenizer_2=components.tokenizer_2,
                vae=components.vae,
                unet=components.unet,
                use_safetensors=checkpoint_path.endswith(".safetensors"),
            )

            self.pipe.to(self.device)
            if self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            self.pipe.enable_attention_slicing()

            print("[FaithDiff] Model loaded successfully")

        except ImportError as exc:
            raise ImportError(
                "FaithDiff requires diffusers>=0.27, transformers>=4.40 and accelerate."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FaithDiff model: {exc}"
            ) from exc

    def restore(self, face_image: Image.Image, strength: float = 0.8) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("FaithDiff model not loaded")

        width, height = face_image.size
        if width > 1024 or height > 1024:
            scale = min(1024 / width, 1024 / height)
            new_size = (max(64, int(width * scale)), max(64, int(height * scale)))
            face_image = face_image.resize(new_size, Image.Resampling.LANCZOS)

        prompt = "sharp detailed face, 8k portrait photography, natural skin texture"
        negative_prompt = "blurry, distorted, disfigured, low quality, extra limbs"
        denoising_strength = float(np.clip(strength * 0.7 + 0.25, 0.3, 0.95))

        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=face_image,
                    strength=denoising_strength,
                    num_inference_steps=25,
                    guidance_scale=7.0,
                ).images[0]
        except Exception as exc:
            print(f"[FaithDiff] Restoration error: {exc}")
            return np.array(face_image)

        if result.size != (width, height):
            result = result.resize((width, height), Image.Resampling.LANCZOS)

        return np.array(result)

    def __del__(self):
        if self.pipe is not None:
            try:
                del self.pipe
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()