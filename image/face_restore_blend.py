# wilddragon_nodes/image/face_restore_blend.py
# Node: 🐉 Face Restore & Blend

import json
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms.v2 import ToTensor, ToPILImage
import cv2

_to_tensor = ToTensor()
_to_image = ToPILImage()


class WD_FaceRestoreBlend:
    """
    Wilddragon • Face Restore & Blend
    - Takes cropped face images and enhances them using face restoration models
    - Seamlessly blends enhanced faces back into original frames
    - Supports multiple restoration backends and blending modes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_faces": ("IMAGE",),
                "face_mask": ("MASK",),
                "bboxes_json": ("STRING", {"default": ""}),
                "restoration_model": (["none", "codeformer", "gfpgan", "realesrgan", "faithdiff"], {"default": "none"}),
                "restoration_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_mode": (["alpha", "poisson", "feather"], {"default": "feather"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "feather_amount": ("INT", {"default": 15, "min": 0, "max": 100, "step": 1}),
                "color_match": ("BOOLEAN", {"default": True, "label_on": "match", "label_off": "no match"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_images",)
    FUNCTION = "execute"
    CATEGORY = "Wilddragon/Image"

    def __init__(self):
        self._restoration_model = None
        self._model_type = None

    def _load_restoration_model(self, model_name: str):
        """Load face restoration model on demand."""
        if model_name == "none":
            return None

        if self._model_type == model_name and self._restoration_model is not None:
            return self._restoration_model

        print(f"[Face Restore & Blend] Loading {model_name} model...")

        try:
            if model_name == "codeformer":
                try:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                    from codeformer import CodeFormer

                    # Initialize CodeFormer
                    net = CodeFormer(
                        dim_embd=512,
                        codebook_size=1024,
                        n_head=8,
                        n_layers=9,
                        connect_list=['32', '64', '128', '256']
                    ).eval()

                    self._restoration_model = net
                    self._model_type = model_name
                    print("[Face Restore & Blend] CodeFormer loaded successfully")
                except ImportError:
                    print("[Face Restore & Blend] CodeFormer not available. Install with: pip install codeformer")
                    return None

            elif model_name == "gfpgan":
                try:
                    from gfpgan import GFPGANer

                    self._restoration_model = GFPGANer(
                        model_path='GFPGANv1.4.pth',
                        upscale=1,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=None
                    )
                    self._model_type = model_name
                    print("[Face Restore & Blend] GFPGAN loaded successfully")
                except ImportError:
                    print("[Face Restore & Blend] GFGAN not available. Install with: pip install gfpgan")
                    return None

            elif model_name == "realesrgan":
                try:
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer

                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                    self._restoration_model = RealESRGANer(
                        scale=2,
                        model_path='RealESRGAN_x2plus.pth',
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=False
                    )
                    self._model_type = model_name
                    print("[Face Restore & Blend] RealESRGAN loaded successfully")
                except ImportError:
                    print("[Face Restore & Blend] RealESRGAN not available. Install with: pip install realesrgan")
                    return None

            elif model_name == "faithdiff":
                try:
                    from .faithdiff_wrapper import FaithDiffRestorer

                    self._restoration_model = FaithDiffRestorer()
                    self._model_type = model_name
                    print("[Face Restore & Blend] FaithDiff loaded successfully")
                except ImportError as e:
                    print(f"[Face Restore & Blend] FaithDiff not available: {e}")
                    print("Install dependencies: pip install diffusers transformers accelerate")
                    return None
                except Exception as e:
                    print(f"[Face Restore & Blend] FaithDiff loading error: {e}")
                    return None

        except Exception as e:
            print(f"[Face Restore & Blend] Error loading model: {e}")
            return None

        return self._restoration_model

    def _enhance_face(self, face_pil: Image.Image, model_name: str, strength: float) -> Image.Image:
        """Apply face restoration to a single face image."""
        if model_name == "none":
            return face_pil

        model = self._load_restoration_model(model_name)
        if model is None:
            print("[Face Restore & Blend] Model not available, returning original")
            return face_pil

        try:
            face_np = np.array(face_pil)
            face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            if model_name == "codeformer":
                # CodeFormer expects normalized tensor input
                face_tensor = torch.from_numpy(face_bgr).float() / 255.0
                face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)

                with torch.no_grad():
                    output = model(face_tensor, w=strength)[0]

                restored_bgr = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            elif model_name == "gfpgan":
                _, _, restored_bgr = model.enhance(face_bgr, has_aligned=False, paste_back=True, weight=strength)

            elif model_name == "realesrgan":
                restored_bgr, _ = model.enhance(face_bgr, outscale=1)

            elif model_name == "faithdiff":
                restored_rgb = model.restore(face_pil, strength)
                restored_bgr = None

            if restored_bgr is not None:
                restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
                restored_pil = Image.fromarray(restored_rgb)
            elif restored_rgb is not None:
                restored_pil = Image.fromarray(restored_rgb) if isinstance(restored_rgb, np.ndarray) else restored_rgb
            else:
                return face_pil

            # Blend with original based on strength
            if strength < 1.0:
                restored_pil = Image.blend(face_pil, restored_pil, strength)

            return restored_pil

        except Exception as e:
            print(f"[Face Restore & Blend] Enhancement error: {e}")
            return face_pil

    def _match_color(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match color distribution of source to target using LAB color space."""
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

        for i in range(3):
            s_mean, s_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
            t_mean, t_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()

            if s_std > 0:
                source_lab[:, :, i] = ((source_lab[:, :, i] - s_mean) * (t_std / s_std)) + t_mean

        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)

    def _create_feathered_mask(self, mask: np.ndarray, feather_amount: int) -> np.ndarray:
        """Create a feathered alpha mask for smooth blending."""
        if feather_amount == 0:
            return mask

        mask_uint8 = (mask * 255).astype(np.uint8)

        # Apply Gaussian blur for feathering
        kernel_size = feather_amount * 2 + 1
        feathered = cv2.GaussianBlur(mask_uint8, (kernel_size, kernel_size), feather_amount / 2)

        return feathered.astype(np.float32) / 255.0

    def _poisson_blend(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray, center: tuple) -> np.ndarray:
        """Use OpenCV's seamless cloning for Poisson blending."""
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)

            result_bgr = cv2.seamlessClone(source_bgr, target_bgr, mask_uint8, center, cv2.NORMAL_CLONE)
            return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[Face Restore & Blend] Poisson blend failed: {e}, using alpha blend")
            return self._alpha_blend(source, target, mask)

    def _alpha_blend(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Simple alpha blending."""
        mask_3ch = np.stack([mask] * 3, axis=2)
        return (source * mask_3ch + target * (1 - mask_3ch)).astype(np.uint8)

    def _blend_face_into_frame(self, original: Image.Image, enhanced_face: Image.Image,
                               bbox: tuple, mask: np.ndarray, blend_mode: str,
                               feather_amount: int, color_match: bool) -> Image.Image:
        """Blend enhanced face back into original frame."""
        x1, y1, x2, y2 = bbox

        if x1 >= x2 or y1 >= y2:
            return original

        original_np = np.array(original)
        H, W = original_np.shape[:2]

        # Ensure bbox is within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            return original

        # Resize enhanced face to match bbox size
        enhanced_resized = enhanced_face.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        enhanced_np = np.array(enhanced_resized)

        # Extract original face region
        original_face_region = original_np[y1:y2, x1:x2].copy()

        # Color matching
        if color_match:
            enhanced_np = self._match_color(enhanced_np, original_face_region)

        # Prepare mask
        if mask.shape != (crop_h, crop_w):
            mask_resized = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_resized = mask

        # Apply feathering
        feathered_mask = self._create_feathered_mask(mask_resized, feather_amount)

        # Blend based on mode
        if blend_mode == "poisson":
            center = (x1 + crop_w // 2, y1 + crop_h // 2)
            result_np = original_np.copy()
            blended_region = self._poisson_blend(enhanced_np, original_face_region, feathered_mask, (crop_w // 2, crop_h // 2))
            result_np[y1:y2, x1:x2] = blended_region
        else:
            # Alpha or feather blend
            result_np = original_np.copy()
            blended_region = self._alpha_blend(enhanced_np, original_face_region, feathered_mask)
            result_np[y1:y2, x1:x2] = blended_region

        return Image.fromarray(result_np)

    def execute(
        self,
        original_images: torch.Tensor,
        cropped_faces: torch.Tensor,
        face_mask: torch.Tensor,
        bboxes_json: str,
        restoration_model: str,
        restoration_strength: float,
        blend_mode: str,
        blend_strength: float,
        feather_amount: int,
        color_match: bool,
    ):
        B_orig, H_orig, W_orig, C_orig = original_images.shape
        B_crop = cropped_faces.shape[0]

        if B_orig != B_crop:
            print(f"[Face Restore & Blend] Frame count mismatch: {B_orig} vs {B_crop}")
            return (original_images,)

        # Parse bboxes
        try:
            bboxes = json.loads(bboxes_json)
            if len(bboxes) != B_orig:
                print(f"[Face Restore & Blend] Bbox count mismatch: {len(bboxes)} vs {B_orig}")
                return (original_images,)
        except:
            print("[Face Restore & Blend] Failed to parse bboxes_json")
            return (original_images,)

        print(f"[Face Restore & Blend] Processing {B_orig} frames with model: {restoration_model}")

        result_frames = []

        for i in range(B_orig):
            # Convert to PIL
            original_pil = _to_image(original_images[i].permute(2, 0, 1))
            face_pil = _to_image(cropped_faces[i].permute(2, 0, 1))
            mask_np = face_mask[i].cpu().numpy()
            bbox = tuple(bboxes[i])

            # Enhance face
            if restoration_model != "none":
                enhanced_face = self._enhance_face(face_pil, restoration_model, restoration_strength)
            else:
                enhanced_face = face_pil

            # Blend back into original
            result_pil = self._blend_face_into_frame(
                original_pil,
                enhanced_face,
                bbox,
                mask_np,
                blend_mode,
                feather_amount,
                color_match
            )

            # Apply blend strength
            if blend_strength < 1.0:
                result_pil = Image.blend(original_pil, result_pil, blend_strength)

            result_tensor = _to_tensor(result_pil).permute(1, 2, 0)
            result_frames.append(result_tensor)

            if i % 30 == 0:
                print(f"  Processed {i}/{B_orig} frames")

        result_batch = torch.stack(result_frames, dim=0)

        print(f"[Face Restore & Blend] Completed processing {B_orig} frames")
        return (result_batch,)


NODE_CLASS_MAPPINGS = {
    "WD_FaceRestoreBlend": WD_FaceRestoreBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_FaceRestoreBlend": "🐉 Face Restore & Blend",
}
