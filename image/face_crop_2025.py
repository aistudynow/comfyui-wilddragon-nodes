# wilddragon_nodes/image/face_crop_2025.py
# Node: 🐉 Image Face Crop (2025)

import json
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision.transforms.v2 import ToTensor, ToPILImage

from ..utils.face_detectors import (
    get_detector,
    expand_bbox,
    eye_rotation_radians_from_kps,
)

_to_tensor = ToTensor()
_to_image = ToPILImage()


class WD_ImageFaceCrop2025:
    """
    Wilddragon • Image Face Crop (2025)
    - Backends: auto (InsightFace→RetinaFace), insightface, retinaface
    - Margin & aspect-aware crop, optional eye-alignment
    - Select faces: largest | confidence | index | all
    - Returns IMAGE (batch), MASK (crop or original space), FACE_DETECTED, BBOXES_JSON
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "backend": (["auto", "insightface", "retinaface"], {"default": "auto"}),
                "crop_width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
                "crop_height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 1}),
                "margin": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "align_rotation": ("BOOLEAN", {"default": True, "label_on": "align", "label_off": "no align"}),
                "selection": (["largest", "confidence", "index", "all"], {"default": "largest"}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "max_faces": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
                "mask_space": (["crop", "original"], {"default": "crop"}),
                "mask_feather": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "min_score": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "cache_detections": ("BOOLEAN", {"default": True, "label_on": "yes", "label_off": "no"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FACE_DETECTED", "BBOXES_JSON")
    FUNCTION = "execute"
    CATEGORY = "Wilddragon/Image"

    # -------------------- internals --------------------

    def _build_mask_crop(self, w: int, h: int, feather: int) -> torch.Tensor:
        mask = Image.new("L", (w, h), color=255)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        arr = np.asarray(mask, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def _build_mask_original(self, W: int, H: int, box: Tuple[int, int, int, int], feather: int) -> torch.Tensor:
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        base = Image.new("L", (W, H), color=0)
        rect = Image.new("L", (bw, bh), color=255)
        base.paste(rect, (x1, y1))
        if feather > 0:
            base = base.filter(ImageFilter.GaussianBlur(radius=feather))
        arr = np.asarray(base, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    def _select_faces(self, dets: List[dict], selection: str, index: int, max_faces: int) -> List[dict]:
        if not dets:
            return []
        if selection == "largest":
            dets = sorted(dets, key=lambda d: -(d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            return dets[:max_faces]
        elif selection == "confidence":
            dets = sorted(dets, key=lambda d: -float(d.get("score", 0.0)))
            return dets[:max_faces]
        elif selection == "index":
            idx = max(0, min(index, len(dets) - 1))
            return [dets[idx]]
        elif selection == "all":
            return dets[:max_faces]
        return dets[:max_faces]

    # -------------------- execute --------------------

    def __init__(self):
        self._detection_cache = {}
        self._last_bbox = None

    def _process_frame_fast(self, pil_img: Image.Image, box: Tuple[int, int, int, int],
                            crop_width: int, crop_height: int, align_rotation: bool,
                            target_aspect: float, kps=None) -> Image.Image:
        x1, y1, x2, y2 = box
        patch = pil_img.crop((x1, y1, x2, y2))

        if align_rotation and kps is not None:
            angle_rad = eye_rotation_radians_from_kps(kps)
            angle_deg = float(np.degrees(angle_rad))
            if abs(angle_deg) > 0.5:
                rotated = patch.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)
                rw, rh = rotated.size
                if (rw / rh) > target_aspect:
                    ch = rh
                    cw = int(round(target_aspect * ch))
                else:
                    cw = rw
                    ch = int(round(cw / target_aspect))
                cx, cy = rw // 2, rh // 2
                rx1 = max(0, cx - cw // 2)
                ry1 = max(0, cy - ch // 2)
                rx2 = min(rw, rx1 + cw)
                ry2 = min(rh, ry1 + ch)
                patch = rotated.crop((rx1, ry1, rx2, ry2))

        return patch.resize((crop_width, crop_height), Image.Resampling.LANCZOS)

    def execute(
        self,
        images: torch.Tensor,
        backend: str,
        crop_width: int,
        crop_height: int,
        margin: float,
        align_rotation: bool,
        selection: str,
        index: int,
        max_faces: int,
        mask_space: str,
        mask_feather: int,
        min_score: float,
        batch_size: int,
        cache_detections: bool,
    ):
        B, H, W, C = images.shape
        target_aspect = float(crop_width) / float(crop_height)
        detector = get_detector(backend)

        out_images: List[torch.Tensor] = []
        out_masks: List[torch.Tensor] = []
        all_boxes: List[Tuple[int, int, int, int]] = []

        print(f"Processing {B} frames with face detection...")

        cache_key = f"{backend}_{W}_{H}_{min_score}"
        use_cached = cache_detections and cache_key in self._detection_cache and self._last_bbox is not None

        for b in range(B):
            pil = _to_image(images[b].permute(2, 0, 1))

            if use_cached and b % 5 != 0:
                box = self._last_bbox
                kps = None
            else:
                try:
                    raw_dets = detector.detect(pil)
                except Exception:
                    raw_dets = []

                dets = [d for d in raw_dets if float(d.get("score", 1.0)) >= float(min_score)]
                selected = self._select_faces(dets, selection, index, max_faces)

                if not selected:
                    blank = Image.new("RGB", (crop_width, crop_height), color=(0, 0, 0))
                    out_images.append(_to_tensor(blank))
                    out_masks.append(torch.zeros((crop_height, crop_width), dtype=torch.float32))
                    all_boxes.append((0, 0, 0, 0))
                    continue

                det = selected[0]
                box = expand_bbox(tuple(det["bbox"]), margin, target_aspect, W, H)
                kps = det.get("kps")
                self._last_bbox = box

            cropped = self._process_frame_fast(pil, box, crop_width, crop_height,
                                                align_rotation, target_aspect, kps)
            out_images.append(_to_tensor(cropped))

            if mask_space == "original":
                mask = self._build_mask_original(W, H, box, mask_feather)
            else:
                mask = self._build_mask_crop(crop_width, crop_height, mask_feather)
            out_masks.append(mask)
            all_boxes.append(box)

            if b % 30 == 0:
                print(f"  Processed {b}/{B} frames")

        if cache_detections:
            self._detection_cache[cache_key] = True

        img_batch = torch.stack(out_images, dim=0).permute(0, 2, 3, 1)
        mask_batch = torch.stack(out_masks, dim=0)
        face_found = any((x2 - x1) > 0 and (y2 - y1) > 0 for (x1, y1, x2, y2) in all_boxes)
        bboxes_json = json.dumps(all_boxes)

        print(f"Completed processing {B} frames. Faces detected: {face_found}")
        return (img_batch[:, :, :, :3], mask_batch, face_found, bboxes_json)
