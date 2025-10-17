# wilddragon_nodes/image/person_bbox_tracker.py
# Node: 🐉 Person BBox Tracker

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import ToTensor, ToPILImage

_to_tensor = ToTensor()
_to_image = ToPILImage()


class WD_PersonBBoxTracker:
    """
    Wilddragon • Person BBox Tracker
    - Takes the selected person's bbox from Person Selector
    - Crops all video frames to that person
    - Outputs cropped frames for Pose Detection to process only that person
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "selected_bbox": ("BBOX",),
                "expansion_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "min_width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "min_height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOX", "INT", "INT")
    RETURN_NAMES = ("cropped_images", "crop_bbox", "crop_width", "crop_height")
    FUNCTION = "execute"
    CATEGORY = "Wilddragon/Image"

    def _expand_bbox(self, bbox, expansion_factor, img_width, img_height):
        """Expand bbox by factor while keeping it in image bounds."""
        x1, y1, x2, y2 = bbox

        # Calculate center and size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Expand
        new_w = w * expansion_factor
        new_h = h * expansion_factor

        # Calculate new bbox
        new_x1 = int(max(0, cx - new_w / 2))
        new_y1 = int(max(0, cy - new_h / 2))
        new_x2 = int(min(img_width, cx + new_w / 2))
        new_y2 = int(min(img_height, cy + new_h / 2))

        return (new_x1, new_y1, new_x2, new_y2)

    def _make_even_dimensions(self, w, h):
        """Make dimensions even (required by many video codecs)."""
        return (w // 2) * 2, (h // 2) * 2

    def execute(self, images, selected_bbox, expansion_factor, min_width, min_height):
        B, H, W, C = images.shape

        # Extract bbox
        if isinstance(selected_bbox, list) and len(selected_bbox) > 0:
            bbox = selected_bbox[0]
        else:
            bbox = selected_bbox

        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            print(f"[Person BBox Tracker] Invalid bbox: {bbox}, using full frame")
            bbox = (0, 0, W, H)

        # Ensure bbox values are integers
        bbox = tuple(int(v) for v in bbox[:4])
        x1, y1, x2, y2 = bbox

        # Validate bbox
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            print(f"[Person BBox Tracker] Invalid bbox values: {bbox}, using full frame")
            bbox = (0, 0, W, H)
            x1, y1, x2, y2 = bbox

        # Expand bbox
        expanded_bbox = self._expand_bbox(bbox, expansion_factor, W, H)
        crop_x1, crop_y1, crop_x2, crop_y2 = expanded_bbox

        # Calculate crop dimensions
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # Ensure minimum size
        scale_w = min_width / crop_w if crop_w < min_width else 1.0
        scale_h = min_height / crop_h if crop_h < min_height else 1.0

        if scale_w > 1.0 or scale_h > 1.0:
            scale = max(scale_w, scale_h)
            crop_w = int(crop_w * scale)
            crop_h = int(crop_h * scale)

            # Recalculate bbox centered on original
            cx = (crop_x1 + crop_x2) / 2
            cy = (crop_y1 + crop_y2) / 2
            crop_x1 = int(max(0, cx - crop_w / 2))
            crop_y1 = int(max(0, cy - crop_h / 2))
            crop_x2 = int(min(W, cx + crop_w / 2))
            crop_y2 = int(min(H, cy + crop_h / 2))

        # Make dimensions even
        crop_w, crop_h = self._make_even_dimensions(crop_x2 - crop_x1, crop_y2 - crop_y1)

        # Adjust x2, y2 to match even dimensions
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h

        # Ensure still in bounds
        if crop_x2 > W:
            crop_x2 = W
            crop_x1 = W - crop_w
        if crop_y2 > H:
            crop_y2 = H
            crop_y1 = H - crop_h

        print(f"[Person BBox Tracker] Original bbox: {bbox}")
        print(f"[Person BBox Tracker] Expanded bbox: ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})")
        print(f"[Person BBox Tracker] Crop size: {crop_w}x{crop_h}")
        print(f"[Person BBox Tracker] Processing {B} frames")

        # Crop all frames
        cropped_frames = []
        for i in range(B):
            frame = images[i]
            # Convert to PIL for cropping
            frame_pil = _to_image(frame.permute(2, 0, 1))
            # Crop
            cropped = frame_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            # Convert back to tensor
            cropped_tensor = _to_tensor(cropped).permute(1, 2, 0)
            cropped_frames.append(cropped_tensor)

        # Stack into batch
        cropped_batch = torch.stack(cropped_frames, dim=0)

        return (
            cropped_batch,
            [(crop_x1, crop_y1, crop_x2, crop_y2)],
            crop_w,
            crop_h
        )


NODE_CLASS_MAPPINGS = {
    "WD_PersonBBoxTracker": WD_PersonBBoxTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_PersonBBoxTracker": "🐉 Person BBox Tracker",
}
