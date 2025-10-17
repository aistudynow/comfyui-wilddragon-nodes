# wilddragon_nodes/image/person_selector.py
# Node: 🐉 Person Selector

import json
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision.transforms.v2 import ToTensor, ToPILImage

from ..utils.face_detectors import get_detector

_to_tensor = ToTensor()
_to_image = ToPILImage()


class WD_PersonSelector:
    """
    Wilddragon • Person Selector
    - Detects all people in first frame of video
    - Generates preview grid showing each detected person
    - Allows selection of which person to track
    - Outputs selected person index for use with face crop nodes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "person_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "backend": (["auto", "insightface", "retinaface"], {"default": "auto"}),
                "min_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preview_size": ("INT", {"default": 256, "min": 64, "max": 512, "step": 64}),
                "use_gpu": ("BOOLEAN", {"default": True, "label_on": "GPU", "label_off": "CPU"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("IMAGES", "SELECTED_INDEX", "TOTAL_PEOPLE", "DETECTION_INFO", "PREVIEW_GRID")
    FUNCTION = "execute"
    CATEGORY = "Wilddragon/Image"

    def _create_preview_grid(
        self,
        first_frame: Image.Image,
        detections: List[dict],
        selected_index: int,
        preview_size: int
    ) -> Image.Image:
        """Create a grid showing all detected people with labels."""
        if not detections:
            # Return blank image if no detections
            return Image.new("RGB", (preview_size, preview_size), color=(0, 0, 0))

        # Calculate grid dimensions
        num_people = len(detections)
        cols = min(4, num_people)
        rows = (num_people + cols - 1) // cols

        grid_width = cols * preview_size
        grid_height = rows * preview_size
        grid = Image.new("RGB", (grid_width, grid_height), color=(20, 20, 20))

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]

            # Crop face with margin
            margin = 50
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(first_frame.width, x2 + margin)
            y2 = min(first_frame.height, y2 + margin)

            face_crop = first_frame.crop((x1, y1, x2, y2))

            # Resize to preview size
            face_crop.thumbnail((preview_size - 20, preview_size - 60), Image.Resampling.LANCZOS)

            # Calculate position in grid
            col = idx % cols
            row = idx // cols

            # Create cell
            cell = Image.new("RGB", (preview_size, preview_size), color=(40, 40, 40))

            # Border color - green for selected, gray for others
            border_color = (0, 255, 0) if idx == selected_index else (100, 100, 100)
            draw = ImageDraw.Draw(cell)
            draw.rectangle([0, 0, preview_size - 1, preview_size - 1], outline=border_color, width=4)

            # Paste face crop centered
            face_w, face_h = face_crop.size
            paste_x = (preview_size - face_w) // 2
            paste_y = (preview_size - face_h - 40) // 2
            cell.paste(face_crop, (paste_x, paste_y))

            # Add label
            label = f"Person {idx}"
            if idx == selected_index:
                label += " ✓"

            # Draw label background
            draw.rectangle([5, preview_size - 35, preview_size - 5, preview_size - 5], fill=(0, 0, 0))

            # Draw label text
            text_color = (0, 255, 0) if idx == selected_index else (255, 255, 255)

            # Calculate text position for centering
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
            except Exception:
                text_width = len(label) * 10

            text_x = (preview_size - text_width) // 2
            draw.text((text_x, preview_size - 32), label, fill=text_color, font=font)

            # Paste cell into grid
            grid.paste(cell, (col * preview_size, row * preview_size))

        return grid

    def execute(
        self,
        images: torch.Tensor,
        person_index: int,
        backend: str,
        min_score: float,
        preview_size: int,
        use_gpu: bool,
    ):
        B, H, W, C = images.shape

        # Use first frame for detection
        first_frame_tensor = images[0]
        first_frame = _to_image(first_frame_tensor.permute(2, 0, 1))

        # Detect all people in first frame
        detector = get_detector(backend, use_gpu=use_gpu)

        try:
            raw_dets = detector.detect(first_frame)
        except Exception as e:
            print(f"[Person Selector] Detection failed: {e}")
            raw_dets = []

        # Filter by confidence
        detections = [d for d in raw_dets if float(d.get("score", 0.0)) >= min_score]

        # Sort by size (largest first) for consistent ordering
        detections = sorted(
            detections,
            key=lambda d: -(d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
        )

        total_people = len(detections)

        # Clamp selected index
        selected_index = max(0, min(person_index, total_people - 1)) if total_people > 0 else 0

        # Create detection info
        detection_info = {
            "total_people": total_people,
            "selected_index": selected_index,
            "detections": [
                {
                    "index": i,
                    "bbox": det["bbox"],
                    "score": float(det.get("score", 0.0)),
                    "selected": (i == selected_index)
                }
                for i, det in enumerate(detections)
            ]
        }

        # Create preview grid
        preview_grid = self._create_preview_grid(first_frame, detections, selected_index, preview_size)
        preview_tensor = _to_tensor(preview_grid).permute(1, 2, 0).unsqueeze(0)

        # Print info
        print(f"[Person Selector] Detected {total_people} people in frame")
        print(f"[Person Selector] Selected: Person {selected_index}")
        if total_people > 0:
            selected_det = detections[selected_index]
            print(f"[Person Selector] Selected person confidence: {selected_det.get('score', 0.0):.3f}")

        return (
            images,  # Pass through original images
            selected_index,  # Selected person index
            total_people,  # Total number of people detected
            json.dumps(detection_info, indent=2),  # Detection info as JSON
            preview_tensor  # Preview grid image
        )


NODE_CLASS_MAPPINGS = {
    "WD_PersonSelector": WD_PersonSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_PersonSelector": "🐉 Person Selector",
}
