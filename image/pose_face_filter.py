# wilddragon_nodes/image/pose_face_filter.py
# Node: 🐉 Pose Face Filter

import numpy as np
import torch

class WD_PoseFaceFilter:
    """
    Wilddragon • Pose Face Filter
    - Filters pose and face detection results to only include the selected person
    - Uses bounding box from Person Selector to match detections
    - Works with Pose and Face Detection node from WanAnimatePreprocess
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "selected_bbox": ("BBOX",),
                "face_images": ("IMAGE",),
                "bboxes": ("BBOX",),
                "face_bboxes": ("BBOX,",),
                "iou_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "BBOX", "BBOX")
    RETURN_NAMES = ("filtered_pose_data", "filtered_face_images", "filtered_bboxes", "filtered_face_bboxes")
    FUNCTION = "execute"
    CATEGORY = "Wilddragon/Image"

    def _compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _normalize_bbox(self, bbox):
        """Normalize bbox to [x1, y1, x2, y2] format."""
        if isinstance(bbox, (list, tuple)):
            if len(bbox) >= 4:
                # Already in correct format or has extra data
                return list(bbox[:4])
        return [0, 0, 0, 0]

    def _find_matching_frame_index(self, selected_bbox, bboxes, iou_threshold):
        """Find the frame where the bbox best matches the selected person."""
        # Normalize selected bbox
        if isinstance(selected_bbox, list) and len(selected_bbox) > 0:
            selected_bbox = selected_bbox[0]

        selected_bbox = self._normalize_bbox(selected_bbox)

        if selected_bbox == [0, 0, 0, 0]:
            print(f"[Pose Face Filter] Invalid selected bbox, using first frame")
            return 0

        best_iou = 0.0
        best_index = 0

        print(f"[Pose Face Filter] Selected bbox: {selected_bbox}")
        print(f"[Pose Face Filter] Comparing against {len(bboxes)} detected bboxes")

        for i, bbox in enumerate(bboxes):
            normalized_bbox = self._normalize_bbox(bbox)
            if normalized_bbox != [0, 0, 0, 0]:
                iou = self._compute_iou(selected_bbox, normalized_bbox)
                if i < 3:  # Debug first few
                    print(f"  Frame {i} bbox {normalized_bbox}: IOU = {iou:.3f}")
                if iou > best_iou:
                    best_iou = iou
                    best_index = i

        print(f"[Pose Face Filter] Best matching frame: {best_index} with IOU: {best_iou:.3f}")

        if best_iou < iou_threshold:
            print(f"[Pose Face Filter] Warning: Best IOU {best_iou:.3f} is below threshold {iou_threshold}")
            print(f"[Pose Face Filter] This may indicate the selected person is not in the detected frames")

        return best_index

    def execute(
        self,
        pose_data,
        selected_bbox,
        face_images,
        bboxes,
        face_bboxes,
        iou_threshold,
    ):
        # Find which frame corresponds to the selected person
        frame_index = self._find_matching_frame_index(selected_bbox, bboxes, iou_threshold)

        print(f"[Pose Face Filter] Filtering pose data to frame {frame_index}")

        # Extract pose data for the selected person
        pose_metas = pose_data.get("pose_metas", [])
        pose_metas_original = pose_data.get("pose_metas_original", [])

        if frame_index < len(pose_metas):
            filtered_pose_data = {
                "retarget_image": pose_data.get("retarget_image"),
                "pose_metas": [pose_metas[frame_index]],
                "refer_pose_meta": pose_data.get("refer_pose_meta"),
                "pose_metas_original": [pose_metas_original[frame_index]] if frame_index < len(pose_metas_original) else [],
            }
        else:
            print(f"[Pose Face Filter] Warning: frame_index {frame_index} out of range, using first frame")
            filtered_pose_data = {
                "retarget_image": pose_data.get("retarget_image"),
                "pose_metas": [pose_metas[0]] if pose_metas else [],
                "refer_pose_meta": pose_data.get("refer_pose_meta"),
                "pose_metas_original": [pose_metas_original[0]] if pose_metas_original else [],
            }
            frame_index = 0

        # Extract face image for the selected person
        if frame_index < len(face_images):
            filtered_face_images = face_images[frame_index:frame_index+1]
        else:
            print(f"[Pose Face Filter] Warning: No face image at index {frame_index}, using first")
            filtered_face_images = face_images[0:1] if len(face_images) > 0 else face_images

        # Extract bboxes for the selected person
        filtered_bboxes = [bboxes[frame_index]] if frame_index < len(bboxes) else [bboxes[0]] if bboxes else [(0, 0, 0, 0)]
        filtered_face_bboxes = [face_bboxes[frame_index]] if frame_index < len(face_bboxes) else [face_bboxes[0]] if face_bboxes else [(0, 0, 0, 0)]

        print(f"[Pose Face Filter] Filtered to {len(filtered_pose_data['pose_metas'])} pose(s)")

        return (
            filtered_pose_data,
            filtered_face_images,
            filtered_bboxes,
            filtered_face_bboxes
        )


NODE_CLASS_MAPPINGS = {
    "WD_PoseFaceFilter": WD_PoseFaceFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_PoseFaceFilter": "🐉 Pose Face Filter",
}
