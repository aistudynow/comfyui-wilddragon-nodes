# wilddragon_nodes/utils/face_detectors.py
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import folder_paths  # provided by ComfyUI

# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------

class BaseFaceDetector:
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Returns detections: list of dicts with
          - bbox: [x1, y1, x2, y2] ints
          - score: float
          - kps: optional (5x2) ndarray of landmarks (eyes/nose/mouth)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# InsightFace backend (recommended)
# ---------------------------------------------------------------------------

class InsightFaceDetector(BaseFaceDetector):
    def __init__(self, providers: Optional[List[str]] = None):
        # Lazy import to avoid heavy module load at package import time
        from insightface.app import FaceAnalysis  # type: ignore
        models_root = os.path.join(folder_paths.models_dir, "insightface")
        os.makedirs(models_root, exist_ok=True)

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.app = FaceAnalysis(providers=providers, root=models_root)
        try:
            self.app.prepare(ctx_id=0, det_size=(320, 320))
        except Exception:
            self.app.prepare(ctx_id=-1, det_size=(320, 320))

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        im_np = np.array(image.convert("RGB"))
        faces = self.app.get(im_np)
        out: List[Dict[str, Any]] = []
        for f in faces:
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            bbox = np.asarray(bbox).astype(int).tolist()
            score = float(getattr(f, "det_score", 1.0))
            kps = getattr(f, "kps", None)
            if kps is not None:
                kps = np.asarray(kps).astype(float)
            out.append({"bbox": bbox, "score": score, "kps": kps})
        return out


# ---------------------------------------------------------------------------
# RetinaFace backend (optional; may pull TF deps on some setups)
# ---------------------------------------------------------------------------

class RetinaFaceDetector(BaseFaceDetector):
    def __init__(self):
        # Requires `retina-face` and OpenCV. Import lazily.
        from retinaface import RetinaFace  # type: ignore
        self.rf = RetinaFace
        try:
            self.model = self.rf.build_model()
        except Exception:
            self.model = None

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        im_np = np.array(image.convert("RGB"))
        try:
            faces = self.rf.detect_faces(im_np, model=self.model)
        except TypeError:
            faces = self.rf.detect_faces(im_np)
        out: List[Dict[str, Any]] = []
        if isinstance(faces, dict):
            for f in faces.values():
                area = f.get("facial_area")
                if area is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in area]
                score = float(f.get("score", 1.0))
                lms = f.get("landmarks", None)
                kps = None
                if lms:
                    # order: left_eye, right_eye, nose, mouth_left, mouth_right
                    kps = np.array(
                        [
                            lms.get("left_eye"),
                            lms.get("right_eye"),
                            lms.get("nose"),
                            lms.get("mouth_left"),
                            lms.get("mouth_right"),
                        ],
                        dtype=float,
                    )
                out.append({"bbox": [x1, y1, x2, y2], "score": score, "kps": kps})
        return out


# ---------------------------------------------------------------------------
# Factory + helpers
# ---------------------------------------------------------------------------

_DETECTORS = {
    "insightface": InsightFaceDetector,
    "retinaface": RetinaFaceDetector,
}

def get_detector(name: str, **kwargs) -> BaseFaceDetector:
    """
    name:
      - 'auto': try insightface then retinaface
      - 'insightface'
      - 'retinaface'
    """
    name = (name or "insightface").lower()
    if name == "auto":
        for cand in ("insightface", "retinaface"):
            try:
                return _DETECTORS[cand](**kwargs)
            except Exception:
                continue
        raise RuntimeError("No face detector backends available. Install 'insightface' or 'retina-face'.")
    if name not in _DETECTORS:
        raise ValueError(f"Detector '{name}' not available.")
    return _DETECTORS[name](**kwargs)


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    margin: float,
    target_aspect: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0

    side = max(bw, bh)
    side *= (1.0 + float(margin) * 2.0)

    if target_aspect >= 1.0:
        w = side
        h = side / target_aspect
    else:
        h = side
        w = side * target_aspect

    x1n = int(round(cx - w / 2.0))
    y1n = int(round(cy - h / 2.0))
    x2n = int(round(cx + w / 2.0))
    y2n = int(round(cy + h / 2.0))

    x1n = max(0, x1n)
    y1n = max(0, y1n)
    x2n = min(width, x2n)
    y2n = min(height, y2n)

    if x2n <= x1n:
        x2n = min(width, x1n + 1)
    if y2n <= y1n:
        y2n = min(height, y1n + 1)
    return (x1n, y1n, x2n, y2n)


def eye_rotation_radians_from_kps(kps: Optional[np.ndarray]) -> float:
    """Angle from left/right eye landmarks. Returns 0 if not available."""
    if kps is None or len(kps) < 2:
        return 0.0
    p0, p1 = kps[0], kps[1]
    dy = float(p1[1] - p0[1])
    dx = float(p1[0] - p0[0])
    if dx == 0:
        return 0.0
    return float(np.arctan2(dy, dx))
