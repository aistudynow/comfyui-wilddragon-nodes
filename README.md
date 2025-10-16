### Backends
- **auto (default):** tries *InsightFace*, then *RetinaFace* if available.
- **insightface:** fast ONNX, recommended.
- **retinaface:** optional, requires `retina-face` + `opencv-python-headless`.


#### RetinaFace install notes
If you hit import errors, try:
```bash
pip install --upgrade pip setuptools wheel
pip install retina-face opencv-python-headless