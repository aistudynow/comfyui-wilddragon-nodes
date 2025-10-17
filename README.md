### Backends
- **auto (default):** tries *InsightFace*, then *RetinaFace* if available.
- **insightface:** fast ONNX, recommended.
- **retinaface:** optional, requires `retina-face` + `opencv-python-headless`.


#### RetinaFace install notes
If you hit import errors, try:
```bash
pip install --upgrade pip setuptools wheel
pip install retina-face opencv-python-headless
```

### Face Restoration Models

The `🐉 Face Restore & Blend` node supports multiple face enhancement models:

- **none:** No restoration, just blending
- **realesrgan:** General upscaling (fastest, works without face detection)
- **gfpgan:** Face-specific restoration (balanced speed/quality)
- **codeformer:** Best quality face restoration (slower)

#### Installation
Face restoration models are optional. Install as needed:

```bash
# For RealESRGAN (general upscaling)
pip install basicsr realesrgan

# For GFPGAN (face restoration)
pip install basicsr gfpgan facexlib

# For CodeFormer (best quality, requires the above)
pip install codeformer
```

**Note:** You'll also need to download model weights. The node will guide you on first use.