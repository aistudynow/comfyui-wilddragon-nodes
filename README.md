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
- **faithdiff:** SDXL-based restoration (highest quality, requires ComfyUI model integration)

#### Installation
Face restoration models are optional. Install as needed:

```bash
# For RealESRGAN (general upscaling)
pip install basicsr realesrgan

# For GFPGAN (face restoration)
pip install basicsr gfpgan facexlib

# For CodeFormer (best quality, requires the above)
pip install basicsr realesrgan
# Note: CodeFormer weights need to be manually downloaded

# For FaithDiff (requires diffusers)
pip install diffusers transformers accelerate
```

#### Model Weights Setup

Place model weights in ComfyUI's model directory:

```
ComfyUI/models/
├── upscale_models/
│   └── RealESRGAN_x2plus.pth
├── gfpgan/
│   └── GFPGANv1.4.pth
├── codeformer/
│   └── codeformer.pth
└── widdragon/face/
    ├── FaithDiff.bin
    ├── Real_4_SDXL/  (or any SDXL base model)
    └── VAE_FP16/     (optional)
```

**Download Links:**
- RealESRGAN: https://github.com/xinntao/Real-ESRGAN/releases
- GFPGAN: https://github.com/TencentARC/GFPGAN/releases
- CodeFormer: https://github.com/sczhou/CodeFormer/releases
- FaithDiff: https://huggingface.co/jychen9811/FaithDiff

### How It Works

The face restoration workflow has two main steps:

1. **🐉 Image Face Crop (2025)** - Detects and crops faces from images
   - Returns: cropped face images, masks, bounding boxes (JSON)

2. **🐉 Face Restore & Blend** - Enhances and blends faces back
   - Takes: original images, cropped faces, masks, bboxes
   - Enhances faces using selected restoration model
   - Seamlessly blends enhanced faces back into original images
   - Supports color matching and multiple blend modes

### Key Features

**Face Crop Node:**
- Multiple detection backends (InsightFace, RetinaFace)
- Eye alignment for better restoration
- Face tracking across video frames
- Configurable crop margins and aspect ratios

**Face Restore & Blend Node:**
- Multiple restoration models (RealESRGAN, GFPGAN, CodeFormer, FaithDiff)
- Smart blending modes (alpha, Poisson, feather)
- Color matching between restored and original
- Automatic mask feathering for seamless results
- FaithDiff integration with ComfyUI's BASIC_PIPE