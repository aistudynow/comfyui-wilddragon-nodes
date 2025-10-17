# Workflow Examples

## Basic Face Enhancement Workflow

### 1. Simple Photo Enhancement

```
[Load Image]
    ↓
[🐉 Image Face Crop (2025)]
    ↓ (IMAGE, MASK, FACE_DETECTED, BBOXES_JSON)
[🐉 Face Restore & Blend]
    ↓
[Save Image]
```

**Settings for Face Crop:**
- Backend: `auto`
- Crop Size: `512x512`
- Selection: `largest`
- Align Rotation: `ON`

**Settings for Restore & Blend:**
- Restoration Model: `gfpgan` (good balance)
- Restoration Strength: `0.8`
- Blend Mode: `feather`
- Feather Amount: `15`
- Color Match: `ON`

---

## Advanced Workflows

### 2. High Quality Portrait Enhancement (using FaithDiff)

```
[Load Checkpoint (SDXL)]
    ↓ (MODEL, CLIP, VAE)
[To Basic Pipe]
    ↓ (BASIC_PIPE)

[Load Image]
    ↓
[🐉 Image Face Crop (2025)]
    ↓
[🐉 Face Restore & Blend]
    ↑ (connect BASIC_PIPE here)
    ↓
[Save Image]
```

**Why use BASIC_PIPE:**
- FaithDiff requires SDXL components (model, clip, vae)
- BASIC_PIPE provides these from ComfyUI's loaded checkpoint
- No need to specify paths manually

**Settings:**
- Restoration Model: `faithdiff`
- Restoration Strength: `0.8`
- Connect `faithdiff_basic_pipe` to BASIC_PIPE output

---

### 3. Video Face Enhancement

```
[Load Video (VHS)]
    ↓ (IMAGE batch)
[🐉 Image Face Crop (2025)]
    ↓
[🐉 Face Restore & Blend]
    ↓
[Save Video (VHS)]
```

**Important Settings:**
- **Detect Interval: `10`** - Only detect every 10th frame (faster)
- **Selection: `index`** - Tracks same face across frames
- **Max Faces: `1`** - Single face tracking

**For multiple faces:**
- Set `max_faces` to desired count
- Use `selection: all` to enhance all detected faces

---

### 4. Batch Processing with Different Models

Process the same image with multiple restoration models for comparison:

```
[Load Image]
    ↓
[🐉 Image Face Crop (2025)]
    ↓→ [🐉 Face Restore & Blend] (RealESRGAN) → [Preview]
    ↓→ [🐉 Face Restore & Blend] (GFPGAN) → [Preview]
    ↓→ [🐉 Face Restore & Blend] (CodeFormer) → [Preview]
    ↓→ [🐉 Face Restore & Blend] (FaithDiff) → [Preview]
```

**Compare results:**
- RealESRGAN: Fast, general enhancement
- GFPGAN: Balanced, face-specific
- CodeFormer: High quality, artifact reduction
- FaithDiff: Highest quality, photorealistic

---

### 5. Multi-Face Group Photo Enhancement

```
[Load Image]
    ↓
[🐉 Image Face Crop (2025)]
    Selection: all
    Max Faces: 10
    ↓
[🐉 Face Restore & Blend]
    Restoration Model: gfpgan
    ↓
[Save Image]
```

**Key Points:**
- `selection: all` processes all detected faces
- Each face enhanced independently
- Blended back at original positions
- Color matching ensures consistency

---

## Tips for Best Results

### Face Crop Settings

**For portraits:**
- Margin: `0.2-0.3` (more context)
- Crop Size: `512x512` or `1024x1024`
- Align Rotation: `ON` (better alignment)

**For group photos:**
- Selection: `all`
- Max Faces: `5-10`
- Min Score: `0.3` (catch more faces)

**For video:**
- Detect Interval: `10-30` (faster)
- Selection: `index` (track one face)
- Align Rotation: `ON` (stability)

### Restoration Settings

**For natural look:**
- Strength: `0.6-0.8`
- Blend Strength: `0.8-1.0`
- Color Match: `ON`
- Feather: `15-20`

**For maximum enhancement:**
- Strength: `0.9-1.0`
- Blend Strength: `1.0`
- Color Match: `ON`
- Feather: `20-30`

**For creative control:**
- Strength: `0.4-0.6` (subtle)
- Try different blend modes
- Adjust feather for hard/soft edges

---

## Integration with ComfyUI Workflows

### With Upscaling

```
[Load Image]
    ↓
[Upscale (4x)]
    ↓
[🐉 Image Face Crop (2025)]
    Crop Size: 1024x1024  (larger for upscaled image)
    ↓
[🐉 Face Restore & Blend]
    ↓
[Save Image]
```

### With Image Generation

```
[KSampler] (generate image)
    ↓
[VAE Decode]
    ↓
[🐉 Image Face Crop (2025)]
    ↓
[🐉 Face Restore & Blend]
    ↓
[Save Image]
```

### With Inpainting

```
[Load Image + Mask]
    ↓
[Inpaint]
    ↓
[🐉 Image Face Crop (2025)]
    ↓
[🐉 Face Restore & Blend]
    ↓
[Save Image]
```

---

## Model Selection Guide

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| none | ⚡⚡⚡ | - | 0GB | Testing, no enhancement |
| RealESRGAN | ⚡⚡⚡ | ⭐⭐ | 2GB | Fast processing, general |
| GFPGAN | ⚡⚡ | ⭐⭐⭐ | 4GB | Balanced, face-specific |
| CodeFormer | ⚡ | ⭐⭐⭐⭐ | 6GB | High quality, artifacts |
| FaithDiff | ⚡ | ⭐⭐⭐⭐⭐ | 8GB+ | Photorealistic, portraits |

**Processing Time (512x512 face, RTX 3090):**
- RealESRGAN: ~0.1s
- GFPGAN: ~0.3s
- CodeFormer: ~0.5s
- FaithDiff: ~2-3s

---

## Common Workflow Patterns

### Pattern 1: Detection → Enhancement → Blend
The standard workflow for most use cases.

### Pattern 2: Crop → Manual Edit → Blend
Skip restoration, manually edit cropped faces in external software, then blend back.

### Pattern 3: Multi-Pass Enhancement
Run restoration multiple times with different settings:
```
Crop → Restore (strength 0.5) → Crop Again → Restore (strength 0.3) → Blend
```

### Pattern 4: Conditional Enhancement
Use face detection results to conditionally apply enhancement:
```
if FACE_DETECTED == True:
    apply enhancement
else:
    pass through original
```

---

## Troubleshooting Workflows

### Not Getting Good Results?

1. **Check face detection:**
   - Preview the cropped faces output
   - Ensure faces are properly aligned
   - Adjust `min_score` if faces not detected

2. **Check enhancement:**
   - Try different restoration models
   - Increase restoration strength
   - Verify model weights loaded

3. **Check blending:**
   - Increase feather amount for smoother blend
   - Enable color match
   - Try different blend modes

### Performance Issues?

1. **Reduce resolution:**
   - Use smaller crop sizes (256x256)
   - Downscale input images first

2. **Optimize detection:**
   - Increase detect_interval for video
   - Use simpler backend (insightface)

3. **Use faster models:**
   - Start with RealESRGAN
   - Only use FaithDiff for final renders

---

For more help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
