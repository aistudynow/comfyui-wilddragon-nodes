# comfyui-wilddragon-nodes/__init__.py
# Robust import even if the folder has hyphens in its name.

try:
    from .image.face_crop_2025 import WD_ImageFaceCrop2025
except Exception as _import_err:
    # Fallback shim node that surfaces the import error in UI
    _ERR = repr(_import_err)

    class WD_ImageFaceCrop2025:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error": ("STRING", {"default": _ERR, "multiline": True})}}

        RETURN_TYPES = ("STRING",)
        FUNCTION = "echo"
        CATEGORY = "Wilddragon/Errors"

        def echo(self, error):
            return (error,)

NODE_CLASS_MAPPINGS = {
    "WD_ImageFaceCrop2025": WD_ImageFaceCrop2025,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_ImageFaceCrop2025": "🐉 Image Face Crop (2025)",
}

__all__ = ["WD_ImageFaceCrop2025"]
