# comfyui-wilddragon-nodes/__init__.py
# Robust import even if the folder has hyphens in its name.

try:
    from .image.face_crop_2025 import WD_ImageFaceCrop2025
    from .image.person_selector import WD_PersonSelector
    from .image.pose_face_filter import WD_PoseFaceFilter
    from .image.person_bbox_tracker import WD_PersonBBoxTracker
    _import_success = True
except Exception as _import_err:
    # Fallback shim node that surfaces the import error in UI
    _ERR = repr(_import_err)
    _import_success = False

    class WD_ImageFaceCrop2025:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error": ("STRING", {"default": _ERR, "multiline": True})}}

        RETURN_TYPES = ("STRING",)
        FUNCTION = "echo"
        CATEGORY = "Wilddragon/Errors"

        def echo(self, error):
            return (error,)

    class WD_PersonSelector:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error": ("STRING", {"default": _ERR, "multiline": True})}}

        RETURN_TYPES = ("STRING",)
        FUNCTION = "echo"
        CATEGORY = "Wilddragon/Errors"

        def echo(self, error):
            return (error,)

    class WD_PoseFaceFilter:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error": ("STRING", {"default": _ERR, "multiline": True})}}

        RETURN_TYPES = ("STRING",)
        FUNCTION = "echo"
        CATEGORY = "Wilddragon/Errors"

        def echo(self, error):
            return (error,)

    class WD_PersonBBoxTracker:
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
    "WD_PersonSelector": WD_PersonSelector,
    "WD_PoseFaceFilter": WD_PoseFaceFilter,
    "WD_PersonBBoxTracker": WD_PersonBBoxTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_ImageFaceCrop2025": "🐉 Image Face Crop (2025)",
    "WD_PersonSelector": "🐉 Person Selector",
    "WD_PoseFaceFilter": "🐉 Pose Face Filter",
    "WD_PersonBBoxTracker": "🐉 Person BBox Tracker",
}

__all__ = ["WD_ImageFaceCrop2025", "WD_PersonSelector", "WD_PoseFaceFilter", "WD_PersonBBoxTracker"]
