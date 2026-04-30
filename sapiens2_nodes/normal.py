from typing import Any, Dict

from .constants import COMBINE_MODES
from .processing import (
    _combine_channels,
    _process_mask,
    _require_result,
    _signed_to_mask,
    _threshold_mask,
)


class Sapiens2NormalChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw": ("SAPIENS2_RESULT",)}}

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("x", "y", "z", "normal_image")
    FUNCTION = "split"
    CATEGORY = "Sapiens2/Normal"

    def split(self, raw: Dict[str, Any]):
        _require_result(raw, "normal")
        normal = raw["normal"].detach().cpu().float()
        image = ((normal + 1.0) * 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        return (
            _signed_to_mask(normal[:, 0]),
            _signed_to_mask(normal[:, 1]),
            _signed_to_mask(normal[:, 2]),
            image,
        )


class Sapiens2NormalSelectChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("SAPIENS2_RESULT",),
                "channel": (("x", "y", "z"),),
            },
            "optional": {
                "grow_pixels": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select"
    CATEGORY = "Sapiens2/Normal"

    def select(self, raw: Dict[str, Any], channel: str, grow_pixels: int = 0, blur_pixels: int = 0, invert: bool = False):
        _require_result(raw, "normal")
        index = {"x": 0, "y": 1, "z": 2}[channel]
        mask = _signed_to_mask(raw["normal"][:, index])
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)


class Sapiens2NormalCombineChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("SAPIENS2_RESULT",),
                "x": ("BOOLEAN", {"default": True}),
                "y": ("BOOLEAN", {"default": True}),
                "z": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mode": (COMBINE_MODES,),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow_pixels": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"
    CATEGORY = "Sapiens2/Normal"

    def combine(
        self,
        raw: Dict[str, Any],
        x: bool,
        y: bool,
        z: bool,
        mode: str = "max",
        threshold: float = 0.0,
        grow_pixels: int = 0,
        blur_pixels: int = 0,
        invert: bool = False,
    ):
        _require_result(raw, "normal")
        normal = raw["normal"].detach().cpu().float()
        channels = []
        if x:
            channels.append(_signed_to_mask(normal[:, 0]))
        if y:
            channels.append(_signed_to_mask(normal[:, 1]))
        if z:
            channels.append(_signed_to_mask(normal[:, 2]))
        mask = _threshold_mask(_combine_channels(channels, mode), threshold)
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)
