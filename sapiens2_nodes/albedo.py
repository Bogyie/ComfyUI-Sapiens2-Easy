from typing import Any, Dict

from .constants import COMBINE_MODES
from .processing import _combine_channels, _process_mask, _require_result, _threshold_mask


class Sapiens2AlbedoChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw": ("SAPIENS2_RESULT",)}}

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("red", "green", "blue", "luminance", "albedo_image")
    FUNCTION = "split"
    CATEGORY = "Sapiens2/Albedo"

    def split(self, raw: Dict[str, Any]):
        _require_result(raw, "albedo")
        albedo = raw["albedo"].detach().cpu().float().clamp(0, 1)
        lum = (0.2126 * albedo[:, 0] + 0.7152 * albedo[:, 1] + 0.0722 * albedo[:, 2]).clamp(0, 1)
        return (
            albedo[:, 0],
            albedo[:, 1],
            albedo[:, 2],
            lum,
            albedo.permute(0, 2, 3, 1),
        )


class Sapiens2AlbedoSelectChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("SAPIENS2_RESULT",),
                "channel": (("red", "green", "blue", "luminance"),),
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
    CATEGORY = "Sapiens2/Albedo"

    def select(self, raw: Dict[str, Any], channel: str, grow_pixels: int = 0, blur_pixels: int = 0, invert: bool = False):
        _require_result(raw, "albedo")
        albedo = raw["albedo"].detach().cpu().float().clamp(0, 1)
        if channel == "luminance":
            mask = (0.2126 * albedo[:, 0] + 0.7152 * albedo[:, 1] + 0.0722 * albedo[:, 2]).clamp(0, 1)
        else:
            index = {"red": 0, "green": 1, "blue": 2}[channel]
            mask = albedo[:, index]
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)


class Sapiens2AlbedoCombineChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("SAPIENS2_RESULT",),
                "red": ("BOOLEAN", {"default": True}),
                "green": ("BOOLEAN", {"default": True}),
                "blue": ("BOOLEAN", {"default": True}),
                "luminance": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "Sapiens2/Albedo"

    def combine(
        self,
        raw: Dict[str, Any],
        red: bool,
        green: bool,
        blue: bool,
        luminance: bool,
        mode: str = "max",
        threshold: float = 0.0,
        grow_pixels: int = 0,
        blur_pixels: int = 0,
        invert: bool = False,
    ):
        _require_result(raw, "albedo")
        albedo = raw["albedo"].detach().cpu().float().clamp(0, 1)
        lum = (0.2126 * albedo[:, 0] + 0.7152 * albedo[:, 1] + 0.0722 * albedo[:, 2]).clamp(0, 1)
        channels = []
        if red:
            channels.append(albedo[:, 0])
        if green:
            channels.append(albedo[:, 1])
        if blue:
            channels.append(albedo[:, 2])
        if luminance:
            channels.append(lum)
        mask = _threshold_mask(_combine_channels(channels, mode), threshold)
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)
