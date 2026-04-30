import torch

from .constants import COMBINE_MODES
from .processing import (
    _combine_channels,
    _normalize_mask_channel,
    _process_mask,
    _require_result,
    _threshold_mask,
)
from .unified import run_result


def _pointmap_result(model, image, mask=None, preserve_background: bool = False):
    raw = run_result(model, image, mask=mask, preserve_background=preserve_background)
    _require_result(raw, "pointmap")
    return raw


class Sapiens2PointmapChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("x", "y", "z_depth", "valid_mask", "depth_image")
    FUNCTION = "split"
    CATEGORY = "Sapiens2/Pointmap"

    def split(self, model, image):
        raw = _pointmap_result(model, image)
        pointmap = raw["pointmap"].detach().cpu().float()
        valid = torch.isfinite(pointmap[:, 2]) & (pointmap[:, 2] > 0)
        depth = raw.get("depth_preview")
        if depth is None:
            depth = _normalize_mask_channel(pointmap[:, 2], valid)
        image = torch.stack((depth, depth, depth), dim=-1).clamp(0, 1)
        return (
            _normalize_mask_channel(pointmap[:, 0], valid),
            _normalize_mask_channel(pointmap[:, 1], valid),
            _normalize_mask_channel(pointmap[:, 2], valid),
            valid.float(),
            image,
        )


class Sapiens2PointmapSelectChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "channel": (("x", "y", "z_depth", "depth_preview", "valid_mask"),),
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
    CATEGORY = "Sapiens2/Pointmap"

    def select(self, model, image, channel: str, grow_pixels: int = 0, blur_pixels: int = 0, invert: bool = False):
        raw = _pointmap_result(model, image)
        pointmap = raw["pointmap"].detach().cpu().float()
        valid = torch.isfinite(pointmap[:, 2]) & (pointmap[:, 2] > 0)
        if channel == "valid_mask":
            mask = valid.float()
        elif channel == "depth_preview":
            mask = raw.get("depth_preview", _normalize_mask_channel(pointmap[:, 2], valid)).float()
        else:
            index = {"x": 0, "y": 1, "z_depth": 2}[channel]
            mask = _normalize_mask_channel(pointmap[:, index], valid)
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)


class Sapiens2PointmapDepthRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "min_depth": ("FLOAT", {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01}),
                "max_depth": ("FLOAT", {"default": 100000.0, "min": -100000.0, "max": 100000.0, "step": 0.01}),
            },
            "optional": {
                "grow_pixels": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "range_mask"
    CATEGORY = "Sapiens2/Pointmap"

    def range_mask(
        self,
        model,
        image,
        min_depth: float,
        max_depth: float,
        grow_pixels: int = 0,
        blur_pixels: int = 0,
        invert: bool = False,
    ):
        raw = _pointmap_result(model, image)
        depth = raw["pointmap"][:, 2].detach().cpu().float()
        low, high = sorted((float(min_depth), float(max_depth)))
        valid = torch.isfinite(depth) & (depth > 0)
        mask = (valid & (depth >= low) & (depth <= high)).float()
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)


class Sapiens2PointmapCombineChannels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "x": ("BOOLEAN", {"default": False}),
                "y": ("BOOLEAN", {"default": False}),
                "z_depth": ("BOOLEAN", {"default": True}),
                "valid_mask": ("BOOLEAN", {"default": False}),
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
    CATEGORY = "Sapiens2/Pointmap"

    def combine(
        self,
        model,
        image,
        x: bool,
        y: bool,
        z_depth: bool,
        valid_mask: bool,
        mode: str = "max",
        threshold: float = 0.0,
        grow_pixels: int = 0,
        blur_pixels: int = 0,
        invert: bool = False,
    ):
        raw = _pointmap_result(model, image)
        pointmap = raw["pointmap"].detach().cpu().float()
        valid = torch.isfinite(pointmap[:, 2]) & (pointmap[:, 2] > 0)
        channels = []
        if x:
            channels.append(_normalize_mask_channel(pointmap[:, 0], valid))
        if y:
            channels.append(_normalize_mask_channel(pointmap[:, 1], valid))
        if z_depth:
            channels.append(_normalize_mask_channel(pointmap[:, 2], valid))
        if valid_mask:
            channels.append(valid.float())
        mask = _threshold_mask(_combine_channels(channels, mode), threshold)
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)
