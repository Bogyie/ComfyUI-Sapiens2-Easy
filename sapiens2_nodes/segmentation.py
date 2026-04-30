from typing import Any, Dict

import torch

from .constants import SEG_OUTPUT_NAMES, SEG_PART_OPTIONS, SEG_PARTS, SEG_TOGGLE_KEYS
from .processing import _mask_for_part, _process_mask, _seg_class_ids
from .unified import run_result


def _segmentation_result(model, image):
    raw = run_result(model, image)
    if raw.get("task") != "segmentation":
        raise ValueError("This node needs a segmentation model.")
    return raw


class Sapiens2SegmentationPartMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK",) * len(SEG_PARTS)
    RETURN_NAMES = SEG_OUTPUT_NAMES
    FUNCTION = "split"
    CATEGORY = "Sapiens2/Segmentation"

    def split(self, model, image):
        raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        return tuple((class_ids == idx).float() for idx in range(len(SEG_PARTS)))


class Sapiens2SegmentationCombine:
    @classmethod
    def INPUT_TYPES(cls):
        toggles = {
            key: ("BOOLEAN", {"default": idx != 0})
            for idx, key in enumerate(SEG_TOGGLE_KEYS)
        }
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                **toggles,
            },
            "optional": {
                "grow_pixels": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "blur_pixels": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"
    CATEGORY = "Sapiens2/Segmentation"

    def combine(self, model, image, grow_pixels: int = 0, blur_pixels: int = 0, invert: bool = False, **toggles):
        raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        selected = [
            idx for idx, key in enumerate(SEG_TOGGLE_KEYS) if bool(toggles.get(key, False))
        ]
        mask = torch.zeros_like(class_ids, dtype=torch.float32)
        for idx in selected:
            mask = torch.maximum(mask, (class_ids == idx).float())
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)


class Sapiens2SegmentationSelectPart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "part": (SEG_PART_OPTIONS,),
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
    CATEGORY = "Sapiens2/Segmentation"

    def select(
        self,
        model,
        image,
        part: str,
        grow_pixels: int = 0,
        blur_pixels: int = 0,
        invert: bool = False,
    ):
        raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        mask = _mask_for_part(class_ids, part)
        return (_process_mask(mask, grow_pixels, blur_pixels, invert),)
