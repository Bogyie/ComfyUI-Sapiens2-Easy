from collections import namedtuple

import torch

from .constants import SEG_OUTPUT_NAMES, SEG_PART_OPTIONS, SEG_PARTS, SEG_TOGGLE_KEYS
from .processing import _mask_for_part, _process_mask, _seg_class_ids
from .unified import Sapiens2RunAdvanced


SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


def _segmentation_result(model, image):
    preview, merged_mask, _, raw = Sapiens2RunAdvanced().run(model, image)
    if raw.get("task") != "segmentation":
        raise ValueError("This node needs a segmentation model.")
    return preview, merged_mask, raw


def _part_masks(class_ids: torch.Tensor) -> torch.Tensor:
    masks = [(class_ids == idx).float() for idx in range(len(SEG_PARTS))]
    return torch.stack(masks, dim=1).flatten(0, 1)


def _selected_ids(toggles: dict) -> list[int]:
    selected = [idx for idx, key in enumerate(SEG_TOGGLE_KEYS) if bool(toggles.get(key, False))]
    return selected or list(range(1, len(SEG_PARTS)))


def _merge_parts(class_ids: torch.Tensor, part_ids: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(class_ids, dtype=torch.float32)
    for idx in part_ids:
        mask = torch.maximum(mask, (class_ids == idx).float())
    return mask


def _mask_preview(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    image = image.detach().float().cpu().clamp(0, 1)
    mask = mask.detach().float().cpu().clamp(0, 1)
    color = torch.tensor([0.0, 1.0, 1.0], dtype=image.dtype).view(1, 1, 1, 3)
    preview = torch.where(mask.unsqueeze(-1) > 0, image * (1.0 - alpha) + color * alpha, image)
    return preview.clamp(0, 1)


def _segs_from_mask(mask: torch.Tensor, label: str = "sapiens2") -> tuple:
    mask = mask.detach().float().cpu().clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected MASK shape [H,W] or [B,H,W], got {tuple(mask.shape)}")
    height, width = mask.shape[-2:]
    segs = []
    for sample in mask:
        ys, xs = torch.where(sample > 0.5)
        if xs.numel() == 0:
            continue
        x1 = int(xs.min().item())
        y1 = int(ys.min().item())
        x2 = int(xs.max().item()) + 1
        y2 = int(ys.max().item()) + 1
        crop_region = [x1, y1, x2, y2]
        bbox = (x1, y1, x2, y2)
        cropped_mask = sample[y1:y2, x1:x2].numpy()
        segs.append(SEG(None, cropped_mask, 1.0, crop_region, bbox, label, None))
    return (height, width), segs


class Sapiens2Segmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK", "MASK", "SEGS", "IMAGE")
    RETURN_NAMES = ("masks", "merged_mask", "segm", "preview")
    FUNCTION = "segment"
    CATEGORY = "Sapiens2/Easy"

    def segment(self, model, image):
        preview, merged_mask, raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        return (_part_masks(class_ids), merged_mask, _segs_from_mask(merged_mask), preview)


class Sapiens2SegmentationPartMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK",) * len(SEG_PARTS)
    RETURN_NAMES = SEG_OUTPUT_NAMES
    FUNCTION = "split"
    CATEGORY = "Sapiens2/Segmentation"

    def split(self, model, image):
        _, _, raw = _segmentation_result(model, image)
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

    RETURN_TYPES = ("MASK", "MASK", "SEGS", "IMAGE")
    RETURN_NAMES = ("masks", "merged_mask", "segm", "preview")
    FUNCTION = "combine"
    CATEGORY = "Sapiens2/Advanced"

    def combine(self, model, image, grow_pixels: int = 0, blur_pixels: int = 0, invert: bool = False, **toggles):
        _, _, raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        masks = _part_masks(class_ids)
        mask = _merge_parts(class_ids, _selected_ids(toggles))
        merged_mask = _process_mask(mask, grow_pixels, blur_pixels, invert)
        return (masks, merged_mask, _segs_from_mask(merged_mask), _mask_preview(image, merged_mask))


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

    RETURN_TYPES = ("MASK", "MASK", "SEGS", "IMAGE")
    RETURN_NAMES = ("masks", "merged_mask", "segm", "preview")
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
        _, _, raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        mask = _mask_for_part(class_ids, part)
        merged_mask = _process_mask(mask, grow_pixels, blur_pixels, invert)
        return (merged_mask, merged_mask, _segs_from_mask(merged_mask, part), _mask_preview(image, merged_mask))
