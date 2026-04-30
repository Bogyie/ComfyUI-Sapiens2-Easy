from collections import namedtuple

import torch

from .constants import SEG_CLASS_COUNT, SEG_OUTPUT_NAMES, SEG_PALETTE, SEG_PART_OPTIONS, SEG_PARTS, SEG_TOGGLE_KEYS
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
    return _comfy_image(preview), _comfy_mask(merged_mask), raw


def _comfy_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().float().cpu().clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected MASK shape [H,W] or [B,H,W], got {tuple(mask.shape)}")
    return mask


def _comfy_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected IMAGE shape [H,W,C] or [B,H,W,C], got {tuple(image.shape)}")
    return image


def _part_masks(class_ids: torch.Tensor, part_ids: list[int] | None = None) -> torch.Tensor:
    part_ids = part_ids if part_ids is not None else list(range(len(SEG_PARTS)))
    masks = [(class_ids == idx).float() for idx in part_ids]
    return _comfy_mask(torch.stack(masks, dim=1).flatten(0, 1))


def _selected_ids(toggles: dict) -> list[int]:
    selected = [idx for idx, key in enumerate(SEG_TOGGLE_KEYS) if bool(toggles.get(key, False))]
    return selected or list(range(1, len(SEG_PARTS)))


def _merge_parts(class_ids: torch.Tensor, part_ids: list[int]) -> torch.Tensor:
    mask = torch.zeros_like(class_ids, dtype=torch.float32)
    for idx in part_ids:
        mask = torch.maximum(mask, (class_ids == idx).float())
    return _comfy_mask(mask)


def _seg_preview(image: torch.Tensor, class_ids: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    image = _comfy_image(image)
    class_ids = class_ids.detach().cpu().long()
    if class_ids.ndim == 2:
        class_ids = class_ids.unsqueeze(0)
    if class_ids.ndim != 3:
        raise ValueError(f"Expected class id shape [H,W] or [B,H,W], got {tuple(class_ids.shape)}")
    palette = SEG_PALETTE.to(dtype=image.dtype)
    color = palette[class_ids.clamp(0, SEG_CLASS_COUNT - 1)]
    return _comfy_image(image * (1.0 - alpha) + color * alpha)


def _segs_from_mask(mask: torch.Tensor, label: str = "sapiens2", image: torch.Tensor | None = None) -> tuple:
    mask = _comfy_mask(mask)
    image = _comfy_image(image) if image is not None else None
    height, width = mask.shape[-2:]
    segs = []
    for idx, sample in enumerate(mask):
        ys, xs = torch.where(sample > 0.5)
        if xs.numel() == 0:
            continue
        x1 = int(xs.min().item())
        y1 = int(ys.min().item())
        x2 = int(xs.max().item()) + 1
        y2 = int(ys.max().item()) + 1
        crop_region = (x1, y1, x2, y2)
        bbox = (x1, y1, x2, y2)
        cropped_mask = sample[y1:y2, x1:x2].contiguous()
        cropped_image = None
        if image is not None and idx < image.shape[0]:
            cropped_image = image[idx, y1:y2, x1:x2, :].contiguous()
        segs.append(SEG(cropped_image, cropped_mask, 1.0, crop_region, bbox, label, None))
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
        _, merged_mask, raw = _segmentation_result(model, image)
        class_ids = _seg_class_ids(raw)
        preview = _seg_preview(image, class_ids)
        return (_part_masks(class_ids), merged_mask, _segs_from_mask(merged_mask, image=image), preview)


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
        return tuple(_comfy_mask(class_ids == idx) for idx in range(len(SEG_PARTS)))


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
        part_ids = _selected_ids(toggles)
        masks = _part_masks(class_ids, part_ids)
        mask = _merge_parts(class_ids, part_ids)
        merged_mask = _process_mask(mask, grow_pixels, blur_pixels, invert)
        merged_mask = _comfy_mask(merged_mask)
        return (masks, merged_mask, _segs_from_mask(merged_mask, image=image), _seg_preview(image, class_ids))


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
    RETURN_NAMES = ("mask", "merged_mask", "segm", "preview")
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
        merged_mask = _comfy_mask(merged_mask)
        return (merged_mask, merged_mask, _segs_from_mask(merged_mask, part, image=image), _seg_preview(image, class_ids))
