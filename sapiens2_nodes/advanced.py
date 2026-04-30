import torch
import torch.nn.functional as F

from .constants import SEG_CLASS_COUNT, SEG_PARTS
from .easy import (
    POSE_TARGETS,
    PREVIEW_MODES,
    SEG_GROUPS,
    _comfy_mask,
    _format_preview,
    _mask_preview,
    _merge_parts,
    _openpose_json,
    _part_masks,
    _pose_target_image,
    _require_task,
    _selected_parts,
)
from .inference import Sapiens2DenseInference
from .pose import Sapiens2PoseInference
from .types import Sapiens2PoseModel


def _match_mask(mask, batch_size: int, height: int, width: int) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = _comfy_mask(mask)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.repeat(batch_size, 1, 1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"Mask batch size ({mask.shape[0]}) does not match image batch size ({batch_size}).")
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
    return mask.clamp(0, 1)


def _mask_part_batch(masks: torch.Tensor, input_mask: torch.Tensor | None, part_count: int) -> torch.Tensor:
    if input_mask is None:
        return masks
    if part_count <= 0:
        return masks * input_mask
    return masks * input_mask.repeat_interleave(part_count, dim=0)


class Sapiens2SegmentationAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "overlay_opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_background": ("BOOLEAN", {"default": False}),
                "invert": ("BOOLEAN", {"default": False}),
                "parts": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "SAPIENS2_LABELS", "SAPIENS2_RESULT")
    RETURN_NAMES = ("preview", "foreground_mask", "merged_mask", "masks", "labels", "result")
    FUNCTION = "segment"
    CATEGORY = "Sapiens2/Advanced"

    def segment(
        self,
        model,
        image,
        overlay_opacity: float = 0.55,
        preserve_background: bool = False,
        invert: bool = False,
        parts: str = "",
        mask=None,
    ):
        _require_task(model, "segmentation")
        _, foreground_mask, labels_mask, raw = Sapiens2DenseInference().run(
            model,
            image,
            overlay_opacity=overlay_opacity,
            preserve_background=preserve_background,
            mask=mask,
        )
        class_ids = raw["class_ids"].detach().cpu().long()
        input_mask = _match_mask(mask, class_ids.shape[0], class_ids.shape[-2], class_ids.shape[-1])
        selected = _selected_parts(parts)
        part_ids = selected if selected is not None else list(range(1, SEG_CLASS_COUNT))
        merged_mask = _merge_parts(class_ids, part_ids, invert)
        if input_mask is not None:
            merged_mask = _comfy_mask(merged_mask * input_mask)
        masks = _mask_part_batch(_part_masks(class_ids, part_ids), input_mask, len(part_ids))
        labels = {
            "class_ids": class_ids,
            "label_mask": labels_mask,
            "parts": SEG_PARTS,
            "groups": SEG_GROUPS,
            "selected_part_ids": part_ids,
        }
        return (
            _mask_preview(image, merged_mask, alpha=float(overlay_opacity)),
            foreground_mask,
            merged_mask,
            masks,
            labels,
            raw,
        )


class Sapiens2NormalAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preview_mode": (PREVIEW_MODES, {"default": "result"}),
                "preserve_background": ("BOOLEAN", {"default": False}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SAPIENS2_RESULT")
    RETURN_NAMES = ("normal_map", "foreground_mask", "result")
    FUNCTION = "run"
    CATEGORY = "Sapiens2/Advanced"

    def run(self, model, image, preview_mode: str = "result", preserve_background: bool = False, mask=None):
        _require_task(model, "normal")
        preview, foreground_mask, _, raw = Sapiens2DenseInference().run(
            model,
            image,
            preserve_background=preserve_background,
            mask=mask,
        )
        return (_format_preview(image, preview, preview_mode), foreground_mask, raw)


class Sapiens2PoseAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "target": (POSE_TARGETS, {"default": "BODY_25"}),
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "fallback_full_image_bbox": ("BOOLEAN", {"default": True}),
                "flip_test": ("BOOLEAN", {"default": True}),
                "show_points": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
            },
            "optional": {"bboxes": ("SAPIENS2_BBOXES",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING", "SAPIENS2_POSE_RESULT")
    RETURN_NAMES = ("openpose_image", "preview", "keypoint_mask", "openpose_json", "result")
    FUNCTION = "run"
    CATEGORY = "Sapiens2/Advanced"

    def run(
        self,
        model,
        image,
        target: str = "BODY_25",
        keypoint_threshold: float = 0.3,
        bbox_threshold: float = 0.3,
        nms_threshold: float = 0.3,
        radius: int = 4,
        thickness: int = 2,
        fallback_full_image_bbox: bool = True,
        flip_test: bool = True,
        show_points: bool = True,
        show_skeleton: bool = True,
        bboxes=None,
    ):
        if not isinstance(model, Sapiens2PoseModel):
            _require_task(model, "pose")
        _, keypoint_mask, raw = Sapiens2PoseInference().run(
            pose_model=model,
            image=image,
            keypoint_threshold=keypoint_threshold,
            bbox_threshold=bbox_threshold,
            nms_threshold=nms_threshold,
            radius=radius,
            thickness=thickness,
            fallback_full_image_bbox=fallback_full_image_bbox,
            flip_test=flip_test,
            show_points=show_points,
            show_skeleton=show_skeleton,
            bboxes=bboxes,
            render_outputs=True,
        )
        return (
            _pose_target_image(
                raw,
                image,
                target,
                radius=radius,
                thickness=thickness,
                show_points=show_points,
                show_skeleton=show_skeleton,
            ),
            _pose_target_image(
                raw,
                image,
                target,
                overlay=True,
                radius=radius,
                thickness=thickness,
                show_points=show_points,
                show_skeleton=show_skeleton,
            ),
            keypoint_mask,
            _openpose_json(raw, target),
            raw,
        )
