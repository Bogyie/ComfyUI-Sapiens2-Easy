import torch
import torch.nn.functional as F

from .processing import (
    _albedo_outputs,
    _normal_outputs,
    _pointmap_outputs,
    _prepare_inputs,
    _segmentation_outputs,
)
from .types import Sapiens2Model


class Sapiens2DenseInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "overlay_opacity": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "preserve_background": ("BOOLEAN", {"default": False}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "SAPIENS2_RESULT")
    RETURN_NAMES = ("image", "foreground_mask", "aux_mask", "result")
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(
        self,
        model: Sapiens2Model,
        image: torch.Tensor,
        overlay_opacity: float = 0.55,
        preserve_background: bool = False,
        mask: torch.Tensor | None = None,
    ):
        source = image.detach().float().cpu().clamp(0, 1)
        input_mask = _prepare_optional_mask(mask, source) if mask is not None else None
        inputs, metas = _prepare_inputs(image, model.task, model.device, model.dtype)

        with torch.inference_mode():
            output = model.model(inputs)

        if model.task == "segmentation":
            logits = output.float().detach().cpu()
            vis, fg, labels, class_ids = _segmentation_outputs(
                logits, metas, overlay_opacity, source
            )
            raw = {
                "task": model.task,
                "class_ids": class_ids,
                "labels": labels,
                "checkpoint": model.checkpoint_path,
            }
            if input_mask is not None:
                vis, fg, labels = _apply_optional_mask(
                    vis, fg, labels, source, input_mask, preserve_background
                )
            return (vis, fg, labels, raw)

        if model.task == "normal":
            pred = output.float().detach().cpu()
            vis, fg, aux, normal = _normal_outputs(pred, metas)
            raw = {"task": model.task, "normal": normal, "checkpoint": model.checkpoint_path}
            if input_mask is not None:
                vis, fg, aux = _apply_optional_mask(
                    vis, fg, aux, source, input_mask, preserve_background
                )
            return (vis, fg, aux, raw)

        if model.task == "albedo":
            pred = output.float().detach().cpu()
            vis, fg, aux, albedo = _albedo_outputs(pred, metas)
            raw = {"task": model.task, "albedo": albedo, "checkpoint": model.checkpoint_path}
            if input_mask is not None:
                vis, fg, aux = _apply_optional_mask(
                    vis, fg, aux, source, input_mask, preserve_background
                )
            return (vis, fg, aux, raw)

        if model.task == "pointmap":
            pointmap, scale = output
            vis, fg, depth, restored_pointmap = _pointmap_outputs(
                pointmap.float().detach().cpu(), scale.float().detach().cpu(), metas
            )
            raw = {
                "task": model.task,
                "pointmap": restored_pointmap,
                "depth_preview": depth,
                "scale": scale.detach().cpu().float(),
                "checkpoint": model.checkpoint_path,
            }
            if input_mask is not None:
                vis, fg, depth = _apply_optional_mask(
                    vis, fg, depth, source, input_mask, preserve_background
                )
            return (vis, fg, depth, raw)

        raise ValueError(f"Unsupported Sapiens2 task: {model.task}")


def _prepare_optional_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().float().cpu()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        if mask.shape[-1] in (1, 3, 4):
            mask = mask[..., 0] if mask.shape[-1] == 1 else mask.mean(dim=-1)
        elif mask.shape[1] in (1, 3, 4):
            mask = mask[:, 0] if mask.shape[1] == 1 else mask.mean(dim=1)
        else:
            raise ValueError(f"Unsupported optional mask shape: {tuple(mask.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"Expected optional mask shape [H,W], [B,H,W], [B,H,W,C], or [B,C,H,W], got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.repeat(image.shape[0], 1, 1)
    if mask.shape[0] != image.shape[0]:
        raise ValueError(
            f"Optional mask batch size ({mask.shape[0]}) does not match image batch size ({image.shape[0]})."
        )
    if mask.shape[-2:] != image.shape[1:3]:
        mask = F.interpolate(
            mask.unsqueeze(1),
            size=image.shape[1:3],
            mode="nearest",
        ).squeeze(1)
    return mask.clamp(0, 1)


def _apply_optional_mask(
    image: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    source: torch.Tensor,
    input_mask: torch.Tensor,
    preserve_background: bool,
):
    mask_image = input_mask.unsqueeze(-1)
    if preserve_background:
        image = image * mask_image + source * (1.0 - mask_image)
    else:
        image = image * mask_image
    return image, mask_a * input_mask, mask_b * input_mask
