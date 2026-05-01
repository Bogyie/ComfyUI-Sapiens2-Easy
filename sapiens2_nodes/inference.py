import numpy as np
import torch
import torch.nn.functional as F

from .constants import SEG_CLASS_COUNT, SEG_PALETTE
from .progress import NodeProgress
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

        if model.task == "segmentation":
            vis, fg, labels, class_ids = _run_segmentation(model, source, overlay_opacity)
            raw = {
                "task": model.task,
                "class_ids": class_ids,
                "labels": labels,
                "checkpoint": model.checkpoint_path,
                "config": model.config_path,
            }
            if input_mask is not None:
                vis, fg, labels = _apply_optional_mask(
                    vis, fg, labels, source, input_mask, preserve_background
                )
            return (vis, fg, labels, raw)

        if model.task == "normal":
            vis, fg, aux, normal = _run_normal(model, source)
            raw = {
                "task": model.task,
                "normal": normal,
                "checkpoint": model.checkpoint_path,
                "config": model.config_path,
            }
            if input_mask is not None:
                vis, fg, aux = _apply_optional_mask(
                    vis, fg, aux, source, input_mask, preserve_background
                )
            return (vis, fg, aux, raw)

        if model.task == "pointmap":
            vis, fg, depth, pointmap, scale = _run_pointmap(model, source)
            raw = {
                "task": model.task,
                "pointmap": pointmap,
                "depth_preview": depth,
                "scale": scale,
                "checkpoint": model.checkpoint_path,
                "config": model.config_path,
            }
            if input_mask is not None:
                vis, fg, depth = _apply_optional_mask(
                    vis, fg, depth, source, input_mask, preserve_background
                )
            return (vis, fg, depth, raw)

        raise ValueError(f"Unsupported Sapiens2 task: {model.task}")


def _to_bgr_uint8(image: torch.Tensor) -> np.ndarray:
    rgb = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return rgb[:, :, ::-1].copy()


def _run_pipeline(model: Sapiens2Model, image_rgb: torch.Tensor) -> dict:
    data = model.model.pipeline(dict(img=_to_bgr_uint8(image_rgb)))
    data = model.model.data_preprocessor(data)
    if model.dtype != torch.float32:
        data["inputs"] = data["inputs"].to(dtype=model.dtype)
    return data


def _padding_from_data(data: dict) -> tuple[int, int, int, int]:
    padding = data["data_samples"]["meta"].get("padding_size", (0, 0, 0, 0))
    if isinstance(padding, torch.Tensor):
        padding = padding.detach().cpu().tolist()
    return tuple(int(value) for value in padding)


def _crop_padding(pred: torch.Tensor, data: dict) -> torch.Tensor:
    inputs = data["inputs"]
    pad_left, pad_right, pad_top, pad_bottom = _padding_from_data(data)
    return pred[
        :,
        :,
        pad_top : inputs.shape[2] - pad_bottom,
        pad_left : inputs.shape[3] - pad_right,
    ]


def _resize_to_image(pred: torch.Tensor, image_rgb: torch.Tensor, crop_padding: bool, data: dict) -> torch.Tensor:
    if crop_padding:
        pred = _crop_padding(pred, data)
    return F.interpolate(
        pred,
        size=(int(image_rgb.shape[0]), int(image_rgb.shape[1])),
        mode="bilinear",
        align_corners=False,
    )


def _run_segmentation(model: Sapiens2Model, image_batch: torch.Tensor, opacity: float):
    previews = []
    masks = []
    labels_batch = []
    class_ids_batch = []
    palette = SEG_PALETTE
    progress = NodeProgress(len(image_batch))

    for image_rgb in image_batch:
        data = _run_pipeline(model, image_rgb)
        with torch.inference_mode():
            logits = model.model(data["inputs"])
        logits = _resize_to_image(logits.float(), image_rgb, crop_padding=False, data=data)
        class_ids = logits.argmax(dim=1).squeeze(0).detach().cpu().long()
        color = palette[class_ids.clamp(0, SEG_CLASS_COUNT - 1)]
        preview = (image_rgb * (1.0 - opacity) + color * opacity).clamp(0, 1)
        previews.append(preview)
        masks.append((class_ids > 0).float())
        labels_batch.append(class_ids.float() / float(SEG_CLASS_COUNT - 1))
        class_ids_batch.append(class_ids)
        progress.update()

    return (
        torch.stack(previews, 0),
        torch.stack(masks, 0),
        torch.stack(labels_batch, 0),
        torch.stack(class_ids_batch, 0),
    )


def _run_normal(model: Sapiens2Model, image_batch: torch.Tensor):
    previews = []
    masks = []
    normals = []
    progress = NodeProgress(len(image_batch))
    for image_rgb in image_batch:
        data = _run_pipeline(model, image_rgb)
        with torch.inference_mode():
            normal = model.model(data["inputs"])
        normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)
        normal = _resize_to_image(normal.float(), image_rgb, crop_padding=True, data=data)
        normal = normal.squeeze(0).detach().cpu()
        preview = ((normal.movedim(0, -1) + 1.0) * 0.5).clamp(0, 1)
        previews.append(preview)
        masks.append(torch.ones(image_rgb.shape[:2], dtype=torch.float32))
        normals.append(normal)
        progress.update()
    mask_batch = torch.stack(masks, 0)
    return torch.stack(previews, 0), mask_batch, mask_batch, torch.stack(normals, 0)


def _run_pointmap(model: Sapiens2Model, image_batch: torch.Tensor):
    previews = []
    masks = []
    depths = []
    pointmaps = []
    scales = []
    progress = NodeProgress(len(image_batch))
    for image_rgb in image_batch:
        data = _run_pipeline(model, image_rgb)
        with torch.inference_mode():
            pointmap, scale = model.model(data["inputs"])
        scale = scale.float().reshape(-1, 1, 1, 1).clamp(min=1e-8)
        pointmap = pointmap.float() / scale
        pointmap = _resize_to_image(pointmap, image_rgb, crop_padding=True, data=data)
        pointmap = pointmap.squeeze(0).detach().cpu()
        depth = pointmap[2].float()
        valid = torch.isfinite(depth) & (depth > 0)
        preview = _depth_preview(depth, valid)
        previews.append(preview.unsqueeze(-1).repeat(1, 1, 3))
        masks.append(valid.float())
        depths.append(preview)
        pointmaps.append(pointmap)
        scales.append(scale.detach().cpu().reshape(-1)[0])
        progress.update()
    return (
        torch.stack(previews, 0),
        torch.stack(masks, 0),
        torch.stack(depths, 0),
        torch.stack(pointmaps, 0),
        torch.stack(scales, 0),
    )


def _depth_preview(depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return torch.zeros_like(depth, dtype=torch.float32)
    values = depth[valid]
    lo = torch.quantile(values, 0.01)
    hi = torch.quantile(values, 0.99)
    return ((depth - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)


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
        raise ValueError(
            "Expected optional mask shape [H,W], [B,H,W], [B,H,W,C], "
            f"or [B,C,H,W], got {tuple(mask.shape)}"
        )
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.repeat(image.shape[0], 1, 1)
    if mask.shape[0] != image.shape[0]:
        raise ValueError(
            f"Optional mask batch size ({mask.shape[0]}) does not match image batch size ({image.shape[0]})."
        )
    if mask.shape[-2:] != image.shape[1:3]:
        mask = F.interpolate(mask.unsqueeze(1), size=image.shape[1:3], mode="nearest").squeeze(1)
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
