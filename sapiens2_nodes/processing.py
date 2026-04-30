from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .constants import IMAGENET_MEAN, IMAGENET_STD, SEG_CLASS_COUNT, SEG_PALETTE, TARGET_SIZE


def _fit_resize_pad(image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
    _, h, w = image.shape
    target_h, target_w = TARGET_SIZE
    scale = min(target_w / w, target_h / h)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    meta = {
        "orig_h": h,
        "orig_w": w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }
    return padded, meta


def _prepare_inputs(
    image: torch.Tensor, task: str, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, List[Dict[str, int]]]:
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"Expected ComfyUI IMAGE tensor [B,H,W,3], got {image.shape}")
    image = image.detach().float().clamp(0, 1).permute(0, 3, 1, 2).cpu()
    metas = []
    processed = []
    if task == "segmentation":
        resized = F.interpolate(
            image,
            size=TARGET_SIZE,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        batch_size, _, h, w = image.shape
        for _ in range(batch_size):
            metas.append({"orig_h": h, "orig_w": w})
        processed = [resized[i] for i in range(resized.shape[0])]
    else:
        for sample in image:
            padded, meta = _fit_resize_pad(sample)
            processed.append(padded)
            metas.append(meta)

    batch = torch.stack(processed, dim=0) * 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=batch.dtype).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    return batch.to(device=device, dtype=dtype), metas


def _restore_spatial(pred: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
    pred_h, pred_w = pred.shape[-2:]
    target_h, target_w = TARGET_SIZE
    top = round(meta.get("pad_top", 0) * pred_h / target_h)
    bottom = round(meta.get("pad_bottom", 0) * pred_h / target_h)
    left = round(meta.get("pad_left", 0) * pred_w / target_w)
    right = round(meta.get("pad_right", 0) * pred_w / target_w)
    cropped = pred[..., top : pred_h - bottom, left : pred_w - right]
    return F.interpolate(
        cropped.unsqueeze(0),
        size=(meta["orig_h"], meta["orig_w"]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _segmentation_outputs(
    logits: torch.Tensor, metas: List[Dict[str, int]], overlay_opacity: float, source: torch.Tensor
):
    label_previews = []
    class_ids = []
    colors = []
    masks = []
    for i, meta in enumerate(metas):
        resized = F.interpolate(
            logits[i : i + 1],
            size=(meta["orig_h"], meta["orig_w"]),
            mode="bilinear",
            align_corners=False,
        )
        label = resized.argmax(dim=1).squeeze(0).cpu()
        color = SEG_PALETTE[label.clamp(0, SEG_CLASS_COUNT - 1)]
        base = source[i].detach().float().cpu().clamp(0, 1)
        vis = base * (1.0 - overlay_opacity) + color * overlay_opacity
        label_previews.append(label.float() / float(SEG_CLASS_COUNT - 1))
        class_ids.append(label.long())
        colors.append(vis.clamp(0, 1))
        masks.append((label > 0).float())
    return (
        torch.stack(colors, 0),
        torch.stack(masks, 0),
        torch.stack(label_previews, 0),
        torch.stack(class_ids, 0),
    )


def _normal_outputs(pred: torch.Tensor, metas: List[Dict[str, int]]):
    images = []
    masks = []
    normals = []
    for i, meta in enumerate(metas):
        normal = _restore_spatial(pred[i], meta)
        normal = normal / torch.linalg.vector_norm(normal, dim=0, keepdim=True).clamp(min=1e-8)
        vis = ((normal + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0).cpu()
        images.append(vis)
        masks.append(torch.ones((meta["orig_h"], meta["orig_w"]), dtype=torch.float32))
        normals.append(normal.cpu())
    return torch.stack(images, 0), torch.stack(masks, 0), torch.stack(masks, 0), torch.stack(normals, 0)


def _albedo_outputs(pred: torch.Tensor, metas: List[Dict[str, int]]):
    images = []
    masks = []
    albedos = []
    for i, meta in enumerate(metas):
        albedo = _restore_spatial(pred[i], meta).clamp(0, 1)
        images.append(albedo.permute(1, 2, 0).cpu())
        masks.append(torch.ones((meta["orig_h"], meta["orig_w"]), dtype=torch.float32))
        albedos.append(albedo.cpu())
    return torch.stack(images, 0), torch.stack(masks, 0), torch.stack(masks, 0), torch.stack(albedos, 0)


def _pointmap_outputs(pointmap: torch.Tensor, scale: torch.Tensor, metas: List[Dict[str, int]]):
    images = []
    depth_masks = []
    raw_depths = []
    pointmaps = []
    scale = scale.reshape(-1, 1, 1, 1).clamp(min=1e-8)
    pointmap = pointmap / scale
    for i, meta in enumerate(metas):
        restored = _restore_spatial(pointmap[i], meta)
        depth = restored[2].float().cpu()
        valid = torch.isfinite(depth) & (depth > 0)
        if valid.any():
            vals = depth[valid]
            lo = torch.quantile(vals, 0.01)
            hi = torch.quantile(vals, 0.99)
            norm = (depth - lo) / (hi - lo).clamp(min=1e-6)
        else:
            norm = torch.zeros_like(depth)
        gray = norm.clamp(0, 1)
        images.append(torch.stack((gray, gray, gray), dim=-1))
        depth_masks.append(valid.float())
        raw_depths.append(gray.float())
        pointmaps.append(restored.cpu())
    return (
        torch.stack(images, 0),
        torch.stack(depth_masks, 0),
        torch.stack(raw_depths, 0),
        torch.stack(pointmaps, 0),
    )


def _require_result(raw: Dict[str, Any], task: str) -> None:
    if not isinstance(raw, dict):
        raise TypeError("Expected SAPIENS2_RESULT from Sapiens2 Dense Inference.")
    if raw.get("task") != task:
        raise ValueError(f"Expected {task} result, got {raw.get('task')!r}.")


def _seg_class_ids(raw: Dict[str, Any]) -> torch.Tensor:
    _require_result(raw, "segmentation")
    if "class_ids" in raw:
        return raw["class_ids"].detach().cpu().long()
    if "labels" in raw:
        return torch.round(raw["labels"].detach().cpu().float() * float(SEG_CLASS_COUNT - 1)).long().clamp(0, SEG_CLASS_COUNT - 1)
    raise KeyError("Segmentation result does not contain class_ids or labels.")


def _part_id(part: str) -> int:
    return int(part.split(":", 1)[0])


def _mask_for_part(class_ids: torch.Tensor, part: str) -> torch.Tensor:
    return (class_ids == _part_id(part)).float()


def _pool_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels == 0:
        return mask
    kernel = abs(pixels) * 2 + 1
    x = mask.unsqueeze(1).float()
    if pixels > 0:
        x = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=abs(pixels))
    else:
        x = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=abs(pixels))
    return x.squeeze(1).clamp(0, 1)


def _process_mask(mask: torch.Tensor, grow_pixels: int = 0, blur_pixels: int = 0, invert: bool = False):
    mask = _pool_mask(mask.float().clamp(0, 1), int(grow_pixels))
    blur_pixels = int(blur_pixels)
    if blur_pixels > 0:
        kernel = blur_pixels * 2 + 1
        mask = F.avg_pool2d(
            mask.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=blur_pixels,
        ).squeeze(1)
    if invert:
        mask = 1.0 - mask
    return mask.clamp(0, 1)


def _threshold_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0:
        return mask.float().clamp(0, 1)
    return (mask.float() >= float(threshold)).float()


def _combine_channels(channels: List[torch.Tensor], mode: str) -> torch.Tensor:
    if not channels:
        raise ValueError("Select at least one channel.")
    stack = torch.stack([channel.float().clamp(0, 1) for channel in channels], dim=0)
    if mode == "average":
        return stack.mean(dim=0)
    if mode == "min":
        return stack.min(dim=0).values
    return stack.max(dim=0).values


def _normalize_mask_channel(channel: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    channel = channel.detach().cpu().float()
    if channel.ndim == 2:
        channel = channel.unsqueeze(0)
    output = []
    for idx, sample in enumerate(channel):
        if valid is not None:
            sample_valid = valid[idx].detach().cpu().bool()
        else:
            sample_valid = torch.isfinite(sample)
        if sample_valid.any():
            vals = sample[sample_valid]
            lo = vals.min()
            hi = vals.max()
            norm = (sample - lo) / (hi - lo).clamp(min=1e-6)
            if valid is not None:
                norm = torch.where(sample_valid, norm, torch.zeros_like(norm))
        else:
            norm = torch.zeros_like(sample)
        output.append(norm.clamp(0, 1))
    return torch.stack(output, 0)


def _signed_to_mask(channel: torch.Tensor) -> torch.Tensor:
    return ((channel.detach().cpu().float() + 1.0) * 0.5).clamp(0, 1)
