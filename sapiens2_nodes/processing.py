from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .constants import SEG_CLASS_COUNT


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
