from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Sapiens2Model:
    model: torch.nn.Module
    task: str
    arch: str
    checkpoint_path: str
    device: torch.device
    dtype: torch.dtype


@dataclass
class Sapiens2PoseModel:
    model: torch.nn.Module
    arch: str
    checkpoint_path: str
    detector_path: str
    device: torch.device
    dtype: torch.dtype
    codec: Any
    metainfo: dict[str, Any]
    task: str = "pose"
