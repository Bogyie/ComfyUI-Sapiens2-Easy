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
    config_path: str = ""


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
    repo_path: str = ""
    detector_config_path: str = ""
    task: str = "pose"
