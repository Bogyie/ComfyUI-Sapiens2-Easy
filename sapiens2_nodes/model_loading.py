import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

from .constants import ARCH_SPECS
from .progress import NodeProgress
from .types import Sapiens2Model


_MODEL_CACHE: dict[tuple[str, str, str, str, str, str, int, int], Sapiens2Model] = {}

_DENSE_CONFIG_TEMPLATES = {
    "segmentation": (
        "sapiens/dense/configs/seg/shutterstock_goliath/"
        "{arch}_seg_shutterstock_goliath-1024x768.py"
    ),
    "normal": (
        "sapiens/dense/configs/normal/metasim_render_people/"
        "{arch}_normal_metasim_render_people-1024x768.py"
    ),
    "pointmap": (
        "sapiens/dense/configs/pointmap/render_people/"
        "{arch}_pointmap_render_people-1024x768.py"
    ),
}


def _candidate_repo_paths(repo_path: str) -> List[Path]:
    custom_node_root = Path(__file__).resolve().parents[1]
    candidates = []
    if repo_path.strip():
        candidates.append(Path(os.path.expanduser(os.path.expandvars(repo_path))))
    env_path = os.environ.get("SAPIENS2_REPO", "")
    if env_path:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(env_path))))
    candidates.extend(
        [
            custom_node_root / "vendor" / "sapiens2",
            Path.home() / "sapiens2",
            Path.home() / "repo" / "github.com" / "facebookresearch" / "sapiens2",
        ]
    )
    return candidates


def _ensure_sapiens_importable(repo_path: str) -> None:
    searched = _candidate_repo_paths(repo_path)
    for candidate in searched:
        if (candidate / "sapiens").is_dir():
            candidate_path = str(candidate)
            if candidate_path not in sys.path:
                sys.path.insert(0, candidate_path)
            break

    try:
        importlib.import_module("sapiens.backbones")
        importlib.import_module("sapiens.dense")
    except Exception as exc:
        searched_text = ", ".join(str(path) for path in searched)
        raise RuntimeError(
            "Could not import the official Sapiens2 code. Check that the repo path is "
            "correct and that its Python dependencies can import in this ComfyUI venv. "
            "You can clone https://github.com/facebookresearch/sapiens2 into this custom "
            "node's vendor/sapiens2 directory, set SAPIENS2_REPO, or pass sapiens_repo_path. "
            f"Searched: {searched_text}. Original import error: {exc}"
        ) from exc


def get_sapiens_repo_path(repo_path: str) -> Path:
    for candidate in _candidate_repo_paths(repo_path):
        if (candidate / "sapiens").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the official Sapiens2 repo. Clone "
        "https://github.com/facebookresearch/sapiens2 into this custom node's "
        "vendor/sapiens2 directory, set SAPIENS2_REPO, or pass sapiens_repo_path."
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "fp32" or device.type in ("cpu", "mps"):
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while state_dict and changed:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in state_dict):
                state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
                changed = True
                break
    return state_dict


def _read_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    return _normalize_state_dict(state_dict)


def _detect_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    if "backbone.patch_embed.projection.weight" in state_dict:
        return "backbone."
    if "patch_embed.projection.weight" in state_dict:
        return ""
    raise ValueError("Could not locate patch_embed.projection.weight in checkpoint.")


def _detect_arch(state_dict: Dict[str, torch.Tensor]) -> str:
    prefix = _detect_prefix(state_dict)
    embed_dim = state_dict[f"{prefix}patch_embed.projection.weight"].shape[0]
    for arch, spec in ARCH_SPECS.items():
        if spec["embed_dim"] == embed_dim:
            return arch
    raise ValueError(f"Unsupported Sapiens2 embed dim in checkpoint: {embed_dim}")


def _detect_task(state_dict: Dict[str, torch.Tensor]) -> str:
    task_keys = {
        "segmentation": "decode_head.conv_seg.weight",
        "normal": "decode_head.conv_normal.weight",
        "pointmap": "decode_head.conv_pointmap.weight",
    }
    for task, key in task_keys.items():
        if key in state_dict:
            return task
    if "decode_head.conv_pose.weight" in state_dict:
        raise ValueError(
            "This checkpoint is a Sapiens2 pose model. Load it with task=pose."
        )
    raise ValueError("Could not infer dense task from checkpoint decode_head keys.")


def _resolve_task_arch(
    requested_task: str,
    requested_arch: str,
    state_dict: Dict[str, torch.Tensor],
) -> tuple[str, str]:
    detected_task = _detect_task(state_dict)
    detected_arch = _detect_arch(state_dict)
    task = detected_task if requested_task == "auto" else requested_task
    arch = detected_arch if requested_arch == "auto" else requested_arch
    if task != detected_task:
        raise ValueError(
            f"Checkpoint appears to be task {detected_task!r}, but {task!r} was requested."
        )
    if arch != detected_arch:
        raise ValueError(
            f"Checkpoint appears to be arch {detected_arch!r}, but {arch!r} was requested."
        )
    return task, arch


def _dense_config_path(repo_path: str, task: str, arch: str) -> Path:
    template = _DENSE_CONFIG_TEMPLATES.get(task)
    if template is None:
        raise ValueError(f"Unsupported dense Sapiens2 task: {task}")
    config_path = get_sapiens_repo_path(repo_path) / template.format(arch=arch)
    if not config_path.is_file():
        raise FileNotFoundError(f"Expected official Sapiens2 config at: {config_path}")
    return config_path


def load_sapiens2_model(
    task: str,
    arch: str,
    device: str,
    dtype: str,
    checkpoint_path: str,
    sapiens_repo_path: str = "",
) -> Sapiens2Model:
    checkpoint_path = os.path.abspath(os.path.expanduser(os.path.expandvars(checkpoint_path)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Sapiens2 checkpoint not found: {checkpoint_path}")
    stat = os.stat(checkpoint_path)
    cache_key = (
        checkpoint_path,
        task,
        arch,
        device,
        dtype,
        sapiens_repo_path,
        stat.st_mtime_ns,
        stat.st_size,
    )
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    progress = NodeProgress(6)
    _ensure_sapiens_importable(sapiens_repo_path)
    progress.update()
    state_dict = _read_checkpoint_state_dict(checkpoint_path)
    progress.update()
    task, arch = _resolve_task_arch(task, arch, state_dict)
    config_path = _dense_config_path(sapiens_repo_path, task, arch)
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    progress.update()

    init_model = importlib.import_module("sapiens.dense.src.models.init_model").init_model
    model = init_model(str(config_path), checkpoint_path, device=str(resolved_device))
    progress.update()
    if resolved_dtype != torch.float32:
        model.to(dtype=resolved_dtype)
    progress.update()
    model.eval()

    loaded = Sapiens2Model(
        model=model,
        task=task,
        arch=arch,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        dtype=resolved_dtype,
        config_path=str(config_path),
    )
    _MODEL_CACHE[cache_key] = loaded
    progress.update()
    return loaded
