import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from .constants import ARCH_CHOICES, ARCH_SPECS, DEVICES, DTYPES, PATCH_SIZE, SEG_CLASS_COUNT, TARGET_SIZE, TASK_CHOICES
from .folders import MODEL_FOLDER_NAMES, get_filename_list, get_full_path
from .types import Sapiens2Model


_MODEL_CACHE: dict[tuple[str, str, str, str, str, str, int, int], Sapiens2Model] = {}


def _checkpoint_names() -> List[str]:
    names = get_filename_list(MODEL_FOLDER_NAMES)
    return names or ["put_sapiens2_checkpoints_in_ComfyUI_models_sapiens2"]


def _resolve_checkpoint(checkpoint_name: str, checkpoint_path: str) -> str:
    explicit = checkpoint_path.strip()
    if explicit:
        path = os.path.expanduser(os.path.expandvars(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Sapiens2 checkpoint not found: {path}")
        return path

    path = get_full_path(MODEL_FOLDER_NAMES, checkpoint_name)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            "Sapiens2 checkpoint not found. Put .safetensors files under "
            "ComfyUI/models/sapiens2 or set checkpoint_path."
        )
    return path


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
    for candidate in _candidate_repo_paths(repo_path):
        if (candidate / "sapiens").is_dir():
            candidate_path = str(candidate)
            if candidate_path not in sys.path:
                sys.path.insert(0, candidate_path)
            break

    try:
        importlib.import_module("sapiens.backbones")
        importlib.import_module("sapiens.dense")
    except Exception as exc:
        raise RuntimeError(
            "Could not import the official Sapiens2 code. Clone "
            "https://github.com/facebookresearch/sapiens2 into this custom node's "
            "vendor/sapiens2 directory, set SAPIENS2_REPO, or pass sapiens_repo_path."
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
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
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
            if all(k.startswith(prefix) for k in state_dict.keys()):
                state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
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
        "albedo": "decode_head.conv_albedo.weight",
    }
    for task, key in task_keys.items():
        if key in state_dict:
            return task
    if "decode_head.conv_pose.weight" in state_dict:
        raise ValueError(
            "This checkpoint is a Sapiens2 pose model. Pose requires detector/top-down "
            "support and is not handled by the dense model loader yet."
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


def _extract_upsample_channels(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    channels = []
    index = 0
    while f"decode_head.upsample_blocks.{index}.0.weight" in state_dict:
        channels.append(state_dict[f"decode_head.upsample_blocks.{index}.0.weight"].shape[0] // 4)
        index += 1
    return channels


def _extract_conv_layers(state_dict: Dict[str, torch.Tensor], prefix: str) -> tuple[List[int], List[int]]:
    channels = []
    kernels = []
    index = 0
    while f"{prefix}.{index}.weight" in state_dict:
        weight = state_dict[f"{prefix}.{index}.weight"]
        channels.append(weight.shape[0])
        kernels.append(weight.shape[2])
        index += 3
    return channels, kernels


def _extract_deconv_layers(state_dict: Dict[str, torch.Tensor]) -> tuple[List[int], List[int]]:
    channels = []
    kernels = []
    index = 0
    while f"decode_head.deconv_layers.{index}.weight" in state_dict:
        weight = state_dict[f"decode_head.deconv_layers.{index}.weight"]
        channels.append(weight.shape[1])
        kernels.append(weight.shape[2])
        index += 3
    return channels, kernels


def _extract_scale_final_layer(state_dict: Dict[str, torch.Tensor]) -> tuple[int, ...]:
    first_key = "decode_head.scale_final_layer.1.weight"
    if first_key not in state_dict:
        return ()
    layers = [state_dict[first_key].shape[1]]
    index = 1
    while f"decode_head.scale_final_layer.{index}.weight" in state_dict:
        layers.append(state_dict[f"decode_head.scale_final_layer.{index}.weight"].shape[0])
        index += 2
    return tuple(layers)


def _build_model_config(
    task: str,
    arch: str,
    state_dict: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, Any]:
    embed_dim = ARCH_SPECS[arch]["embed_dim"]
    backbone = {
        "type": "Sapiens2",
        "arch": arch,
        "img_size": TARGET_SIZE,
        "patch_size": PATCH_SIZE,
        "final_norm": True,
        "use_tokenizer": False,
        "with_cls_token": True,
        "out_type": "featmap",
    }

    if task == "segmentation":
        deconv_channels = (512, 256, 128, 64)
        deconv_kernels = (4, 4, 4, 4)
        conv_channels = (64, 64)
        conv_kernels = (1, 1)
        num_classes = SEG_CLASS_COUNT
        if state_dict is not None:
            detected_deconv_channels, detected_deconv_kernels = _extract_deconv_layers(state_dict)
            detected_conv_channels, detected_conv_kernels = _extract_conv_layers(
                state_dict, "decode_head.conv_layers"
            )
            deconv_channels = tuple(detected_deconv_channels or deconv_channels)
            deconv_kernels = tuple(detected_deconv_kernels or deconv_kernels)
            conv_channels = tuple(detected_conv_channels or conv_channels)
            conv_kernels = tuple(detected_conv_kernels or conv_kernels)
            if "decode_head.conv_seg.weight" in state_dict:
                num_classes = state_dict["decode_head.conv_seg.weight"].shape[0]
        return {
            "type": "SegEstimator",
            "backbone": backbone,
            "decode_head": {
                "type": "SegHead",
                "in_channels": embed_dim,
                "deconv_out_channels": deconv_channels,
                "deconv_kernel_sizes": deconv_kernels,
                "conv_out_channels": conv_channels,
                "conv_kernel_sizes": conv_kernels,
                "num_classes": num_classes,
            },
        }

    is_5b = arch == "sapiens2_5b"
    if task == "normal":
        upsample_channels = [1536, 768, 512, 256] if is_5b else [768, 512, 256, 128]
        conv_out_channels = [128, 64, 32] if is_5b else [64, 32, 16]
        conv_kernel_sizes = [3, 3, 3]
        if state_dict is not None:
            upsample_channels = _extract_upsample_channels(state_dict) or upsample_channels
            detected_conv_channels, detected_conv_kernels = _extract_conv_layers(
                state_dict, "decode_head.conv_layers"
            )
            conv_out_channels = detected_conv_channels or conv_out_channels
            conv_kernel_sizes = detected_conv_kernels or conv_kernel_sizes
        return {
            "type": "NormalEstimator",
            "backbone": backbone,
            "decode_head": {
                "type": "NormalHead",
                "in_channels": embed_dim,
                "upsample_channels": upsample_channels,
                "conv_out_channels": conv_out_channels,
                "conv_kernel_sizes": conv_kernel_sizes,
            },
        }

    if task == "pointmap":
        num_tokens = (TARGET_SIZE[0] // PATCH_SIZE) * (TARGET_SIZE[1] // PATCH_SIZE)
        upsample_channels = [1536, 768, 768, 768] if is_5b else [1536, 768, 512, 256]
        conv_out_channels = [128, 64, 32] if is_5b else [64, 32, 16]
        conv_kernel_sizes = [3, 3, 3]
        scale_conv_out_channels = (1536, 512, 128)
        scale_conv_kernel_sizes = (1, 1, 1)
        scale_final_layer = ((num_tokens // 64) * 128, 512, 128, 1)
        if state_dict is not None:
            upsample_channels = _extract_upsample_channels(state_dict) or upsample_channels
            detected_conv_channels, detected_conv_kernels = _extract_conv_layers(
                state_dict, "decode_head.conv_layers"
            )
            conv_out_channels = detected_conv_channels or conv_out_channels
            conv_kernel_sizes = detected_conv_kernels or conv_kernel_sizes
            detected_scale_channels, detected_scale_kernels = _extract_conv_layers(
                state_dict, "decode_head.scale_conv_layers"
            )
            scale_conv_out_channels = tuple(detected_scale_channels or scale_conv_out_channels)
            scale_conv_kernel_sizes = tuple(detected_scale_kernels or scale_conv_kernel_sizes)
            scale_final_layer = _extract_scale_final_layer(state_dict) or scale_final_layer
        return {
            "type": "PointmapEstimator",
            "canonical_focal_length": 768.0,
            "backbone": backbone,
            "decode_head": {
                "type": "PointmapHead",
                "in_channels": embed_dim,
                "upsample_channels": upsample_channels,
                "conv_out_channels": conv_out_channels,
                "conv_kernel_sizes": conv_kernel_sizes,
                "scale_conv_out_channels": scale_conv_out_channels,
                "scale_conv_kernel_sizes": scale_conv_kernel_sizes,
                "scale_final_layer": scale_final_layer,
            },
        }

    if task == "albedo":
        upsample_channels = [1536, 768, 512, 256] if is_5b else [768, 512, 256, 128]
        conv_out_channels = [64, 32, 16]
        conv_kernel_sizes = [3, 3, 3]
        if state_dict is not None:
            upsample_channels = _extract_upsample_channels(state_dict) or upsample_channels
            detected_conv_channels, detected_conv_kernels = _extract_conv_layers(
                state_dict, "decode_head.conv_layers"
            )
            conv_out_channels = detected_conv_channels or conv_out_channels
            conv_kernel_sizes = detected_conv_kernels or conv_kernel_sizes
        return {
            "type": "AlbedoEstimator",
            "backbone": backbone,
            "decode_head": {
                "type": "AlbedoHead",
                "in_channels": embed_dim,
                "upsample_channels": upsample_channels,
                "conv_out_channels": conv_out_channels,
                "conv_kernel_sizes": conv_kernel_sizes,
            },
        }

    raise ValueError(f"Unsupported task: {task}")


def _load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
):
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"[Sapiens2-ComfyUI] Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[Sapiens2-ComfyUI] Unexpected keys: {len(incompatible.unexpected_keys)}")
    model.to(device)
    model.eval()


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

    _ensure_sapiens_importable(sapiens_repo_path)
    from sapiens.registry import MODELS

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    state_dict = _read_checkpoint_state_dict(checkpoint_path)
    task, arch = _resolve_task_arch(task, arch, state_dict)

    model = MODELS.build(_build_model_config(task, arch, state_dict=state_dict))
    _load_state_dict(model, state_dict, resolved_device)
    if resolved_dtype != torch.float32:
        model.to(dtype=resolved_dtype)

    loaded = Sapiens2Model(
        model=model,
        task=task,
        arch=arch,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    _MODEL_CACHE[cache_key] = loaded
    return loaded


class Sapiens2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_name": (_checkpoint_names(),),
            },
            "optional": {
                "task": (TASK_CHOICES,),
                "arch": (ARCH_CHOICES,),
                "device": (DEVICES, {"default": "auto"}),
                "dtype": (DTYPES, {"default": "auto"}),
                "checkpoint_path": ("STRING", {"default": "", "multiline": False}),
                "sapiens_repo_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("SAPIENS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Sapiens2"

    def load(
        self,
        checkpoint_name: str,
        task: str = "auto",
        arch: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        checkpoint_path: str = "",
        sapiens_repo_path: str = "",
    ):
        resolved_checkpoint = _resolve_checkpoint(checkpoint_name, checkpoint_path)
        return (
            load_sapiens2_model(
                task=task,
                arch=arch,
                device=device,
                dtype=dtype,
                checkpoint_path=resolved_checkpoint,
                sapiens_repo_path=sapiens_repo_path,
            ),
        )
