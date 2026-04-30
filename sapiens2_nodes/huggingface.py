import os
from pathlib import Path
from typing import Tuple

from .constants import DOWNLOAD_MODEL_SIZE_CHOICES, POSE_DETECTOR_REPO, SAPIENS2_HF_ORG
from .folders import get_model_root


HF_TASKS = {
    "pretrain": {"repo_task": "pretrain", "folder": "pretrain", "loadable": False},
    "segmentation": {"repo_task": "seg", "folder": "seg", "loadable": True},
    "normal": {"repo_task": "normal", "folder": "normal", "loadable": True},
    "pointmap": {"repo_task": "pointmap", "folder": "pointmap", "loadable": True},
    "pose": {"repo_task": "pose", "folder": "pose", "loadable": False},
    "albedo_custom": {"repo_task": None, "folder": "albedo", "loadable": True, "custom": True},
}
HF_DOWNLOAD_TASKS = tuple(HF_TASKS)
HF_LOAD_TASKS = tuple(task for task, meta in HF_TASKS.items() if meta["loadable"])


def _size_to_arch(model_size: str) -> str:
    if model_size == "1b_4k":
        return "sapiens2_1b"
    return f"sapiens2_{model_size}"


def _normalize_size_for_hf(model_size: str) -> str:
    return "1b-4k" if model_size == "1b_4k" else model_size


def _normalize_size_for_file(model_size: str) -> str:
    return "1b_4k" if model_size == "1b_4k" else model_size


def _validate_task_size(task: str, model_size: str) -> None:
    if task == "pretrain":
        return
    if model_size in {"0.1b", "1b_4k"}:
        raise ValueError(
            f"Official {task} checkpoints are not listed for size {model_size}. "
            "Use 0.4b, 0.8b, 1b, or 5b, or provide repo_id and filename manually."
        )


def _default_repo_and_filename(task: str, model_size: str) -> Tuple[str, str]:
    meta = HF_TASKS.get(task)
    if not meta:
        raise ValueError(f"Unsupported Hugging Face task: {task}")
    if meta.get("custom"):
        raise ValueError("albedo_custom requires explicit repo_id and filename.")
    _validate_task_size(task, model_size)
    repo_task = str(meta["repo_task"])
    hf_size = _normalize_size_for_hf(model_size)
    file_size = _normalize_size_for_file(model_size)
    if task == "pretrain":
        return (
            f"{SAPIENS2_HF_ORG}/sapiens2-pretrain-{hf_size}",
            f"sapiens2_{file_size}_pretrain.safetensors",
        )
    return (
        f"{SAPIENS2_HF_ORG}/sapiens2-{repo_task}-{hf_size}",
        f"sapiens2_{file_size}_{repo_task}.safetensors",
    )


def _default_local_dir(task: str) -> str:
    subdir = HF_TASKS.get(task, {}).get("folder", task)
    root = get_model_root() / subdir
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _resolve_hf_args(task: str, model_size: str, repo_id: str, filename: str, local_dir: str):
    if task not in HF_DOWNLOAD_TASKS:
        raise ValueError(f"Unsupported Hugging Face task: {task}")
    if model_size not in DOWNLOAD_MODEL_SIZE_CHOICES:
        raise ValueError(f"Unsupported Sapiens2 model size: {model_size}")
    resolved_repo = repo_id.strip()
    resolved_filename = filename.strip()
    if task == "albedo_custom" and bool(resolved_repo) != bool(resolved_filename):
        raise ValueError("albedo_custom requires both repo_id and filename.")
    if not resolved_repo or not resolved_filename:
        default_repo, default_filename = _default_repo_and_filename(task, model_size)
        resolved_repo = resolved_repo or default_repo
        resolved_filename = resolved_filename or default_filename

    resolved_local_dir = local_dir.strip()
    if resolved_local_dir:
        resolved_local_dir = os.path.expanduser(os.path.expandvars(resolved_local_dir))
    else:
        resolved_local_dir = _default_local_dir(task)
    Path(resolved_local_dir).mkdir(parents=True, exist_ok=True)
    return resolved_repo, resolved_filename, resolved_local_dir


def download_sapiens2_from_hf(
    task: str,
    model_size: str,
    repo_id: str = "",
    filename: str = "",
    local_dir: str = "",
    revision: str = "main",
    token: str = "",
    force_download: bool = False,
    local_files_only: bool = False,
) -> Tuple[str, str, str]:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face downloads. "
            "Install this custom node's requirements.txt in the ComfyUI environment."
        ) from exc

    resolved_repo, resolved_filename, resolved_local_dir = _resolve_hf_args(
        task, model_size, repo_id, filename, local_dir
    )
    path = hf_hub_download(
        repo_id=resolved_repo,
        filename=resolved_filename,
        revision=revision.strip() or "main",
        local_dir=resolved_local_dir,
        token=token.strip() or None,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return path, resolved_repo, resolved_filename


def download_sapiens2_pose_detector_from_hf(
    repo_id: str = "",
    local_dir: str = "",
    revision: str = "main",
    token: str = "",
    force_download: bool = False,
    local_files_only: bool = False,
) -> Tuple[str, str]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face downloads. "
            "Install this custom node's requirements.txt in the ComfyUI environment."
        ) from exc

    resolved_repo = repo_id.strip() or POSE_DETECTOR_REPO
    resolved_local_dir = local_dir.strip()
    if resolved_local_dir:
        resolved_local_dir = os.path.expanduser(os.path.expandvars(resolved_local_dir))
    else:
        resolved_local_dir = str(get_model_root() / "detector" / resolved_repo.rsplit("/", 1)[-1])
    Path(resolved_local_dir).mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=resolved_repo,
        revision=revision.strip() or "main",
        local_dir=resolved_local_dir,
        token=token.strip() or None,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return path, resolved_repo
