import os
from pathlib import Path
from typing import Tuple

from .constants import MODEL_SIZE_CHOICES, POSE_DETECTOR_REPO, SAPIENS2_HF_ORG
from .folders import get_model_root


HF_TASKS = {
    "segmentation": {"repo_task": "seg", "folder": "seg"},
    "normal": {"repo_task": "normal", "folder": "normal"},
    "pointmap": {"repo_task": "pointmap", "folder": "pointmap"},
    "pose": {"repo_task": "pose", "folder": "pose"},
}
HF_DOWNLOAD_TASKS = tuple(HF_TASKS)


def _size_to_arch(model_size: str) -> str:
    return f"sapiens2_{model_size}"


def _normalize_size_for_hf(model_size: str) -> str:
    return model_size


def _normalize_size_for_file(model_size: str) -> str:
    return model_size


def _validate_task_size(task: str, model_size: str) -> None:
    if model_size not in MODEL_SIZE_CHOICES:
        raise ValueError(
            f"Official {task} checkpoints are not listed for size {model_size}. "
            "Use 0.4b, 0.8b, 1b, or 5b."
        )


def _default_repo_and_filename(task: str, model_size: str) -> Tuple[str, str]:
    meta = HF_TASKS.get(task)
    if not meta:
        raise ValueError(f"Unsupported Hugging Face task: {task}")
    _validate_task_size(task, model_size)
    repo_task = str(meta["repo_task"])
    hf_size = _normalize_size_for_hf(model_size)
    file_size = _normalize_size_for_file(model_size)
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
    if model_size not in MODEL_SIZE_CHOICES:
        raise ValueError(f"Unsupported Sapiens2 model size: {model_size}")
    resolved_repo = repo_id.strip()
    resolved_filename = filename.strip()
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


def _download_context(kwargs) -> str:
    repo_id = kwargs.get("repo_id", "<unknown repo>")
    filename = kwargs.get("filename")
    revision = kwargs.get("revision")
    if filename:
        target = f"{repo_id}/{filename}"
    else:
        target = str(repo_id)
    if revision:
        target = f"{target}@{revision}"
    return target


def _comfy_tqdm_class():
    try:
        import comfy.utils
        from huggingface_hub.utils import tqdm as hf_tqdm
    except Exception:
        return None

    class ComfyTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._comfy_progress = None
            if self.total:
                try:
                    self._comfy_progress = comfy.utils.ProgressBar(int(self.total))
                except Exception:
                    self._comfy_progress = None

        def update(self, n=1):
            result = super().update(n)
            if self._comfy_progress is not None and n:
                try:
                    self._comfy_progress.update(int(n))
                except Exception:
                    self._comfy_progress = None
            return result

        def reset(self, total=None):
            result = super().reset(total=total)
            self._comfy_progress = None
            if self.total:
                try:
                    self._comfy_progress = comfy.utils.ProgressBar(int(self.total))
                except Exception:
                    self._comfy_progress = None
            return result

    return ComfyTqdm


def _call_download(download_fn, kwargs):
    call_kwargs = dict(kwargs)
    if call_kwargs.get("tqdm_class") is None:
        call_kwargs.pop("tqdm_class", None)
    try:
        return download_fn(**call_kwargs)
    except TypeError as exc:
        if "tqdm_class" not in str(exc):
            raise
        call_kwargs.pop("tqdm_class", None)
        return download_fn(**call_kwargs)


def _download_with_hf_client_retry(download_fn, **kwargs):
    try:
        return _call_download(download_fn, kwargs)
    except RuntimeError as exc:
        if "client has been closed" not in str(exc):
            raise
        try:
            from huggingface_hub.utils import _http

            _http.close_session()
        except Exception:
            pass
        try:
            return _call_download(download_fn, kwargs)
        except Exception as retry_exc:
            raise RuntimeError(
                "Hugging Face download failed after resetting a closed HTTP client "
                f"for {_download_context(kwargs)}. First error: {exc}. Retry error: {retry_exc}"
            ) from retry_exc


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
            "Could not import huggingface_hub for Hugging Face downloads. "
            f"Original import error: {exc}"
        ) from exc

    resolved_repo, resolved_filename, resolved_local_dir = _resolve_hf_args(
        task, model_size, repo_id, filename, local_dir
    )
    path = _download_with_hf_client_retry(
        hf_hub_download,
        repo_id=resolved_repo,
        filename=resolved_filename,
        revision=revision.strip() or "main",
        local_dir=resolved_local_dir,
        token=token.strip() or None,
        force_download=force_download,
        local_files_only=local_files_only,
        tqdm_class=_comfy_tqdm_class(),
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
            "Could not import huggingface_hub for Hugging Face downloads. "
            f"Original import error: {exc}"
        ) from exc

    resolved_repo = repo_id.strip() or POSE_DETECTOR_REPO
    resolved_local_dir = local_dir.strip()
    if resolved_local_dir:
        resolved_local_dir = os.path.expanduser(os.path.expandvars(resolved_local_dir))
    else:
        resolved_local_dir = str(get_model_root() / "detector" / resolved_repo.rsplit("/", 1)[-1])
    Path(resolved_local_dir).mkdir(parents=True, exist_ok=True)
    path = _download_with_hf_client_retry(
        snapshot_download,
        repo_id=resolved_repo,
        revision=revision.strip() or "main",
        local_dir=resolved_local_dir,
        token=token.strip() or None,
        force_download=force_download,
        local_files_only=local_files_only,
        tqdm_class=_comfy_tqdm_class(),
    )
    return path, resolved_repo
