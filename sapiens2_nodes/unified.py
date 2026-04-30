import os
from pathlib import Path
from typing import Any

from .constants import DEVICES, DTYPES, MODEL_SIZE_CHOICES, POSE_DETECTOR_REPO
from .folders import get_model_root
from .huggingface import (
    download_sapiens2_from_hf,
    download_sapiens2_pose_detector_from_hf,
    _resolve_hf_args,
    _size_to_arch,
)
from .inference import Sapiens2DenseInference
from .model_loading import load_sapiens2_model
from .pose import Sapiens2PoseInference, load_sapiens2_pose_model
from .types import Sapiens2Model, Sapiens2PoseModel


TASK_CHOICES = ("segmentation", "normal", "pointmap", "pose", "albedo_custom")
MODEL_SOURCE_CHOICES = ("auto", "local_only", "download")


def model_task(task: str) -> str:
    return "albedo" if task == "albedo_custom" else task


def is_pose_model(model: Any) -> bool:
    return isinstance(model, Sapiens2PoseModel) or getattr(model, "task", None) == "pose"


def describe_model_source(task: str, checkpoint_path: str, source: str, extra: str = "") -> str:
    lines = [
        f"task: {task}",
        f"source: {source}",
        f"checkpoint: {checkpoint_path}",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _default_checkpoint_path(task: str, model_size: str, repo_id: str, filename: str, local_dir: str) -> str:
    _, resolved_filename, resolved_local_dir = _resolve_hf_args(
        task, model_size, repo_id, filename, local_dir
    )
    return str(Path(resolved_local_dir) / resolved_filename)


def _resolve_or_download_checkpoint(
    task: str,
    model_size: str,
    source: str,
    checkpoint_path: str,
    repo_id: str,
    filename: str,
    local_dir: str,
    revision: str,
    token: str,
    force_download: bool,
    local_files_only: bool,
) -> tuple[str, str, str, str]:
    if checkpoint_path.strip():
        path = _expand(checkpoint_path)
        if not Path(path).is_file():
            raise FileNotFoundError(f"Sapiens2 checkpoint not found: {path}")
        return path, "explicit_path", "", ""

    if source != "download":
        local_path = _default_checkpoint_path(task, model_size, repo_id, filename, local_dir)
        if Path(local_path).is_file() and not force_download:
            return local_path, "local_cache", "", ""
        if source == "local_only":
            raise FileNotFoundError(f"Sapiens2 checkpoint not found in local cache: {local_path}")

    path, resolved_repo, resolved_filename = download_sapiens2_from_hf(
        task=task,
        model_size=model_size,
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        revision=revision,
        token=token,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return path, "huggingface_download", resolved_repo, resolved_filename


def _resolve_or_download_detector(
    source: str,
    detector_path: str,
    detector_repo_id: str,
    detector_local_dir: str,
    revision: str,
    token: str,
    force_download: bool,
    local_files_only: bool,
) -> tuple[str, str]:
    if detector_path.strip():
        path = _expand(detector_path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Sapiens2 pose detector not found: {path}")
        return path, "explicit_detector_path"

    repo_id = detector_repo_id.strip() or POSE_DETECTOR_REPO
    local_dir = detector_local_dir.strip()
    if local_dir:
        local_dir = os.path.expanduser(os.path.expandvars(local_dir))
    else:
        local_dir = str(get_model_root() / "detector" / repo_id.rsplit("/", 1)[-1])

    if source != "download" and Path(local_dir).exists() and not force_download:
        return local_dir, "local_detector_cache"
    if source == "local_only":
        raise FileNotFoundError(f"Sapiens2 pose detector not found in local cache: {local_dir}")

    path, _ = download_sapiens2_pose_detector_from_hf(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    return path, "huggingface_detector_download"


class Sapiens2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (TASK_CHOICES,),
                "model_size": (MODEL_SIZE_CHOICES, {"default": "1b"}),
            },
            "optional": {
                "source": (MODEL_SOURCE_CHOICES,),
                "device": (DEVICES, {"default": "auto"}),
                "dtype": (DTYPES, {"default": "auto"}),
                "checkpoint_path": ("STRING", {"default": "", "multiline": False}),
                "repo_id": ("STRING", {"default": "", "multiline": False}),
                "filename": ("STRING", {"default": "", "multiline": False}),
                "local_dir": ("STRING", {"default": "", "multiline": False}),
                "detector_path": ("STRING", {"default": "", "multiline": False}),
                "detector_repo_id": ("STRING", {"default": POSE_DETECTOR_REPO, "multiline": False}),
                "detector_local_dir": ("STRING", {"default": "", "multiline": False}),
                "revision": ("STRING", {"default": "main", "multiline": False}),
                "token": ("STRING", {"default": "", "multiline": False}),
                "force_download": ("BOOLEAN", {"default": False}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "sapiens_repo_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("SAPIENS2_MODEL", "STRING")
    RETURN_NAMES = ("model", "load_info")
    FUNCTION = "load"
    CATEGORY = "Sapiens2"

    def load(
        self,
        task: str,
        model_size: str,
        source: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        checkpoint_path: str = "",
        repo_id: str = "",
        filename: str = "",
        local_dir: str = "",
        detector_path: str = "",
        detector_repo_id: str = POSE_DETECTOR_REPO,
        detector_local_dir: str = "",
        revision: str = "main",
        token: str = "",
        force_download: bool = False,
        local_files_only: bool = False,
        sapiens_repo_path: str = "",
    ):
        if task == "albedo_custom" and not checkpoint_path.strip() and not (repo_id.strip() and filename.strip()):
            raise ValueError("albedo_custom needs checkpoint_path or both repo_id and filename.")

        checkpoint, checkpoint_source, resolved_repo, resolved_filename = _resolve_or_download_checkpoint(
            task=task,
            model_size=model_size,
            source=source,
            checkpoint_path=checkpoint_path,
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            revision=revision,
            token=token,
            force_download=force_download,
            local_files_only=local_files_only,
        )

        if task == "pose":
            detector, detector_source = _resolve_or_download_detector(
                source=source,
                detector_path=detector_path,
                detector_repo_id=detector_repo_id,
                detector_local_dir=detector_local_dir,
                revision=revision,
                token=token,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            model = load_sapiens2_pose_model(
                checkpoint_path=checkpoint,
                detector_path=detector,
                model_size=model_size,
                device=device,
                dtype=dtype,
                sapiens_repo_path=sapiens_repo_path,
            )
            info = describe_model_source(
                task,
                checkpoint,
                checkpoint_source,
                f"detector_source: {detector_source}\ndetector: {detector}",
            )
            return (model, info)

        model = load_sapiens2_model(
            task=model_task(task),
            arch=_size_to_arch(model_size),
            device=device,
            dtype=dtype,
            checkpoint_path=checkpoint,
            sapiens_repo_path=sapiens_repo_path,
        )
        extra = ""
        if resolved_repo or resolved_filename:
            extra = f"repo: {resolved_repo}\nfilename: {resolved_filename}"
        return (model, describe_model_source(task, checkpoint, checkpoint_source, extra))


class Sapiens2Run:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                "overlay_opacity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_background": ("BOOLEAN", {"default": False}),
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fallback_full_image_bbox": ("BOOLEAN", {"default": True}),
                "radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "flip_test": ("BOOLEAN", {"default": True}),
                "show_points": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "SAPIENS2_RESULT")
    RETURN_NAMES = ("image", "mask", "aux_mask", "result")
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(
        self,
        model: Sapiens2Model | Sapiens2PoseModel,
        image,
        mask=None,
        overlay_opacity: float = 0.55,
        preserve_background: bool = False,
        keypoint_threshold: float = 0.3,
        bbox_threshold: float = 0.3,
        nms_threshold: float = 0.3,
        fallback_full_image_bbox: bool = True,
        radius: int = 4,
        thickness: int = 2,
        flip_test: bool = True,
        show_points: bool = True,
        show_skeleton: bool = True,
    ):
        if is_pose_model(model):
            pose_image, keypoint_mask, result = Sapiens2PoseInference().run(
                pose_model=model,
                image=image,
                keypoint_threshold=keypoint_threshold,
                bbox_threshold=bbox_threshold,
                nms_threshold=nms_threshold,
                radius=radius,
                thickness=thickness,
                fallback_full_image_bbox=fallback_full_image_bbox,
                flip_test=flip_test,
                show_points=show_points,
                show_skeleton=show_skeleton,
            )
            return (pose_image, keypoint_mask, keypoint_mask, result)

        return Sapiens2DenseInference().run(
            model=model,
            image=image,
            overlay_opacity=overlay_opacity,
            preserve_background=preserve_background,
            mask=mask,
        )


def run_result(model: Sapiens2Model | Sapiens2PoseModel, image, **kwargs) -> dict[str, Any]:
    return Sapiens2Run().run(model, image, **kwargs)[3]
