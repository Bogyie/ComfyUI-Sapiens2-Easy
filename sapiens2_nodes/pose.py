import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import (
    ARCH_SPECS,
    POSE_CONFIG_DATASET,
    POSE_CONFIG_RESOLUTION,
    POSE_RTMDET_CONFIG_REL,
    POSE_KEYPOINT_COUNT,
)
from .model_loading import (
    _detect_prefix,
    _ensure_sapiens_importable,
    _read_checkpoint_state_dict,
    _resolve_device,
    _resolve_dtype,
    get_sapiens_repo_path,
)
from .types import Sapiens2PoseModel


POSE_GROUPS = ("body", "face", "left_hand", "right_hand", "feet", "extra")
POSE_MODEL_CACHE: dict[tuple[str, str, str, str, str, str, int, int], Sapiens2PoseModel] = {}
DETECTOR_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


def _pose_result(model, image, radius: int = 4, threshold: float = 0.3):
    if not isinstance(model, Sapiens2PoseModel):
        raise ValueError("This node needs a pose model.")
    _, _, raw = Sapiens2PoseInference().run(
        pose_model=model,
        image=image,
        keypoint_threshold=threshold,
        bbox_threshold=0.3,
        nms_threshold=0.3,
        radius=radius,
        thickness=2,
        fallback_full_image_bbox=True,
        flip_test=True,
        show_points=True,
        show_skeleton=True,
    )
    return raw


def _config_path(sapiens_repo_path: str, arch: str) -> str:
    repo = get_sapiens_repo_path(sapiens_repo_path)
    path = (
        repo
        / "sapiens"
        / "pose"
        / "configs"
        / "keypoints308"
        / POSE_CONFIG_DATASET
        / f"{arch}_keypoints{POSE_KEYPOINT_COUNT}_{POSE_CONFIG_DATASET}-{POSE_CONFIG_RESOLUTION}.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Sapiens2 pose config not found: {path}")
    return str(path)


def _detector_config_path(sapiens_repo_path: str) -> str:
    path = get_sapiens_repo_path(sapiens_repo_path) / POSE_RTMDET_CONFIG_REL
    if path.is_file():
        return str(path)
    fallback = Path(__file__).resolve().parent / "configs" / "rtmdet_m_640-8xb32_coco-person.py"
    if not fallback.is_file():
        raise FileNotFoundError(f"Sapiens2 RTMDet config not found: {path}")
    return str(fallback)


def _detect_pose_arch(checkpoint_path: str) -> str:
    state_dict = _read_checkpoint_state_dict(checkpoint_path)
    if "decode_head.conv_pose.weight" not in state_dict:
        raise ValueError("This checkpoint does not look like a Sapiens2 pose checkpoint.")
    prefix = _detect_prefix(state_dict)
    embed_dim = state_dict[f"{prefix}patch_embed.projection.weight"].shape[0]
    for arch, spec in ARCH_SPECS.items():
        if spec["embed_dim"] == embed_dim:
            return arch
    raise ValueError(f"Unsupported Sapiens2 pose embed dim in checkpoint: {embed_dim}")


def _keypoint_names_from_metainfo(metainfo: dict[str, Any]) -> list[str]:
    id_to_name = metainfo.get("keypoint_id2name", {})
    return [str(id_to_name[index]) for index in sorted(id_to_name.keys())]


def _setup_pose_model(model, sapiens_repo_path: str):
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo

    repo = get_sapiens_repo_path(sapiens_repo_path)
    base_config = repo / "sapiens" / "pose" / "configs" / "_base_" / "keypoints308.py"
    metainfo = parse_pose_metainfo(dict(from_file=str(base_config)))
    codec_cfg = dict(model.cfg.codec)
    codec_type = codec_cfg.pop("type")
    if codec_type != "UDPHeatmap":
        raise ValueError(f"Unsupported Sapiens2 pose codec: {codec_type}")
    return UDPHeatmap(**codec_cfg), metainfo


def load_sapiens2_pose_model(
    checkpoint_path: str,
    detector_path: str,
    model_size: str,
    device: str,
    dtype: str,
    sapiens_repo_path: str = "",
) -> Sapiens2PoseModel:
    checkpoint_path = os.path.abspath(os.path.expanduser(os.path.expandvars(checkpoint_path)))
    detector_path = os.path.abspath(os.path.expanduser(os.path.expandvars(detector_path)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Sapiens2 pose checkpoint not found: {checkpoint_path}")
    if not os.path.exists(detector_path):
        raise FileNotFoundError(f"Sapiens2 pose detector not found: {detector_path}")
    stat = os.stat(checkpoint_path)
    cache_key = (
        checkpoint_path,
        detector_path,
        model_size,
        device,
        dtype,
        sapiens_repo_path,
        stat.st_mtime_ns,
        stat.st_size,
    )
    cached = POSE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _ensure_sapiens_importable(sapiens_repo_path)
    from sapiens.pose.models import init_model

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)
    arch = _detect_pose_arch(checkpoint_path) if model_size == "auto" else f"sapiens2_{model_size}"
    config_path = _config_path(sapiens_repo_path, arch)
    repo_path = str(get_sapiens_repo_path(sapiens_repo_path))
    detector_config_path = _detector_config_path(sapiens_repo_path) if os.path.isfile(detector_path) else ""
    model = init_model(config_path, checkpoint_path, device=str(resolved_device))
    codec, metainfo = _setup_pose_model(model, sapiens_repo_path)
    if int(getattr(model.cfg, "num_keypoints", 0)) != POSE_KEYPOINT_COUNT:
        raise ValueError(f"Only the official {POSE_KEYPOINT_COUNT}-keypoint Sapiens2 pose models are supported.")
    if model_size != "auto":
        detected_arch = _detect_pose_arch(checkpoint_path)
        if arch != detected_arch:
            raise ValueError(
                f"Checkpoint appears to be arch {detected_arch!r}, but {arch!r} was requested."
            )
    if resolved_dtype != torch.float32:
        model.to(dtype=resolved_dtype)
    model.eval()

    loaded = Sapiens2PoseModel(
        model=model,
        arch=arch,
        checkpoint_path=checkpoint_path,
        detector_path=detector_path,
        device=resolved_device,
        dtype=resolved_dtype,
        codec=codec,
        metainfo=metainfo,
        repo_path=repo_path,
        detector_config_path=detector_config_path,
    )
    POSE_MODEL_CACHE[cache_key] = loaded
    return loaded


def _mmdet_install_message(original_error: Exception) -> str:
    return (
        "RTMDet pose detection requires MMDetection dependencies "
        "(`mmdet`, `mmengine`, and `mmcv` or `mmcv-lite`). "
        "Install versions compatible with your ComfyUI torch/CUDA stack, then retry. "
        f"Original import error: {original_error}"
    )


def _import_mmdet_apis():
    try:
        import sys

        sys.modules["mmpretrain"] = None
        mmdet_apis = __import__("mmdet.apis", fromlist=["init_detector", "inference_detector"])
        mmdet_datasets = __import__("mmdet.datasets", fromlist=["transforms"])
    except Exception as exc:
        raise RuntimeError(_mmdet_install_message(exc)) from exc
    return mmdet_apis.init_detector, mmdet_apis.inference_detector, mmdet_datasets.transforms


def _patch_mmdet_pipeline(cfg: Any, transforms_module: Any):
    if "test_dataloader" not in cfg:
        return cfg
    available = dir(transforms_module)
    for trans in cfg.test_dataloader.dataset.pipeline:
        if isinstance(trans, dict) and trans.get("type") in available:
            trans["type"] = "mmdet." + trans["type"]
    return cfg


def _get_detector(pose_model: Sapiens2PoseModel):
    cache_key = (pose_model.detector_path, str(pose_model.device))
    cached = DETECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if os.path.isfile(pose_model.detector_path):
        init_detector, inference_detector, transforms_module = _import_mmdet_apis()
        config_path = pose_model.detector_config_path
        if not config_path:
            config_path = str(Path(pose_model.repo_path) / POSE_RTMDET_CONFIG_REL)
        detector = init_detector(config_path, pose_model.detector_path, device=str(pose_model.device))
        detector.cfg = _patch_mmdet_pipeline(detector.cfg, transforms_module)
        cached = ("rtmdet", detector, inference_detector)
        DETECTOR_CACHE[cache_key] = cached
        return cached

    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except Exception as exc:
        raise RuntimeError(
            "Could not import transformers for Sapiens2 pose person detection. "
            f"Original import error: {exc}"
        ) from exc

    processor = DetrImageProcessor.from_pretrained(pose_model.detector_path)
    detector = DetrForObjectDetection.from_pretrained(pose_model.detector_path).eval().to(pose_model.device)
    cached = ("detr", processor, detector)
    DETECTOR_CACHE[cache_key] = cached
    return cached


def _nms_boxes(boxes: np.ndarray, threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    from sapiens.pose.evaluators import nms

    return boxes[nms(boxes, threshold)]


def _detect_persons(image_rgb: np.ndarray, pose_model: Sapiens2PoseModel, bbox_threshold: float, nms_threshold: float):
    detector_kind, first, second = _get_detector(pose_model)
    if detector_kind == "rtmdet":
        detector = first
        inference_detector = second
        det_result = inference_detector(detector, image_rgb[..., ::-1].copy())
        pred = det_result.pred_instances.cpu().numpy()
        keep = np.logical_and(pred.labels == 0, pred.scores > bbox_threshold)
        if not keep.any():
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        bboxes = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)[keep]
        detected = _nms_boxes(bboxes, nms_threshold)
        return detected[:, :4].astype(np.float32), detected[:, 4].astype(np.float32)

    from PIL import Image

    processor = first
    detector = second
    inputs = processor(images=Image.fromarray(image_rgb), return_tensors="pt").to(pose_model.device)
    with torch.no_grad():
        outputs = detector(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]], device=pose_model.device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=bbox_threshold)[0]
    person_mask = results["labels"] == 1
    boxes = results["boxes"][person_mask].detach().cpu().numpy()
    scores = results["scores"][person_mask].detach().cpu().numpy().reshape(-1, 1)
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    detected = _nms_boxes(np.concatenate([boxes, scores], axis=1), nms_threshold)
    return detected[:, :4].astype(np.float32), detected[:, 4].astype(np.float32)


def _full_image_box(height: int, width: int) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, float(max(width - 1, 0)), float(max(height - 1, 0))]], dtype=torch.float32)


def _normalize_boxes(boxes, height: int, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu()
    if tensor.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    if tensor.numel() % 4 != 0:
        raise ValueError(f"BBOX values must contain groups of 4 xyxy numbers, got shape {tuple(tensor.shape)}.")
    tensor = tensor.reshape(-1, 4)
    tensor[:, 0::2] = tensor[:, 0::2].clamp(0, max(width - 1, 0))
    tensor[:, 1::2] = tensor[:, 1::2].clamp(0, max(height - 1, 0))
    x1 = torch.minimum(tensor[:, 0], tensor[:, 2])
    y1 = torch.minimum(tensor[:, 1], tensor[:, 3])
    x2 = torch.maximum(tensor[:, 0], tensor[:, 2])
    y2 = torch.maximum(tensor[:, 1], tensor[:, 3])
    return torch.stack((x1, y1, x2, y2), dim=1)


def _normalize_scores(scores, count: int) -> torch.Tensor:
    if scores is None:
        return torch.ones((count,), dtype=torch.float32)
    tensor = torch.as_tensor(scores, dtype=torch.float32).detach().cpu().reshape(-1)
    if tensor.numel() == count:
        return tensor
    if tensor.numel() == 0 and count == 0:
        return tensor
    raise ValueError(f"BBOX scores length ({tensor.numel()}) does not match boxes length ({count}).")


def _coerce_sequence(value, batch_size: int, name: str):
    if value is None:
        return [None] * batch_size
    if isinstance(value, list) and len(value) == batch_size:
        return value
    if isinstance(value, tuple) and len(value) == batch_size:
        return list(value)
    if isinstance(value, (list, tuple)) and batch_size == 1:
        return [value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        tensor = torch.as_tensor(value)
        if tensor.ndim >= 3 and tensor.shape[0] == batch_size:
            return [tensor[index] for index in range(batch_size)]
        if batch_size == 1:
            return [tensor]
    raise ValueError(f"SAPIENS2_BBOXES {name} must contain one item per input image.")


def _coerce_bboxes(raw: dict[str, Any], image_batch: torch.Tensor) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("task") != "bboxes":
        raise ValueError("Expected a SAPIENS2_BBOXES value.")
    batch_size = image_batch.shape[0]
    height = int(image_batch.shape[1])
    width = int(image_batch.shape[2])
    boxes_values = _coerce_sequence(raw.get("boxes"), batch_size, "boxes")
    score_values = _coerce_sequence(raw.get("scores"), batch_size, "scores")
    boxes_batch = []
    scores_batch = []
    for boxes, scores in zip(boxes_values, score_values):
        normalized_boxes = _normalize_boxes(boxes if boxes is not None else [], height, width)
        boxes_batch.append(normalized_boxes)
        scores_batch.append(_normalize_scores(scores, normalized_boxes.shape[0]))
    return {
        "task": "bboxes",
        "boxes": boxes_batch,
        "scores": scores_batch,
        "image_sizes": [(height, width) for _ in range(batch_size)],
        "source": str(raw.get("source", "external")),
    }


def _resolve_pose_bboxes(
    image_batch: torch.Tensor,
    pose_model: Sapiens2PoseModel,
    bboxes: dict[str, Any] | None,
    bbox_threshold: float,
    nms_threshold: float,
    fallback_full_image_bbox: bool,
) -> dict[str, Any]:
    batch_size = image_batch.shape[0]
    height = int(image_batch.shape[1])
    width = int(image_batch.shape[2])
    if bboxes is not None:
        resolved = _coerce_bboxes(bboxes, image_batch)
    else:
        boxes_batch = []
        scores_batch = []
        for batch_index in range(batch_size):
            image_rgb = _to_uint8_rgb(image_batch[batch_index])
            boxes, scores = _detect_persons(image_rgb, pose_model, bbox_threshold, nms_threshold)
            boxes_batch.append(_normalize_boxes(boxes, height, width))
            scores_batch.append(_normalize_scores(scores, len(boxes)))
        resolved = {
            "task": "bboxes",
            "boxes": boxes_batch,
            "scores": scores_batch,
            "image_sizes": [(height, width) for _ in range(batch_size)],
            "source": "detector",
        }
    used_fallback = False
    if fallback_full_image_bbox:
        for index, boxes in enumerate(resolved["boxes"]):
            if boxes.shape[0] == 0:
                resolved["boxes"][index] = _full_image_box(height, width)
                resolved["scores"][index] = torch.ones((1,), dtype=torch.float32)
                used_fallback = True
        if used_fallback and resolved["source"] == "detector":
            resolved["source"] = "detector_with_full_image_fallback"
    return resolved


def _run_pose_one(
    image_rgb: np.ndarray,
    pose_model: Sapiens2PoseModel,
    bboxes: torch.Tensor,
    flip_test: bool,
):
    image_bgr = image_rgb[..., ::-1].copy()
    if bboxes.shape[0] == 0:
        num_keypoints = int(pose_model.metainfo.get("num_keypoints", POSE_KEYPOINT_COUNT))
        return (
            [],
            [],
            np.empty((0, 4), dtype=np.float32),
            torch.empty((0, num_keypoints, 2), dtype=torch.float32),
            torch.empty((0, num_keypoints), dtype=torch.float32),
        )
    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        bbox_np = bbox.detach().cpu().numpy().astype(np.float32)
        data = pose_model.model.pipeline(
            {
                "img": image_bgr,
                "bbox": bbox_np[None],
                "bbox_score": np.ones(1, dtype=np.float32),
            }
        )
        data = pose_model.model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0).to(device=pose_model.device, dtype=pose_model.dtype)
    with torch.no_grad():
        pred = pose_model.model(inputs)
        if flip_test:
            pred_flipped = pose_model.model(inputs.flip(-1)).flip(-1)
            flip_indices = pose_model.metainfo["flip_indices"]
            if len(flip_indices) != pred_flipped.shape[1]:
                raise ValueError("Pose flip-test metadata does not match model output channels.")
            pred = (pred + pred_flipped[:, flip_indices]) / 2.0

    pred = pred.detach().cpu().float().numpy()
    keypoints = []
    keypoint_scores = []
    for index, data_samples in enumerate(data_samples_list):
        keypoints_i, scores_i = pose_model.codec.decode(pred[index])
        meta = data_samples["meta"]
        input_size = meta["input_size"]
        bbox_center = meta["bbox_center"]
        bbox_scale = meta["bbox_scale"]
        keypoints_i = keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        keypoints.append(keypoints_i[0])
        keypoint_scores.append(scores_i[0])
    keypoints_tensor = torch.from_numpy(np.asarray(keypoints, dtype=np.float32))
    scores_tensor = torch.from_numpy(np.asarray(keypoint_scores, dtype=np.float32))
    return keypoints, keypoint_scores, bboxes.detach().cpu().numpy(), keypoints_tensor, scores_tensor


def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)


def _colors(color_source, count: int, default=(255, 0, 0)):
    if color_source is None:
        return [default] * max(count, 1)
    array = np.asarray(color_source)
    if array.ndim == 2 and array.shape[1] == 3:
        return [tuple(int(v) for v in row) for row in array.tolist()]
    if array.size == 3:
        color = tuple(int(v) for v in array.reshape(-1).tolist())
        return [color] * max(count, 1)
    return [default] * max(count, 1)


def _render_pose(
    image_rgb: np.ndarray,
    keypoints,
    keypoint_scores,
    metainfo: dict[str, Any],
    keypoint_threshold: float,
    radius: int,
    thickness: int,
    show_points: bool,
    show_skeleton: bool,
) -> np.ndarray:
    import cv2

    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    skeleton = metainfo.get("skeleton_links", [])
    keypoint_colors = _colors(metainfo.get("keypoint_colors"), len(keypoints[0]) if keypoints else 0)
    link_colors = _colors(metainfo.get("skeleton_link_colors"), len(skeleton), default=(0, 255, 0))

    for points, scores in zip(keypoints, keypoint_scores):
        points = np.asarray(points, dtype=float)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if show_skeleton:
            for link_index, (src, dst) in enumerate(skeleton):
                if src >= len(points) or dst >= len(points):
                    continue
                if scores[src] < keypoint_threshold or scores[dst] < keypoint_threshold:
                    continue
                x1, y1 = np.round(points[src]).astype(int)
                x2, y2 = np.round(points[dst]).astype(int)
                if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                    continue
                cv2.line(canvas, (x1, y1), (x2, y2), link_colors[link_index % len(link_colors)], max(1, thickness), cv2.LINE_AA)
        if show_points:
            for idx, (xy, score) in enumerate(zip(points, scores)):
                if score < keypoint_threshold:
                    continue
                x, y = np.round(xy).astype(int)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(canvas, (x, y), max(1, radius), keypoint_colors[idx % len(keypoint_colors)], -1, cv2.LINE_AA)
    return canvas


def _pose_mask(
    shape: tuple[int, int],
    keypoints,
    keypoint_scores,
    keypoint_threshold: float,
    radius: int,
    metainfo: dict[str, Any],
    groups: tuple[str, ...] = POSE_GROUPS,
) -> torch.Tensor:
    import cv2

    mask = np.zeros(shape, dtype=np.float32)
    group_ids = _group_keypoint_ids(metainfo, groups)
    for points, scores in zip(keypoints, keypoint_scores):
        points = np.asarray(points, dtype=float)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        for idx in group_ids:
            if idx >= len(points) or scores[idx] < keypoint_threshold:
                continue
            x, y = np.round(points[idx]).astype(int)
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                cv2.circle(mask, (x, y), max(1, radius), 1.0, -1, cv2.LINE_AA)
    return torch.from_numpy(mask.clip(0, 1))


def _group_keypoint_ids(metainfo: dict[str, Any], groups: tuple[str, ...]) -> list[int]:
    keypoint_name2id = metainfo.get("keypoint_name2id", {})
    ids: list[int] = []
    group_names = {
        "body": metainfo.get("body_keypoint_names", []),
        "face": metainfo.get("face_keypoint_names", []),
        "left_hand": metainfo.get("left_hand_keypoint_names", []),
        "right_hand": metainfo.get("right_hand_keypoint_names", []),
        "feet": metainfo.get("foot_keypoint_names", []),
        "extra": metainfo.get("extra_keypoint_names", []),
    }
    for group in groups:
        for name in group_names.get(group, []):
            if name in keypoint_name2id:
                ids.append(int(keypoint_name2id[name]))
    return sorted(set(ids))


class Sapiens2PoseInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_model": ("SAPIENS2_POSE_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "keypoint_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "fallback_full_image_bbox": ("BOOLEAN", {"default": True}),
                "flip_test": ("BOOLEAN", {"default": True}),
                "show_points": ("BOOLEAN", {"default": True}),
                "show_skeleton": ("BOOLEAN", {"default": True}),
                "bboxes": ("SAPIENS2_BBOXES",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "SAPIENS2_POSE_RESULT")
    RETURN_NAMES = ("pose_image", "keypoint_mask", "result")
    FUNCTION = "run"
    CATEGORY = "Sapiens2/Pose"

    def run(
        self,
        pose_model: Sapiens2PoseModel,
        image: torch.Tensor,
        keypoint_threshold: float = 0.3,
        bbox_threshold: float = 0.3,
        nms_threshold: float = 0.3,
        radius: int = 4,
        thickness: int = 2,
        fallback_full_image_bbox: bool = True,
        flip_test: bool = True,
        show_points: bool = True,
        show_skeleton: bool = True,
        bboxes: dict[str, Any] | None = None,
    ):
        resolved_bboxes = _resolve_pose_bboxes(
            image_batch=image,
            pose_model=pose_model,
            bboxes=bboxes,
            bbox_threshold=bbox_threshold,
            nms_threshold=nms_threshold,
            fallback_full_image_bbox=fallback_full_image_bbox,
        )
        rendered = []
        masks = []
        frames = []
        instances = []
        for batch_index in range(image.shape[0]):
            image_rgb = _to_uint8_rgb(image[batch_index])
            frame_boxes = resolved_bboxes["boxes"][batch_index]
            keypoints, scores, pose_bboxes, keypoints_tensor, scores_tensor = _run_pose_one(
                image_rgb,
                pose_model,
                frame_boxes,
                flip_test,
            )
            rendered_rgb = _render_pose(
                image_rgb,
                keypoints,
                scores,
                pose_model.metainfo,
                keypoint_threshold,
                radius,
                thickness,
                show_points,
                show_skeleton,
            )
            rendered.append(torch.from_numpy(rendered_rgb.astype(np.float32) / 255.0))
            masks.append(_pose_mask(image_rgb.shape[:2], keypoints, scores, keypoint_threshold, radius, pose_model.metainfo))
            frames.append(
                {
                    "image_size": [int(image_rgb.shape[0]), int(image_rgb.shape[1])],
                    "keypoints": [np.asarray(item, dtype=float).tolist() for item in keypoints],
                    "keypoint_scores": [np.asarray(item, dtype=float).reshape(-1).tolist() for item in scores],
                    "bboxes": [np.asarray(item, dtype=float).reshape(-1)[:4].tolist() for item in pose_bboxes],
                }
            )
            instances.append(
                {
                    "bboxes": frame_boxes,
                    "keypoints": keypoints_tensor,
                    "keypoint_scores": scores_tensor,
                }
            )
        keypoint_names = _keypoint_names_from_metainfo(pose_model.metainfo)
        raw = {
            "task": "pose",
            "arch": pose_model.arch,
            "checkpoint": pose_model.checkpoint_path,
            "detector": pose_model.detector_path,
            "frames": frames,
            "instances": instances,
            "metainfo": pose_model.metainfo,
            "num_keypoints": len(keypoint_names),
            "keypoint_names": keypoint_names,
            "skeleton_links": [list(link) for link in pose_model.metainfo.get("skeleton_links", [])],
            "bbox_format": "xyxy",
            "source": resolved_bboxes["source"],
            "keypoint_threshold": float(keypoint_threshold),
        }
        return (torch.stack(rendered, 0), torch.stack(masks, 0), raw)


class Sapiens2PoseGroupMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",) * len(POSE_GROUPS)
    RETURN_NAMES = POSE_GROUPS
    FUNCTION = "split"
    CATEGORY = "Sapiens2/Pose"

    def split(self, model, image, radius: int = 4, threshold: float = 0.3):
        raw = _pose_result(model, image, radius, threshold)
        return tuple(_raw_group_mask(raw, (group,), radius, threshold) for group in POSE_GROUPS)


class Sapiens2PoseSelectGroup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "group": (POSE_GROUPS,),
            },
            "optional": {
                "radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select"
    CATEGORY = "Sapiens2/Pose"

    def select(self, model, image, group: str, radius: int = 4, threshold: float = 0.3):
        raw = _pose_result(model, image, radius, threshold)
        return (_raw_group_mask(raw, (group,), radius, threshold),)


class Sapiens2PoseCombineGroups:
    @classmethod
    def INPUT_TYPES(cls):
        toggles = {group: ("BOOLEAN", {"default": True}) for group in POSE_GROUPS}
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                **toggles,
            },
            "optional": {
                "radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"
    CATEGORY = "Sapiens2/Pose"

    def combine(self, model, image, radius: int = 4, threshold: float = 0.3, **toggles):
        raw = _pose_result(model, image, radius, threshold)
        groups = tuple(group for group in POSE_GROUPS if bool(toggles.get(group, False)))
        return (_raw_group_mask(raw, groups, radius, threshold),)


class Sapiens2SavePoseJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "sapiens2/pose", "multiline": False}),
                "pretty_json": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    FUNCTION = "save"
    CATEGORY = "Sapiens2/Pose"
    OUTPUT_NODE = True

    def save(self, model, image, filename_prefix: str = "sapiens2/pose", pretty_json: bool = True):
        raw = _pose_result(model, image)
        return ("\n".join(_save_pose_json(raw, filename_prefix, pretty_json)),)


def _raw_group_mask(raw: dict[str, Any], groups: tuple[str, ...], radius: int, threshold: float) -> torch.Tensor:
    frames = raw["frames"]
    metainfo = raw["metainfo"]
    masks = []
    for frame in frames:
        image_size = frame.get("image_size") or [1, 1]
        height, width = int(image_size[0]), int(image_size[1])
        masks.append(
            _pose_mask(
                (height, width),
                frame.get("keypoints", []),
                frame.get("keypoint_scores", []),
                threshold,
                radius,
                metainfo,
                groups,
            )
        )
    return torch.stack(masks, 0)


def _output_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except (ImportError, AttributeError):
        return Path.cwd() / "output"


def _safe_prefix(prefix: str, default: str = "sapiens2/pose") -> Path:
    cleaned = prefix.strip().replace("\\", "/").strip("/")
    if not cleaned:
        cleaned = default
    parts = [part for part in cleaned.split("/") if part not in ("", ".", "..")]
    return Path(*parts) if parts else Path(default)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}_{index:05d}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available output filename near {path}")


def _save_pose_json(raw: dict[str, Any], filename_prefix: str, pretty_json: bool) -> list[str]:
    frames = raw.get("frames")
    if not isinstance(frames, list):
        raise ValueError("SAPIENS2_POSE_RESULT is missing frames.")

    output_base = _output_root() / _safe_prefix(filename_prefix)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    keypoint_names = [str(name) for name in raw.get("keypoint_names", [])]
    skeleton_links = [list(link) for link in raw.get("skeleton_links", [])]
    saved = []

    for image_index, frame in enumerate(frames):
        payload = {
            "task": "pose",
            "image_index": image_index,
            "source": str(raw.get("source", "unknown")),
            "arch": str(raw.get("arch", "")),
            "checkpoint": str(raw.get("checkpoint", "")),
            "detector": str(raw.get("detector", "")),
            "num_keypoints": int(raw.get("num_keypoints", len(keypoint_names))),
            "keypoint_names": keypoint_names,
            "skeleton_links": skeleton_links,
            "bbox_format": "xyxy",
            "image_size": frame.get("image_size", []),
            "instances": [],
        }
        bboxes = frame.get("bboxes", [])
        keypoints = frame.get("keypoints", [])
        scores = frame.get("keypoint_scores", [])
        if not (len(bboxes) == len(keypoints) == len(scores)):
            raise ValueError("Pose export expected matching counts for bboxes, keypoints, and scores.")
        for person_index, (bbox, kpts, kpt_scores) in enumerate(zip(bboxes, keypoints, scores)):
            payload["instances"].append(
                {
                    "person_index": person_index,
                    "bbox": bbox,
                    "keypoints": kpts,
                    "keypoint_scores": kpt_scores,
                }
            )

        suffix = f"_{image_index:05d}.json" if len(frames) > 1 else ".json"
        output_path = _unique_path(output_base.with_name(output_base.name + suffix))
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2 if pretty_json else None, ensure_ascii=True)
            handle.write("\n")
        saved.append(str(output_path))
    return saved
