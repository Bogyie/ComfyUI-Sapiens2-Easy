import json
import struct
from pathlib import Path
from typing import Any

import torch

from .constants import DEVICES, MODEL_SIZE_CHOICES, POSE_DETECTOR_REPO, POSE_RTMDET_FILENAME, SEG_CLASS_COUNT, SEG_PARTS
from .folders import get_model_root
from .huggingface import _resolve_hf_args, _size_to_arch, download_sapiens2_from_hf, download_sapiens2_pose_detector_from_hf
from .inference import Sapiens2DenseInference
from .model_loading import load_sapiens2_model
from .pose import Sapiens2PoseInference, load_sapiens2_pose_model
from .types import Sapiens2PoseModel


TASKS = ("segmentation", "normal", "pointmap", "pose")
SEG_GROUPS = {
    "Background": {"all": (0,)},
    "Apparel": {"all": (1,)},
    "Eyeglass": {"all": (2,)},
    "Face Neck": {"all": (3,)},
    "Hair": {"all": (4,)},
    "Foot": {"all": (5, 14), "left": (5,), "right": (14,)},
    "Hand": {"all": (6, 15), "left": (6,), "right": (15,)},
    "Arm": {"all": (7, 11, 16, 20), "left": (7, 11), "right": (16, 20)},
    "Lower Arm": {"all": (7, 16), "left": (7,), "right": (16,)},
    "Upper Arm": {"all": (11, 20), "left": (11,), "right": (20,)},
    "Leg": {"all": (8, 12, 17, 21), "left": (8, 12), "right": (17, 21)},
    "Lower Leg": {"all": (8, 17), "left": (8,), "right": (17,)},
    "Upper Leg": {"all": (12, 21), "left": (12,), "right": (21,)},
    "Shoe": {"all": (9, 18), "left": (9,), "right": (18,)},
    "Sock": {"all": (10, 19), "left": (10,), "right": (19,)},
    "Clothing": {"all": (13, 23), "upper": (23,), "lower": (13,)},
    "Torso": {"all": (22,)},
    "Lip": {"all": (24, 25), "upper": (25,), "lower": (24,)},
    "Teeth": {"all": (26, 27), "upper": (27,), "lower": (26,)},
    "Tongue": {"all": (28,)},
}


def _default_checkpoint_path(task: str, model_size: str) -> str:
    _, filename, local_dir = _resolve_hf_args(task, model_size, "", "", "")
    return str(Path(local_dir) / filename)


def _checkpoint(task: str, model_size: str) -> tuple[str, str, str, str]:
    local_path = _default_checkpoint_path(task, model_size)
    if Path(local_path).is_file():
        return local_path, "local_cache", "", ""
    path, repo, filename = download_sapiens2_from_hf(task=task, model_size=model_size)
    return path, "huggingface_download", repo, filename


def _detector() -> tuple[str, str]:
    rtmdet = get_model_root() / "detector" / POSE_RTMDET_FILENAME
    if rtmdet.is_file():
        return str(rtmdet), "local_rtmdet"
    local_dir = str(get_model_root() / "detector" / POSE_DETECTOR_REPO.rsplit("/", 1)[-1])
    if Path(local_dir).exists():
        return local_dir, "local_detector_cache"
    path, _ = download_sapiens2_pose_detector_from_hf(local_dir=local_dir)
    return path, "huggingface_detector_download"


def _require_task(model: Any, task: str) -> None:
    actual = getattr(model, "task", None)
    if actual != task:
        raise ValueError(f"This node needs a Sapiens2 {task} model, got {actual!r}.")


def _comfy_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected IMAGE shape [H,W,C] or [B,H,W,C], got {tuple(image.shape)}")
    return image


def _comfy_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().float().cpu().clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected MASK shape [H,W] or [B,H,W], got {tuple(mask.shape)}")
    return mask


def _part_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value < SEG_CLASS_COUNT else None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        head = text.split(":", 1)[0].strip()
        if head.isdigit():
            return _part_id(int(head))
    if text.isdigit():
        return _part_id(int(text))
    normalized = text.lower().replace(" ", "_")
    for index, name in enumerate(SEG_PARTS):
        if normalized == name.lower():
            return index
    return None


def _group_ids(name: Any, detail: Any = "all") -> tuple[int, ...]:
    name_text = str(name).strip()
    detail_text = str(detail or "all").strip().lower()
    if name_text in SEG_GROUPS:
        options = SEG_GROUPS[name_text]
        return options.get(detail_text, options["all"])
    part_index = _part_id(name_text)
    return (part_index,) if part_index is not None else ()


def _selected_parts(parts: str) -> list[int] | None:
    text = (parts or "").strip()
    if not text:
        return None
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [{"part": item.strip(), "enabled": True} for item in text.replace("\n", ",").split(",")]
    selected = set()
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            if not bool(row.get("enabled", True)):
                continue
            ids = _group_ids(row.get("name", row.get("part", "")), row.get("detail", "all"))
        else:
            ids = _group_ids(row)
        for part_index in ids:
            selected.add(part_index)
    return sorted(selected)


def _part_masks(class_ids: torch.Tensor, part_ids: list[int]) -> torch.Tensor:
    if not part_ids:
        return torch.zeros_like(class_ids, dtype=torch.float32)
    masks = [(class_ids == idx).float() for idx in part_ids]
    return _comfy_mask(torch.stack(masks, dim=1).flatten(0, 1))


def _merge_parts(class_ids: torch.Tensor, part_ids: list[int], invert: bool) -> torch.Tensor:
    if not part_ids:
        merged = torch.zeros_like(class_ids, dtype=torch.float32)
    else:
        merged = torch.zeros_like(class_ids, dtype=torch.float32)
        for idx in part_ids:
            merged = torch.maximum(merged, (class_ids == idx).float())
    if invert:
        merged = 1.0 - merged
    return _comfy_mask(merged)


def _output_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd() / "output"


def _unique_path(prefix: str, suffix: str) -> Path:
    base = _output_root() / "sapiens2"
    base.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        path = base / f"{prefix}_{index:05d}{suffix}"
        if not path.exists():
            return path
    raise FileExistsError(f"Could not create a unique {suffix} output path.")


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _write_pointmap_glb(pointmap: torch.Tensor, image: torch.Tensor, max_points: int = 60000) -> str:
    points = pointmap.detach().cpu().float()
    image = _comfy_image(image)[0, :, :, :3]
    valid = torch.isfinite(points[2]) & (points[2] > 0)
    count = int(valid.sum().item())
    path = _unique_path("pointmap", ".glb")
    if count == 0:
        path.write_bytes(b"")
        return str(path)

    stride = max(1, int((count / max_points) ** 0.5))
    sampled_points = points[:, ::stride, ::stride].permute(1, 2, 0).reshape(-1, 3)
    sampled_colors = image[::stride, ::stride].reshape(-1, 3)
    sampled_valid = valid[::stride, ::stride].reshape(-1)
    vertices = sampled_points[sampled_valid].contiguous()
    colors = (sampled_colors[sampled_valid].clamp(0, 1) * 255).round().to(torch.uint8).contiguous()

    vertex_bytes = _pad4(vertices.numpy().astype("float32").tobytes())
    color_bytes = _pad4(colors.numpy().tobytes())
    buffer = vertex_bytes + color_bytes
    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI-Sapiens2-Easy"},
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vertex_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(vertex_bytes), "byteLength": len(color_bytes), "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
                "min": vertices.min(0).values.tolist(),
                "max": vertices.max(0).values.tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "count": int(colors.shape[0]),
                "type": "VEC3",
                "normalized": True,
            },
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_size = 12 + 8 + len(json_chunk) + 8 + len(buffer)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_size))
        handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        handle.write(json_chunk)
        handle.write(struct.pack("<II", len(buffer), 0x004E4942))
        handle.write(buffer)
    return str(path)


def _openpose_json(raw: dict[str, Any]) -> str:
    frames = []
    for image_index, frame in enumerate(raw.get("frames", [])):
        people = []
        for bbox, keypoints, scores in zip(
            frame.get("bboxes", []),
            frame.get("keypoints", []),
            frame.get("keypoint_scores", []),
        ):
            flat = []
            for xy, score in zip(keypoints, scores):
                flat.extend([float(xy[0]), float(xy[1]), float(score)])
            people.append(
                {
                    "person_id": [-1],
                    "bbox": bbox,
                    "pose_keypoints_2d": flat,
                    "sapiens_keypoints_2d": flat,
                }
            )
        frames.append({"version": 1.3, "image_index": image_index, "people": people})
    return json.dumps(frames[0] if len(frames) == 1 else frames, ensure_ascii=True)


class Sapiens2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (TASKS,),
                "model_size": (MODEL_SIZE_CHOICES, {"default": "0.4b"}),
                "device": (DEVICES, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("SAPIENS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Sapiens2"

    def load(self, task: str, model_size: str, device: str = "auto"):
        checkpoint, _, _, _ = _checkpoint(task, model_size)
        if task == "pose":
            detector, _ = _detector()
            return (
                load_sapiens2_pose_model(
                    checkpoint_path=checkpoint,
                    detector_path=detector,
                    model_size=model_size,
                    device=device,
                    dtype="auto",
                ),
            )
        return (
            load_sapiens2_model(
                task=task,
                arch=_size_to_arch(model_size),
                device=device,
                dtype="auto",
                checkpoint_path=checkpoint,
            ),
        )


class Sapiens2Segmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
                "parts": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "SAPIENS2_LABELS")
    RETURN_NAMES = ("preview", "merged_mask", "masks", "labels")
    FUNCTION = "segment"
    CATEGORY = "Sapiens2"

    def segment(self, model, image, invert: bool = False, parts: str = ""):
        _require_task(model, "segmentation")
        preview, _, labels, raw = Sapiens2DenseInference().run(model, image, overlay_opacity=0.5)
        class_ids = raw["class_ids"].detach().cpu().long()
        selected = _selected_parts(parts)
        part_ids = selected if selected is not None else list(range(1, SEG_CLASS_COUNT))
        return (
            _comfy_image(preview),
            _merge_parts(class_ids, part_ids, invert),
            _part_masks(class_ids, part_ids),
            {"class_ids": class_ids, "parts": SEG_PARTS, "groups": SEG_GROUPS},
        )


class Sapiens2Normal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)},
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(self, model, image, mask=None):
        _require_task(model, "normal")
        preview, _, _, _ = Sapiens2DenseInference().run(model, image, mask=mask)
        return (_comfy_image(preview),)


class Sapiens2Pointmap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)},
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "pointmap_glb")
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(self, model, image, mask=None):
        _require_task(model, "pointmap")
        preview, _, _, raw = Sapiens2DenseInference().run(model, image, mask=mask)
        glb_path = _write_pointmap_glb(raw["pointmap"][0], image)
        return (_comfy_image(preview), glb_path)


class Sapiens2Pose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("SAPIENS2_MODEL",), "image": ("IMAGE",)},
            "optional": {"bboxes": ("SAPIENS2_BBOXES",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "openpose_json")
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(self, model, image, bboxes=None):
        if not isinstance(model, Sapiens2PoseModel):
            _require_task(model, "pose")
        preview, _, raw = Sapiens2PoseInference().run(pose_model=model, image=image, bboxes=bboxes)
        return (_comfy_image(preview), _openpose_json(raw))
