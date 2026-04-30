import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import DEVICES, MODEL_SIZE_CHOICES, POSE_DETECTOR_REPO, POSE_RTMDET_FILENAME, SEG_CLASS_COUNT, SEG_PARTS
from .folders import get_model_root
from .huggingface import _resolve_hf_args, _size_to_arch, download_sapiens2_from_hf, download_sapiens2_pose_detector_from_hf
from .inference import Sapiens2DenseInference
from .model_loading import load_sapiens2_model
from .pose import Sapiens2PoseInference, load_sapiens2_pose_model
from .types import Sapiens2PoseModel


TASKS = ("segmentation", "normal", "pointmap", "pose")
MANUAL_MODEL_SIZE_CHOICES = ("auto",) + MODEL_SIZE_CHOICES
PREVIEW_MODES = ("result", "overlay", "side_by_side", "source")
SEG_GROUPS = {
    "Background": {"all": (0,)},
    "Apparel": {"all": (1,)},
    "Eyeglass": {"all": (2,)},
    "Face Neck": {"all": (3,)},
    "Face": {
        "skin": (3,),
        "with eyeglass": (2, 3),
        "with mouth": (3, 24, 25, 26, 27, 28),
        "all": (2, 3, 24, 25, 26, 27, 28),
        "full": (2, 3, 24, 25, 26, 27, 28),
    },
    "Hair": {"all": (4,)},
    "Foot": {"all": (5, 14), "left": (5,), "right": (14,)},
    "Hand": {"all": (6, 15), "left": (6,), "right": (15,)},
    "Arm": {
        "all": (7, 11, 16, 20),
        "left": (7, 11),
        "right": (16, 20),
        "upper": (11, 20),
        "lower": (7, 16),
        "left upper": (11,),
        "left lower": (7,),
        "right upper": (20,),
        "right lower": (16,),
    },
    "Leg": {
        "all": (8, 12, 17, 21),
        "left": (8, 12),
        "right": (17, 21),
        "upper": (12, 21),
        "lower": (8, 17),
        "left upper": (12,),
        "left lower": (8,),
        "right upper": (21,),
        "right lower": (17,),
    },
    "Shoe": {"all": (9, 18), "left": (9,), "right": (18,)},
    "Sock": {"all": (10, 19), "left": (10,), "right": (19,)},
    "Clothing": {"all": (13, 23), "upper": (23,), "lower": (13,)},
    "Torso": {"all": (22,)},
    "Lip": {"all": (24, 25), "upper": (25,), "lower": (24,)},
    "Teeth": {"all": (26, 27), "upper": (27,), "lower": (26,)},
    "Tongue": {"all": (28,)},
    "Mouth": {"all": (24, 25, 26, 27, 28), "lip": (24, 25), "teeth": (26, 27), "tongue": (28,)},
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


def _local_detector() -> str:
    rtmdet = get_model_root() / "detector" / POSE_RTMDET_FILENAME
    if rtmdet.is_file():
        return str(rtmdet)
    local_dir = get_model_root() / "detector" / POSE_DETECTOR_REPO.rsplit("/", 1)[-1]
    if local_dir.exists():
        return str(local_dir)
    raise FileNotFoundError(
        "detector_path is required for manual pose loading unless a default detector already exists locally."
    )


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
    legacy_limb = {
        "Lower Arm": ("Arm", "lower"),
        "Upper Arm": ("Arm", "upper"),
        "Lower Leg": ("Leg", "lower"),
        "Upper Leg": ("Leg", "upper"),
    }
    if name_text in legacy_limb:
        name_text, limb_detail = legacy_limb[name_text]
        if detail_text in ("left", "right"):
            detail_text = f"{detail_text} {limb_detail}"
        elif detail_text == "all":
            detail_text = limb_detail
    if name_text in SEG_GROUPS:
        options = SEG_GROUPS[name_text]
        return options.get(detail_text, next(iter(options.values())))
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
    ids = torch.as_tensor(part_ids, dtype=class_ids.dtype, device=class_ids.device)
    return _comfy_mask(class_ids.unsqueeze(1).eq(ids.view(1, -1, 1, 1)).float().flatten(0, 1))


def _merge_parts(class_ids: torch.Tensor, part_ids: list[int], invert: bool) -> torch.Tensor:
    if not part_ids:
        merged = torch.zeros_like(class_ids, dtype=torch.float32)
    else:
        ids = torch.as_tensor(part_ids, dtype=class_ids.dtype, device=class_ids.device)
        merged = torch.isin(class_ids, ids).float()
    if invert:
        merged = 1.0 - merged
    return _comfy_mask(merged)


def _mask_preview(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.55) -> torch.Tensor:
    image = _comfy_image(image)
    mask = _comfy_mask(mask)
    color = torch.tensor([0.0, 1.0, 1.0], dtype=image.dtype).view(1, 1, 1, 3)
    preview = torch.where(mask.unsqueeze(-1) > 0, image * (1.0 - alpha) + color * alpha, image)
    return _comfy_image(preview)


def _format_preview(source: torch.Tensor, result: torch.Tensor, mode: str, alpha: float = 0.5) -> torch.Tensor:
    source = _comfy_image(source)[..., :3]
    result = _comfy_image(result)
    if result.shape[-1] == 1:
        result = result.repeat(1, 1, 1, 3)
    else:
        result = result[..., :3]
    mode = str(mode or "result")
    if mode == "source":
        return source
    if mode == "overlay":
        return (source * (1.0 - alpha) + result * alpha).clamp(0, 1)
    if mode == "side_by_side":
        return torch.cat((source, result), dim=2)
    return result


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


def _ui_3d_entry(path: str | Path) -> dict[str, str]:
    path = Path(path)
    try:
        relative = path.relative_to(_output_root())
        subfolder = "" if relative.parent == Path(".") else relative.parent.as_posix()
        return {"filename": relative.name, "subfolder": subfolder, "type": "output", "mediaType": "3d"}
    except Exception:
        return {"filename": str(path), "subfolder": "", "type": "output", "mediaType": "3d"}


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


POSE_TARGETS = ("BODY_25", "308-keypoint", "COCO_18", "OpenPose hand 21 + 21", "OpenPose face 70")
_COCO18 = (0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3)
_BODY25 = (0, 69, 6, 8, 41, 5, 7, 62, (9, 10), 10, 12, 14, 9, 11, 13, 2, 1, 4, 3, 15, 16, 17, 18, 19, 20)
_RIGHT_HAND21 = (41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37)
_LEFT_HAND21 = (62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58)
_BODY25_EDGES = (
    (1, 8),
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 1),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
)
_COCO18_EDGES = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
_HAND21_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
_FACE70_EDGES = tuple(zip(range(17, 21), range(18, 22))) + tuple(zip(range(22, 26), range(23, 27))) + (
    (27, 28),
    (28, 29),
    (29, 30),
    (31, 32),
    (32, 33),
    (33, 34),
    (34, 35),
) + tuple(zip((36, 37, 38, 39, 40, 41), (37, 38, 39, 40, 41, 36))) + tuple(
    zip((42, 43, 44, 45, 46, 47), (43, 44, 45, 46, 47, 42))
) + tuple(zip(range(48, 59), range(49, 60))) + ((59, 48),) + tuple(zip(range(60, 67), range(61, 68))) + ((67, 60),)
_GOLIATH_FACE = dict(
    r_brow_up=[78, 80, 81, 83, 84],
    l_brow_up=[87, 89, 90, 92, 93],
    nose_bridge=[70, 71, 73, 74, 75, 178],
    nose_base=[180, 184, 179, 187, 181],
    r_eye_outer=121,
    r_eye_inner=120,
    r_eye_upper=[122, 123, 124, 125, 126, 127, 128],
    r_eye_lower=[163, 164, 165, 166, 167, 168, 169],
    l_eye_inner=96,
    l_eye_outer=97,
    l_eye_upper=[98, 99, 100, 101, 102, 103, 104],
    l_eye_lower=[146, 147, 148, 149, 150, 151, 152],
    lip_o_r_corner=188,
    lip_o_l_corner=189,
    cupid=190,
    lower_o_center=191,
    lip_o_upper=[192, 193, 196, 197, 198, 199],
    lip_o_lower=[194, 195, 200, 201, 202, 203],
    lip_i_r_corner=204,
    lip_i_l_corner=205,
    upper_i_center=206,
    lower_i_center=207,
    lip_i_upper=[208, 209, 212, 213, 214, 215],
    lip_i_lower=[210, 211, 216, 217, 218, 219],
    r_pupil=2,
    l_pupil=1,
)


def _pose_target_key(target: str) -> str:
    text = str(target or "").lower()
    if "308" in text:
        return "sapiens_308"
    if "coco" in text:
        return "coco_18"
    if "hand" in text:
        return "hand_21"
    if "face" in text:
        return "face_70"
    return "body_25"


def _triples(keypoints: Any, scores: Any) -> np.ndarray:
    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    conf = np.asarray(scores, dtype=np.float32).reshape(-1)
    count = min(points.shape[0], conf.shape[0])
    return np.concatenate([points[:count], conf[:count, None]], axis=1)


def _pick(triples: np.ndarray, spec: Any) -> np.ndarray:
    if isinstance(spec, tuple):
        picked = [_pick(triples, item) for item in spec]
        valid = [item for item in picked if item[2] > 0]
        if not valid:
            return np.zeros(3, dtype=np.float32)
        valid_arr = np.stack(valid)
        return np.array([valid_arr[:, 0].mean(), valid_arr[:, 1].mean(), valid_arr[:, 2].min()], dtype=np.float32)
    if spec is None or int(spec) < 0 or int(spec) >= triples.shape[0]:
        return np.zeros(3, dtype=np.float32)
    return triples[int(spec)].astype(np.float32)


def _subset(triples: np.ndarray, indices: tuple[Any, ...]) -> np.ndarray:
    return np.stack([_pick(triples, item) for item in indices]).astype(np.float32)


def _face70(triples: np.ndarray) -> np.ndarray:
    face: list[np.ndarray] = []
    mapping = _GOLIATH_FACE

    def push(indices: Any) -> None:
        for index in np.asarray(indices, dtype=np.int64).reshape(-1):
            face.append(_pick(triples, int(index)))

    def push_zeros(count: int) -> None:
        for _ in range(count):
            face.append(np.zeros(3, dtype=np.float32))

    def sort_x(indices: Any) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = values[(values >= 0) & (values < triples.shape[0])]
        return values[np.argsort(triples[values, 0])] if len(values) else values

    push_zeros(17)
    push(sort_x(mapping["r_brow_up"]))
    push(sort_x(mapping["l_brow_up"]))
    nose_bridge = sort_x(mapping["nose_bridge"])
    nose_bridge = nose_bridge[np.argsort(triples[nose_bridge, 1])] if len(nose_bridge) else nose_bridge
    push([nose_bridge[0], nose_bridge[len(nose_bridge) // 3], nose_bridge[2 * len(nose_bridge) // 3], nose_bridge[-1]])
    push(mapping["nose_base"])

    right_upper = sort_x(mapping["r_eye_upper"])
    right_lower = sort_x(mapping["r_eye_lower"])
    push(
        [
            mapping["r_eye_outer"],
            right_upper[len(right_upper) // 3],
            right_upper[2 * len(right_upper) // 3],
            mapping["r_eye_inner"],
            right_lower[2 * len(right_lower) // 3],
            right_lower[len(right_lower) // 3],
        ]
    )
    left_upper = sort_x(mapping["l_eye_upper"])
    left_lower = sort_x(mapping["l_eye_lower"])
    push(
        [
            mapping["l_eye_inner"],
            left_upper[len(left_upper) // 3],
            left_upper[2 * len(left_upper) // 3],
            mapping["l_eye_outer"],
            left_lower[2 * len(left_lower) // 3],
            left_lower[len(left_lower) // 3],
        ]
    )

    cupid_x = float(_pick(triples, mapping["cupid"])[0])
    upper_lip = sort_x(mapping["lip_o_upper"])
    left = upper_lip[triples[upper_lip, 0] < cupid_x]
    right = upper_lip[triples[upper_lip, 0] > cupid_x]
    push([mapping["lip_o_r_corner"]])
    push([int(left[0]), int(left[-1])] if len(left) >= 2 else list(left) + [mapping["lip_o_r_corner"]] * (2 - len(left)))
    push([mapping["cupid"]])
    push([int(right[0]), int(right[-1])] if len(right) >= 2 else list(right) + [mapping["lip_o_l_corner"]] * (2 - len(right)))
    push([mapping["lip_o_l_corner"]])

    lower_center_x = float(_pick(triples, mapping["lower_o_center"])[0])
    lower_lip = sort_x(mapping["lip_o_lower"])
    left_lower_lip = lower_lip[triples[lower_lip, 0] < lower_center_x]
    right_lower_lip = lower_lip[triples[lower_lip, 0] > lower_center_x]
    push(
        [int(right_lower_lip[-1]), int(right_lower_lip[0])]
        if len(right_lower_lip) >= 2
        else list(right_lower_lip) + [mapping["lower_o_center"]] * (2 - len(right_lower_lip))
    )
    push([mapping["lower_o_center"]])
    push(
        [int(left_lower_lip[-1]), int(left_lower_lip[0])]
        if len(left_lower_lip) >= 2
        else list(left_lower_lip) + [mapping["lower_o_center"]] * (2 - len(left_lower_lip))
    )

    upper_inner_x = float(_pick(triples, mapping["upper_i_center"])[0])
    upper_inner = sort_x(mapping["lip_i_upper"])
    left_inner = upper_inner[triples[upper_inner, 0] < upper_inner_x]
    right_inner = upper_inner[triples[upper_inner, 0] > upper_inner_x]
    lower_inner_x = float(_pick(triples, mapping["lower_i_center"])[0])
    lower_inner = sort_x(mapping["lip_i_lower"])
    bottom_left = lower_inner[triples[lower_inner, 0] < lower_inner_x]
    bottom_right = lower_inner[triples[lower_inner, 0] > lower_inner_x]
    push(
        [
            mapping["lip_i_r_corner"],
            int(left_inner[0]) if len(left_inner) else mapping["upper_i_center"],
            mapping["upper_i_center"],
            int(right_inner[-1]) if len(right_inner) else mapping["upper_i_center"],
            mapping["lip_i_l_corner"],
            int(bottom_right[-1]) if len(bottom_right) else mapping["lower_i_center"],
            mapping["lower_i_center"],
            int(bottom_left[0]) if len(bottom_left) else mapping["lower_i_center"],
        ]
    )
    push([mapping["r_pupil"], mapping["l_pupil"]])
    return np.stack(face).astype(np.float32)


def _target_triples(triples: np.ndarray, target: str) -> np.ndarray:
    key = _pose_target_key(target)
    if key == "coco_18":
        return _subset(triples, _COCO18)
    if key == "sapiens_308":
        return triples.astype(np.float32)
    if key == "hand_21":
        return np.concatenate([_subset(triples, _LEFT_HAND21), _subset(triples, _RIGHT_HAND21)], axis=0)
    if key == "face_70":
        return _face70(triples)
    return _subset(triples, _BODY25)


def _flat(values: np.ndarray) -> list[float]:
    return [float(item) for item in values.reshape(-1)]


def _draw_pose(
    canvas: np.ndarray,
    triples: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    threshold: float,
    radius: int = 3,
    thickness: int = 3,
    show_points: bool = True,
    show_skeleton: bool = True,
) -> None:
    import cv2

    colors = (
        (255, 0, 85),
        (255, 85, 0),
        (255, 170, 0),
        (170, 255, 0),
        (85, 255, 0),
        (0, 255, 85),
        (0, 255, 170),
        (0, 170, 255),
        (0, 85, 255),
        (85, 0, 255),
        (170, 0, 255),
        (255, 0, 170),
    )
    height, width = canvas.shape[:2]
    if show_skeleton:
        for index, (left, right) in enumerate(edges):
            if left >= len(triples) or right >= len(triples):
                continue
            a = triples[left]
            b = triples[right]
            if a[2] < threshold or b[2] < threshold:
                continue
            ax, ay = int(round(float(a[0]))), int(round(float(a[1])))
            bx, by = int(round(float(b[0]))), int(round(float(b[1])))
            if 0 <= ax < width and 0 <= ay < height and 0 <= bx < width and 0 <= by < height:
                cv2.line(canvas, (ax, ay), (bx, by), colors[index % len(colors)], max(1, int(thickness)), lineType=cv2.LINE_AA)
    if show_points:
        for index, point in enumerate(triples):
            if point[2] < threshold:
                continue
            x, y = int(round(float(point[0]))), int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), max(1, int(radius)), colors[index % len(colors)], -1, lineType=cv2.LINE_AA)


def _target_edges(raw: dict[str, Any], target: str) -> tuple[tuple[int, int], ...]:
    key = _pose_target_key(target)
    if key == "coco_18":
        return _COCO18_EDGES
    if key == "hand_21":
        return _HAND21_EDGES + tuple((a + 21, b + 21) for a, b in _HAND21_EDGES)
    if key == "face_70":
        return _FACE70_EDGES
    if key == "sapiens_308":
        return tuple(tuple(map(int, link[:2])) for link in raw.get("skeleton_links", []) if len(link) >= 2)
    return _BODY25_EDGES


def _pose_target_image(
    raw: dict[str, Any],
    image: torch.Tensor,
    target: str,
    overlay: bool = False,
    radius: int = 3,
    thickness: int = 3,
    show_points: bool = True,
    show_skeleton: bool = True,
) -> torch.Tensor:
    batch = _comfy_image(image)
    threshold = float(raw.get("keypoint_threshold", 0.3))
    rendered = []
    frames = raw.get("frames", [])
    for index, source in enumerate(batch):
        if overlay:
            canvas = source.numpy()
            canvas = np.repeat(canvas, 3, axis=-1) if canvas.shape[-1] == 1 else canvas[..., :3]
            canvas = (np.clip(canvas, 0, 1) * 255).round().astype(np.uint8)
        else:
            canvas = np.zeros(tuple(source.shape[:2]) + (3,), dtype=np.uint8)
        if index < len(frames):
            frame = frames[index]
            for keypoints, scores in zip(frame.get("keypoints", []), frame.get("keypoint_scores", [])):
                _draw_pose(
                    canvas,
                    _target_triples(_triples(keypoints, scores), target),
                    _target_edges(raw, target),
                    threshold,
                    radius=radius,
                    thickness=thickness,
                    show_points=show_points,
                    show_skeleton=show_skeleton,
                )
        rendered.append(torch.from_numpy(canvas).float() / 255.0)
    return _comfy_image(torch.stack(rendered, dim=0))


def _openpose_json(raw: dict[str, Any], target: str) -> str:
    target_key = _pose_target_key(target)
    frames = []
    for image_index, frame in enumerate(raw.get("frames", [])):
        people = []
        for bbox, keypoints, scores in zip(
            frame.get("bboxes", []),
            frame.get("keypoints", []),
            frame.get("keypoint_scores", []),
        ):
            source = _triples(keypoints, scores)
            person = {
                "person_id": [-1],
                "bbox": bbox,
                "target": target,
                "pose_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "face_keypoints_2d": [],
                "sapiens_keypoints_2d": _flat(source),
            }
            if target_key in ("body_25", "coco_18"):
                person["pose_keypoints_2d"] = _flat(_target_triples(source, target))
            elif target_key == "sapiens_308":
                person["pose_keypoints_2d"] = _flat(source)
            elif target_key == "hand_21":
                person["hand_left_keypoints_2d"] = _flat(_subset(source, _LEFT_HAND21))
                person["hand_right_keypoints_2d"] = _flat(_subset(source, _RIGHT_HAND21))
            elif target_key == "face_70":
                person["face_keypoints_2d"] = _flat(_face70(source))
            people.append(person)
        height, width = frame.get("image_size", [0, 0])
        frames.append(
            {
                "version": 1.3,
                "image_index": image_index,
                "canvas_width": int(width),
                "canvas_height": int(height),
                "target": target,
                "people": people,
            }
        )
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


class Sapiens2ModelLoaderManual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (TASKS,),
                "checkpoint_path": ("STRING", {"default": ""}),
                "model_size": (MANUAL_MODEL_SIZE_CHOICES, {"default": "auto"}),
                "device": (DEVICES, {"default": "auto"}),
            },
            "optional": {
                "detector_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("SAPIENS2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Sapiens2"

    def load(
        self,
        task: str,
        checkpoint_path: str,
        model_size: str = "auto",
        device: str = "auto",
        detector_path: str = "",
    ):
        checkpoint = str(checkpoint_path or "").strip()
        if not checkpoint:
            raise ValueError("checkpoint_path is required for Sapiens2 Manual Model Loader.")
        if task == "pose":
            detector = str(detector_path or "").strip()
            if not detector:
                detector = _local_detector()
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
                arch="auto" if model_size == "auto" else _size_to_arch(model_size),
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
        _, _, labels, raw = Sapiens2DenseInference().run(model, image, overlay_opacity=0.5)
        class_ids = raw["class_ids"].detach().cpu().long()
        selected = _selected_parts(parts)
        part_ids = selected if selected is not None else list(range(1, SEG_CLASS_COUNT))
        merged_mask = _merge_parts(class_ids, part_ids, invert)
        return (
            _mask_preview(image, merged_mask),
            merged_mask,
            _part_masks(class_ids, part_ids),
            {"class_ids": class_ids, "parts": SEG_PARTS, "groups": SEG_GROUPS},
        )


class Sapiens2Normal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preview_mode": (PREVIEW_MODES, {"default": "result"}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(self, model, image, preview_mode: str = "result", mask=None):
        _require_task(model, "normal")
        preview, _, _, _ = Sapiens2DenseInference().run(model, image, mask=mask)
        return (_format_preview(image, preview, preview_mode),)


class Sapiens2Pointmap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preview_mode": (PREVIEW_MODES, {"default": "result"}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "pointmap_glb")
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(self, model, image, preview_mode: str = "result", mask=None):
        _require_task(model, "pointmap")
        preview, _, _, raw = Sapiens2DenseInference().run(model, image, mask=mask)
        glb_path = _write_pointmap_glb(raw["pointmap"][0], image)
        return {"ui": {"3d": [_ui_3d_entry(glb_path)]}, "result": (_format_preview(image, preview, preview_mode), glb_path)}


class Sapiens2Pose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "target": (POSE_TARGETS, {"default": "BODY_25"}),
            },
            "optional": {"bboxes": ("SAPIENS2_BBOXES",)},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("openpose_image", "preview", "openpose_json")
    FUNCTION = "run"
    CATEGORY = "Sapiens2"

    def run(self, model, image, target: str = "BODY_25", bboxes=None):
        if not isinstance(model, Sapiens2PoseModel):
            _require_task(model, "pose")
        _, _, raw = Sapiens2PoseInference().run(pose_model=model, image=image, bboxes=bboxes, render_outputs=False)
        return (
            _pose_target_image(raw, image, target),
            _pose_target_image(raw, image, target, overlay=True),
            _openpose_json(raw, target),
        )
