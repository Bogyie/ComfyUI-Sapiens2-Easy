import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .easy import (
    _comfy_image,
    _comfy_mask,
    _format_preview,
    _adjust_pointmap_geometry,
    _orient_pointmap_vertices,
    _png_bytes,
    _pointmap_center_offset,
    _output_root,
    _require_task,
    _ui_3d_entry,
    _write_pointmap_glb,
)
from .inference import Sapiens2DenseInference


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _save_textured_glb(
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_rgb: np.ndarray,
    path: Path,
) -> None:
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)

    buffer_parts: list[bytes] = []
    buffer_views = []

    def add_buffer_view(data: bytes, target: int | None = None) -> int:
        offset = sum(len(_pad4(part)) for part in buffer_parts)
        index = len(buffer_views)
        buffer_parts.append(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return index

    vertex_view = add_buffer_view(vertices.tobytes(), 34962)
    uv_view = add_buffer_view(uvs.tobytes(), 34962)
    face_view = add_buffer_view(faces.tobytes(), 34963)
    image_view = add_buffer_view(_png_bytes(texture_rgb))

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI-Sapiens2-Easy"},
        "buffers": [],
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": vertex_view,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            },
            {"bufferView": uv_view, "componentType": 5126, "count": int(uvs.shape[0]), "type": "VEC2"},
            {"bufferView": face_view, "componentType": 5125, "count": int(faces.size), "type": "SCALAR"},
        ],
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    buffer = b"".join(_pad4(part) for part in buffer_parts)
    gltf["buffers"] = [{"byteLength": len(buffer)}]
    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_size = 12 + 8 + len(json_bytes) + 8 + len(buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_size))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(buffer), 0x004E4942))
        handle.write(buffer)


def _prepare_mask(mask: torch.Tensor | None, batch_size: int, height: int, width: int) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = _comfy_mask(mask)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.repeat(batch_size, 1, 1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"Mask batch size ({mask.shape[0]}) does not match image batch size ({batch_size}).")
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
    return mask > 0.5


def _valid_depth_mask(
    xyz: torch.Tensor,
    mask: torch.Tensor | None,
    rtol: float,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    depth = xyz[..., 2]
    valid = torch.isfinite(xyz).all(dim=-1) & torch.isfinite(depth)
    valid &= (depth > float(min_depth)) & (depth < float(max_depth))
    if mask is not None:
        valid &= mask
    depth_4d = depth.unsqueeze(0).unsqueeze(0)
    depth_max = F.max_pool2d(depth_4d, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
    depth_min = -F.max_pool2d(-depth_4d, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
    valid &= ((depth_max - depth_min) / depth.abs().clamp(min=1e-6)) <= float(rtol)
    return valid


def _mesh_from_pointmap(
    pointmap: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor | None,
    rtol: float,
    min_depth: float,
    max_depth: float,
    mesh_stride: int,
    center_mesh: bool,
    flip_y: bool,
    flip_z: bool,
    depth_scale: float,
    xy_scale: float,
    depth_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh_stride = max(1, int(mesh_stride))
    adjusted = _adjust_pointmap_geometry(pointmap, depth_scale=depth_scale, xy_scale=xy_scale, depth_bias=depth_bias)
    _, height, width = adjusted.shape
    xyz = adjusted.movedim(0, -1)[::mesh_stride, ::mesh_stride].contiguous()
    sampled_mask = mask[::mesh_stride, ::mesh_stride] if mask is not None else None
    valid = _valid_depth_mask(xyz, sampled_mask, rtol, min_depth, max_depth)
    mesh_height, mesh_width = valid.shape

    rows = torch.arange(mesh_height - 1).view(-1, 1).expand(-1, mesh_width - 1)
    cols = torch.arange(mesh_width - 1).view(1, -1).expand(mesh_height - 1, -1)
    top_left = (rows * mesh_width + cols).reshape(-1)
    triangles = torch.cat(
        [
            torch.stack((top_left, top_left + mesh_width, top_left + 1), dim=1),
            torch.stack((top_left + 1, top_left + mesh_width, top_left + mesh_width + 1), dim=1),
        ],
        dim=0,
    ).long()
    flat_valid = valid.reshape(-1)
    triangles = triangles[flat_valid[triangles].all(dim=1)]
    if triangles.numel() == 0:
        raise RuntimeError("No pointmap mesh faces survived filtering. Relax rtol/min_depth/max_depth or provide a better mask.")

    used = torch.unique(triangles.reshape(-1))
    raw_vertices = xyz.reshape(-1, 3)[used]
    center_offset = _pointmap_center_offset(raw_vertices, flip_y=flip_y, flip_z=flip_z) if center_mesh else None
    vertices = _orient_pointmap_vertices(
        raw_vertices,
        center=center_mesh,
        flip_y=flip_y,
        flip_z=flip_z,
        center_offset=center_offset,
    )

    y_coords = torch.arange(0, height, mesh_stride, dtype=torch.float32)[:mesh_height] / max(height - 1, 1)
    x_coords = torch.arange(0, width, mesh_stride, dtype=torch.float32)[:mesh_width] / max(width - 1, 1)
    u_grid, v_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uvs = torch.stack((u_grid, v_grid), dim=-1).reshape(-1, 2)[used]
    faces = torch.searchsorted(used, triangles)

    texture = _comfy_image(image)[0, :, :, :3]
    texture = (texture.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return vertices.numpy(), uvs.numpy(), faces.numpy(), texture


def _output_path(filename_prefix: str, index: int) -> Path:
    safe_prefix = str(filename_prefix or "sapiens2_pointmap_mesh").strip().replace("\\", "/").strip("/")
    root = _output_root() / "sapiens2"
    root.mkdir(parents=True, exist_ok=True)
    for counter in range(10000):
        suffix = f"_{index:02d}" if index else ""
        path = root / f"{safe_prefix}{suffix}_{counter:05d}.glb"
        if not path.exists():
            return path
    raise FileExistsError("Could not create a unique pointmap mesh GLB path.")


class Sapiens2PointmapMeshAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preview_mode": (("result", "overlay", "side_by_side", "source"), {"default": "result"}),
                "render_mode": (("mesh", "splats"), {"default": "mesh"}),
                "filename_prefix": ("STRING", {"default": "sapiens2_pointmap_mesh"}),
                "mesh_stride": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "rtol": ("FLOAT", {"default": 0.04, "min": 0.001, "max": 1.0, "step": 0.001}),
                "min_depth": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "max_depth": ("FLOAT", {"default": 25.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "center_mesh": ("BOOLEAN", {"default": True}),
                "flip_y": ("BOOLEAN", {"default": True}),
                "flip_z": ("BOOLEAN", {"default": True}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "xy_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "depth_bias": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "splat_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "splat_max_points": ("INT", {"default": 30000, "min": 1000, "max": 100000, "step": 1000}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("preview", "glb_paths", "model_3d")
    FUNCTION = "run"
    CATEGORY = "Sapiens2/Advanced"

    def run(
        self,
        model,
        image,
        preview_mode: str = "result",
        render_mode: str = "mesh",
        filename_prefix: str = "sapiens2_pointmap_mesh",
        mesh_stride: int = 2,
        rtol: float = 0.04,
        min_depth: float = 0.05,
        max_depth: float = 25.0,
        center_mesh: bool = True,
        flip_y: bool = True,
        flip_z: bool = True,
        depth_scale: float = 1.0,
        xy_scale: float = 1.0,
        depth_bias: float = 0.0,
        splat_size: float = 0.0,
        splat_max_points: int = 30000,
        mask=None,
    ):
        _require_task(model, "pointmap")
        preview, _, _, raw = Sapiens2DenseInference().run(model, image, mask=mask)
        pointmaps = raw["pointmap"].detach().cpu().float()
        images = _comfy_image(image)
        masks = _prepare_mask(mask, pointmaps.shape[0], pointmaps.shape[-2], pointmaps.shape[-1])

        paths: list[Path] = []
        ui_entries = []
        for index in range(pointmaps.shape[0]):
            mask_i = masks[index] if masks is not None else None
            image_i = images[min(index, images.shape[0] - 1)].unsqueeze(0)
            if render_mode == "splats":
                path = Path(
                    _write_pointmap_glb(
                        pointmaps[index],
                        image_i,
                        mask=mask,
                        mask_index=index,
                        render_as_splats=True,
                        splat_size=splat_size,
                        depth_scale=depth_scale,
                        xy_scale=xy_scale,
                        depth_bias=depth_bias,
                        max_points=splat_max_points,
                        filename_prefix=filename_prefix,
                    )
                )
                paths.append(path)
                ui_entries.append(_ui_3d_entry(path))
                continue

            vertices, uvs, faces, texture = _mesh_from_pointmap(
                pointmaps[index],
                image_i,
                mask_i,
                rtol,
                min_depth,
                max_depth,
                mesh_stride,
                center_mesh,
                flip_y,
                flip_z,
                depth_scale,
                xy_scale,
                depth_bias,
            )
            path = _output_path(filename_prefix, index)
            _save_textured_glb(vertices, uvs, faces, texture, path)
            paths.append(path)
            ui_entries.append(_ui_3d_entry(path))

        first_path = str(paths[0]) if paths else ""
        result = (_format_preview(image, preview, preview_mode), "\n".join(str(path) for path in paths), first_path)
        return {"ui": {"3d": ui_entries}, "result": result}
