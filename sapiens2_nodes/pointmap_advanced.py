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
from .progress import NodeProgress


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _save_textured_glb(
    vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    texture_rgb: np.ndarray,
    path: Path,
    normals: np.ndarray | None = None,
    unlit_material: bool = True,
) -> None:
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    uvs = np.ascontiguousarray(uvs, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    normals = None if normals is None else np.ascontiguousarray(normals, dtype=np.float32)

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
    normal_view = add_buffer_view(normals.tobytes(), 34962) if normals is not None else None
    face_view = add_buffer_view(faces.tobytes(), 34963)
    image_view = add_buffer_view(_png_bytes(texture_rgb))

    accessors = [
        {
            "bufferView": vertex_view,
            "componentType": 5126,
            "count": int(vertices.shape[0]),
            "type": "VEC3",
            "min": vertices.min(axis=0).tolist(),
            "max": vertices.max(axis=0).tolist(),
        },
        {"bufferView": uv_view, "componentType": 5126, "count": int(uvs.shape[0]), "type": "VEC2"},
    ]
    attributes = {"POSITION": 0, "TEXCOORD_0": 1}
    if normal_view is not None:
        attributes["NORMAL"] = len(accessors)
        accessors.append({"bufferView": normal_view, "componentType": 5126, "count": int(normals.shape[0]), "type": "VEC3"})
    face_accessor = len(accessors)
    accessors.append({"bufferView": face_view, "componentType": 5125, "count": int(faces.size), "type": "SCALAR"})

    material = {
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": True,
    }
    extensions_used = []
    if unlit_material:
        material["extensions"] = {"KHR_materials_unlit": {}}
        extensions_used.append("KHR_materials_unlit")

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI-Sapiens2-Easy"},
        "buffers": [],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [material],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": face_accessor,
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
    if extensions_used:
        gltf["extensionsUsed"] = extensions_used

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


def _smooth_pointmap_surface(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    iterations: int,
    strength: float,
    edge_threshold: float,
) -> torch.Tensor:
    iterations = max(0, int(iterations))
    strength = float(max(0.0, min(1.0, strength)))
    if iterations == 0 or strength <= 0.0 or not bool(valid.any().item()):
        return xyz

    xyz_chw = xyz.movedim(-1, 0).clone()
    valid_mask = valid.to(dtype=xyz_chw.dtype).unsqueeze(0)
    spatial = torch.tensor(
        [[0.5, 0.75, 0.5], [0.75, 1.0, 0.75], [0.5, 0.75, 0.5]],
        dtype=xyz_chw.dtype,
        device=xyz_chw.device,
    ).reshape(1, 1, 9, 1, 1)
    range_sigma = max(0.03, min(abs(float(edge_threshold)) * 0.5, 0.5))
    height, width = valid.shape

    for _ in range(iterations):
        padded_xyz = F.pad(xyz_chw.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        patches = F.unfold(padded_xyz, kernel_size=3).view(1, 3, 9, height, width)
        padded_valid = F.pad(valid_mask.unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0.0)
        valid_patches = F.unfold(padded_valid, kernel_size=3).view(1, 1, 9, height, width)
        center_depth = xyz_chw[2].abs().clamp(min=1e-6).view(1, 1, 1, height, width)
        relative_depth_delta = (patches[:, 2:3] - xyz_chw[2].view(1, 1, 1, height, width)).abs() / center_depth
        range_weights = torch.exp(-0.5 * (relative_depth_delta / range_sigma) ** 2)
        weights = valid_patches * spatial * range_weights
        averaged = (patches * weights).sum(dim=2).squeeze(0) / weights.sum(dim=2).squeeze(0).clamp(min=1e-6)
        blended = xyz_chw * (1.0 - strength) + averaged * strength
        xyz_chw = torch.where(valid_mask > 0, blended, xyz_chw)

    return xyz_chw.movedim(0, -1).contiguous()


def _filter_triangles_by_quality(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
    max_edge_ratio: float,
    max_normal_angle: float,
) -> torch.Tensor:
    if triangles.numel() == 0:
        return triangles

    tri_vertices = vertices[triangles]
    edges = torch.stack(
        (
            tri_vertices[:, 1] - tri_vertices[:, 0],
            tri_vertices[:, 2] - tri_vertices[:, 1],
            tri_vertices[:, 0] - tri_vertices[:, 2],
        ),
        dim=1,
    )
    lengths = torch.linalg.norm(edges, dim=2)
    keep = lengths.min(dim=1).values > 1e-8
    ratio_limit = float(max_edge_ratio)
    if ratio_limit > 0:
        keep &= (lengths.max(dim=1).values / lengths.min(dim=1).values.clamp(min=1e-8)) <= ratio_limit

    angle_limit = float(max_normal_angle)
    if angle_limit > 0 and angle_limit < 180:
        normals = torch.cross(edges[:, 0], -edges[:, 2], dim=1)
        normals = F.normalize(normals, dim=1, eps=1e-8)
        mean_normal = F.normalize(normals[keep].mean(dim=0, keepdim=True), dim=1, eps=1e-8) if bool(keep.any().item()) else normals[:1]
        cos_limit = float(np.cos(np.deg2rad(angle_limit)))
        keep &= (normals @ mean_normal.squeeze(0)) >= cos_limit

    return triangles[keep]


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    normals = torch.zeros_like(vertices)
    if faces.numel() == 0:
        normals[:, 2] = 1.0
        return normals
    tri_vertices = vertices[faces]
    face_normals = torch.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0], dim=1)
    face_normals = F.normalize(face_normals, dim=1, eps=1e-8)
    normals.index_add_(0, faces.reshape(-1), face_normals.repeat_interleave(3, dim=0))
    normals = F.normalize(normals, dim=1, eps=1e-8)
    fallback = torch.tensor([0.0, 0.0, 1.0], dtype=normals.dtype).view(1, 3)
    return torch.where(torch.isfinite(normals).all(dim=1, keepdim=True), normals, fallback)


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
    mesh_smooth_iterations: int,
    mesh_smooth_strength: float,
    smooth_edge_threshold: float,
    max_edge_ratio: float,
    max_normal_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh_stride = max(1, int(mesh_stride))
    adjusted = _adjust_pointmap_geometry(pointmap, depth_scale=depth_scale, xy_scale=xy_scale, depth_bias=depth_bias)
    _, height, width = adjusted.shape
    xyz = adjusted.movedim(0, -1)[::mesh_stride, ::mesh_stride].contiguous()
    sampled_mask = mask[::mesh_stride, ::mesh_stride] if mask is not None else None
    valid = _valid_depth_mask(xyz, sampled_mask, rtol, min_depth, max_depth)
    xyz = _smooth_pointmap_surface(xyz, valid, mesh_smooth_iterations, mesh_smooth_strength, smooth_edge_threshold)
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
    triangles = _filter_triangles_by_quality(xyz.reshape(-1, 3), triangles, max_edge_ratio, max_normal_angle)
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
    normals = _vertex_normals(vertices, faces)

    texture = _comfy_image(image)[0, :, :, :3]
    texture = (texture.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return vertices.numpy(), uvs.numpy(), faces.numpy(), texture, normals.numpy()


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


def _coerce_pointmap_batch(pointmap) -> torch.Tensor:
    if isinstance(pointmap, dict):
        if "pointmap" not in pointmap:
            raise ValueError("Pointmap data is missing the 'pointmap' tensor.")
        pointmap = pointmap["pointmap"]
    if not isinstance(pointmap, torch.Tensor):
        raise TypeError("pointmap must be a pointmap data dict or torch.Tensor.")

    pointmaps = pointmap.detach().cpu().float()
    if pointmaps.ndim == 3:
        pointmaps = pointmaps.unsqueeze(0)
    if pointmaps.ndim != 4 or pointmaps.shape[1] != 3:
        raise ValueError(f"Expected pointmap shape [B, 3, H, W], got {tuple(pointmaps.shape)}.")
    return pointmaps


def _export_pointmap_models(
    pointmap,
    image,
    mask,
    render_mode: str,
    filename_prefix: str,
    mesh_stride: int,
    rtol: float,
    min_depth: float,
    max_depth: float,
    center_mesh: bool,
    flip_y: bool,
    flip_z: bool,
    depth_scale: float,
    xy_scale: float,
    depth_bias: float,
    max_points: int,
    splat_size: float,
    splat_max_points: int,
    mesh_smooth_iterations: int = 0,
    mesh_smooth_strength: float = 0.0,
    smooth_edge_threshold: float = 0.35,
    max_edge_ratio: float = 8.0,
    max_normal_angle: float = 0.0,
    unlit_material: bool = True,
) -> tuple[list[Path], list[dict[str, str]]]:
    pointmaps = _coerce_pointmap_batch(pointmap)
    images = _comfy_image(image)
    masks = _prepare_mask(mask, pointmaps.shape[0], pointmaps.shape[-2], pointmaps.shape[-1])

    paths: list[Path] = []
    ui_entries = []
    progress = NodeProgress(pointmaps.shape[0])
    for index in range(pointmaps.shape[0]):
        mask_i = masks[index] if masks is not None else None
        image_i = images[min(index, images.shape[0] - 1)].unsqueeze(0)
        if render_mode in {"points", "splats"}:
            path = Path(
                _write_pointmap_glb(
                    pointmaps[index],
                    image_i,
                    mask=mask,
                    mask_index=index,
                    render_as_splats=render_mode == "splats",
                    splat_size=splat_size,
                    depth_scale=depth_scale,
                    xy_scale=xy_scale,
                    depth_bias=depth_bias,
                    max_points=splat_max_points if render_mode == "splats" else max(1000, int(max_points)),
                    min_depth=min_depth,
                    max_depth=max_depth,
                    rtol=0.0,
                    filename_prefix=filename_prefix,
                )
            )
            paths.append(path)
            ui_entries.append(_ui_3d_entry(path))
            progress.update()
            continue

        vertices, uvs, faces, texture, normals = _mesh_from_pointmap(
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
            mesh_smooth_iterations,
            mesh_smooth_strength,
            smooth_edge_threshold,
            max_edge_ratio,
            max_normal_angle,
        )
        path = _output_path(filename_prefix, index)
        _save_textured_glb(vertices, uvs, faces, texture, path, normals=normals, unlit_material=unlit_material)
        paths.append(path)
        ui_entries.append(_ui_3d_entry(path))
        progress.update()

    return paths, ui_entries


class Sapiens2PointmapMeshAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAPIENS2_MODEL",),
                "image": ("IMAGE",),
                "preview_mode": (("result", "overlay", "side_by_side", "source"), {"default": "result"}),
                "render_mode": (("points", "splats", "mesh"), {"default": "mesh"}),
                "filename_prefix": ("STRING", {"default": "sapiens2_pointmap_mesh"}),
                "mesh_stride": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "rtol": ("FLOAT", {"default": 0.5, "min": 0.001, "max": 1.0, "step": 0.001}),
                "min_depth": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "max_depth": ("FLOAT", {"default": 25.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "center_mesh": ("BOOLEAN", {"default": True}),
                "flip_y": ("BOOLEAN", {"default": True}),
                "flip_z": ("BOOLEAN", {"default": True}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "xy_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "depth_bias": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "max_points": ("INT", {"default": 60000, "min": 1000, "step": 1000}),
                "splat_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "splat_max_points": ("INT", {"default": 30000, "min": 1000, "max": 100000, "step": 1000}),
                "mesh_smooth_iterations": ("INT", {"default": 4, "min": 0, "max": 16, "step": 1}),
                "mesh_smooth_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smooth_edge_threshold": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_edge_ratio": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "max_normal_angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0}),
                "unlit_material": ("BOOLEAN", {"default": True}),
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
        rtol: float = 0.5,
        min_depth: float = 0.05,
        max_depth: float = 25.0,
        center_mesh: bool = True,
        flip_y: bool = True,
        flip_z: bool = True,
        depth_scale: float = 1.0,
        xy_scale: float = 1.0,
        depth_bias: float = 0.0,
        max_points: int = 60000,
        splat_size: float = 0.0,
        splat_max_points: int = 30000,
        mesh_smooth_iterations: int = 4,
        mesh_smooth_strength: float = 0.35,
        smooth_edge_threshold: float = 0.35,
        max_edge_ratio: float = 8.0,
        max_normal_angle: float = 0.0,
        unlit_material: bool = True,
        mask=None,
    ):
        _require_task(model, "pointmap")
        preview, _, _, raw = Sapiens2DenseInference().run(model, image, mask=mask)
        paths, ui_entries = _export_pointmap_models(
            raw,
            image,
            mask,
            render_mode,
            filename_prefix,
            mesh_stride,
            rtol,
            min_depth,
            max_depth,
            center_mesh,
            flip_y,
            flip_z,
            depth_scale,
            xy_scale,
            depth_bias,
            max_points,
            splat_size,
            splat_max_points,
            mesh_smooth_iterations,
            mesh_smooth_strength,
            smooth_edge_threshold,
            max_edge_ratio,
            max_normal_angle,
            unlit_material,
        )

        first_path = str(paths[0]) if paths else ""
        result = (_format_preview(image, preview, preview_mode), "\n".join(str(path) for path in paths), first_path)
        return {"ui": {"3d": ui_entries}, "result": result}
