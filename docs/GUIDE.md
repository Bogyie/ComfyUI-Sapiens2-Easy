# ComfyUI-Sapiens2-Easy Guide

[English](GUIDE.md) | [Korean](GUIDE.ko.md) | [Back to README](../README.md)

This guide keeps setup, model behavior, node details, and troubleshooting out of the main README.

## Install

Clone this repository into `ComfyUI/custom_nodes/` and run the installer with the same Python environment that runs ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Bogyie/ComfyUI-Sapiens2-Easy.git
cd ComfyUI-Sapiens2-Easy
python install.py
```

Restart ComfyUI after installation. Nodes appear under:

```text
Sapiens2
Sapiens2/Advanced
```

The installer:

- installs this node's Python dependencies
- clones the official Sapiens2 source into `vendor/sapiens2`
- checks that important runtime packages still import correctly
- protects the existing ComfyUI PyTorch/CUDA/xformers stack with temporary pip constraints

Useful install modes:

```bash
# Install Python dependencies and clone Sapiens2 source
python install.py

# Install only this node's direct requirements
python install.py --no-deps

# Clone Sapiens2 source without installing dependencies
python install.py --skip-deps

# Use your own Sapiens2 checkout
python install.py --skip-clone
export SAPIENS2_REPO=/path/to/facebookresearch/sapiens2
```

Do **not** blindly install the upstream Sapiens2 `requirements.txt` into an existing ComfyUI environment unless you are intentionally managing PyTorch/CUDA versions yourself.

## Models

Supported model sizes:

```text
0.4b, 0.8b, 1b, 5b
```

Supported devices:

```text
auto, cuda, mps, cpu
```

`auto` prefers CUDA when available, otherwise CPU. It does not automatically choose MPS because some PyTorch/MPS combinations can produce incorrect Sapiens2 segmentation label maps. You can still choose `mps` manually.

Downloaded files are saved under:

```text
ComfyUI/models/sapiens2/<task>/
```

The loader maps tasks to the official Hugging Face repositories:

```text
segmentation -> facebook/sapiens2-seg-{0.4b,0.8b,1b,5b}
normal       -> facebook/sapiens2-normal-{0.4b,0.8b,1b,5b}
pointmap     -> facebook/sapiens2-pointmap-{0.4b,0.8b,1b,5b}
pose         -> facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}
```

## Recommended First Workflow

1. Add **Image Load**.
2. Add **Sapiens2 Model Loader** with `task = segmentation`.
3. Add **Sapiens2 Segmentation** and create a foreground/person mask.
4. Add another **Sapiens2 Model Loader** with `task = normal`.
5. Add **Sapiens2 Normal** and feed it the image plus optional mask.
6. Add another **Sapiens2 Model Loader** with `task = pointmap`.
7. Add **Sapiens2 Pointmap**, connect image, mask, and normal map.
8. Preview or export the generated `.glb` as `points`, `splats`, or `mesh`.

Good starting settings:

```text
model_size: 1b
device: cuda
preview_mode: result
render_mode: points or mesh
quality: high
mesh_smoothing: balanced
normal_detail: balanced
```

Use `0.4b` or `0.8b` for faster iteration, and move up to `1b` or `5b` when quality matters more than speed.

## Nodes

### Easy Nodes

| Node | Use It For |
| --- | --- |
| **Sapiens2 Model Loader** | Download and load a task-specific Sapiens2 model |
| **Sapiens2 Manual Model Loader** | Load a local checkpoint manually |
| **Sapiens2 Segmentation** | Create body-part masks and selected-area previews |
| **Sapiens2 Normal** | Generate normal maps with optional masking |
| **Sapiens2 Pointmap** | Export pointmap results as points, splats, or mesh `.glb` files |
| **Sapiens2 Pose** | Generate Sapiens2/OpenPose-style pose images and JSON |

### Advanced Nodes

| Node | Extra Control |
| --- | --- |
| **Sapiens2 Segmentation Advanced** | Overlay opacity, foreground mask, preserve background, raw result |
| **Sapiens2 Normal Advanced** | Foreground mask, preserve background, raw result |
| **Sapiens2 Pointmap Mesh Advanced** | Mesh stride, depth scale, smoothing, filters, splat size, normal texture options |
| **Sapiens2 Pose Advanced** | Detector thresholds, keypoint threshold, render size, fallback bbox, flip test |

## Segmentation Part Selection

Segmentation part rows are compact and visual in the node UI:

```text
<enable> <part group> <detail> <remove>
```

Examples:

- `Face / all`: face-neck + eyeglass + lip + teeth + tongue
- `Face / skin`: face-neck only
- `Mouth / all`: lip + teeth + tongue
- `Arm / left lower`: left lower arm
- `Leg / upper`: both upper legs
- `Clothing / upper`: upper clothing

If no part rows are added, the node merges all foreground body parts.

## Pointmap Export

**Sapiens2 Pointmap** can export:

- `points`: lightweight colored point cloud
- `splats`: fuller point-based rendering
- `mesh`: textured triangle mesh with optional normal detail

The easy node includes practical presets:

- `camera_lens`: `default`, `wide`, `telephoto`
- `quality`: `low`, `mid`, `high`, `super high`
- `mesh_smoothing`: `off`, `light`, `balanced`, `strong`, `extra smooth`
- `normal_detail`: `off`, `subtle`, `balanced`, `strong`

Connect a normal map from **Sapiens2 Normal** to improve mesh shading when `render_mode = mesh`.

Note: pointmap mesh export reconstructs the visible surface from a single image. It can connect and fill the front-facing pointmap, but it does not infer a hidden backside or produce a watertight full body/object volume.

## Pose Targets

**Sapiens2 Pose** supports these output targets:

- `BODY_25`
- `308-keypoint`
- `COCO_18`
- `OpenPose hand 21 + 21`
- `OpenPose face 70`

Outputs include a black-background pose render, an overlay preview, and OpenPose-style JSON. Raw Sapiens2 keypoint data is retained in the JSON as `sapiens_keypoints_2d`.

Pose detection uses a local RTMDet file when available:

```text
ComfyUI/models/sapiens2/detector/rtmdet_m.pth
```

If RTMDet is not available, pose falls back to Hugging Face DETR. RTMDet support can require `mmdet`, `mmengine`, and `mmcv`; those packages are not installed automatically because they can affect the PyTorch/CUDA stack.

## Tips

- Start with `0.4b` or `0.8b` while building a graph, then switch to a larger model for final output.
- Use segmentation masks before pointmap export to reduce background geometry.
- Use the normal output as `normal_map` for mesh pointmap exports when you want more surface detail.
- If a wide-angle image creates exaggerated depth, try `camera_lens = wide`.
- If mesh output looks too noisy, lower quality or increase smoothing before reaching for the advanced node.

## Troubleshooting

| Problem | Try This |
| --- | --- |
| Model download is slow | The first run downloads large Sapiens2 weights; later runs reuse local files |
| CUDA/PyTorch changed after install | Reinstall the correct ComfyUI torch stack, then rerun `python install.py` |
| MPS segmentation labels look wrong | Use `cuda` or `cpu`, or manually test another PyTorch/MPS build |
| Pose detector packages are missing | Use the DETR fallback, or install RTMDet dependencies only if they match your environment |
| Mesh has too much background | Feed a segmentation mask into the pointmap node |
