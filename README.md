# ComfyUI-Sapiens2-Easy

ComfyUI custom nodes for Meta Sapiens2, designed around one idea:

**start easy, stay flexible when you need detail.**

This project wraps Sapiens2 human-centric vision models with a small, practical ComfyUI interface. Pick a task, pick a model size, pick a device, and the loader will use the local model if it exists or download it from Hugging Face when it does not.

## Highlights

- **One model loader for every task**
  Segmentation, normal, pointmap, and pose all use the same loader interface.

- **Auto download, auto reuse**
  Models are downloaded into `ComfyUI/models/sapiens2/` and reused on the next run. Download, model loading, batch inference, pose detection, and pointmap GLB export progress are shown in ComfyUI when supported.

- **Beginner-friendly segmentation selection**
  Add part rows visually instead of editing JSON. Pick a part group, choose detail, and get a ready-to-use merged mask.

- **Smart body-part groups**
  Left/right and upper/lower labels are grouped into natural choices like `Hand`, `Arm`, `Leg`, `Clothing`, `Face`, and `Mouth`.

- **Useful outputs by default**
  Segmentation returns `preview`, `merged_mask`, `masks`, and `labels`. Pointmap exports GLB. Pose returns a preview and OpenPose-style JSON.

- **ComfyUI environment safe install**
  The installer avoids silently replacing your PyTorch, CUDA, xformers, or related ComfyUI runtime stack.

## Supported Tasks

| Task | What You Get |
| --- | --- |
| Segmentation | Body-part masks, merged mask, selected-area preview, raw labels |
| Normal | Surface normal map |
| Pointmap | Depth-style preview plus `.glb` export as points, splats, or mesh |
| Pose | 308-keypoint pose preview plus selectable OpenPose-compatible outputs |

Supported model sizes:

```text
0.4b, 0.8b, 1b, 5b
```

Supported devices:

```text
auto, cuda, mps, cpu
```

`auto` prefers CUDA when available, otherwise CPU. It does not automatically select MPS because some PyTorch/MPS combinations can produce incorrect Sapiens2 segmentation label maps. You can still choose `mps` manually.

## Nodes

### Sapiens2 Model Loader

Loads the requested model, downloading it first if needed.

Options:

- `task`: `segmentation`, `normal`, `pointmap`, `pose`
- `model_size`: `0.4b`, `0.8b`, `1b`, `5b`
- `device`: `auto`, `cuda`, `mps`, `cpu`

Outputs:

- `model`: task-aware `SAPIENS2_MODEL`

### Sapiens2 Manual Model Loader

Loads a checkpoint from a path you provide. Use this when you manage model files yourself or cannot download through Hugging Face from the ComfyUI environment.

Options:

- `task`: `segmentation`, `normal`, `pointmap`, `pose`
- `checkpoint_path`: local `.safetensors` or checkpoint path
- `model_size`: `auto`, `0.4b`, `0.8b`, `1b`, `5b`
- `device`: `auto`, `cuda`, `mps`, `cpu`
- `detector_path`: optional pose detector path. If empty for pose, an already-downloaded default detector is used.

Outputs:

- `model`: task-aware `SAPIENS2_MODEL`

### Sapiens2 Segmentation

Runs Sapiens2 body segmentation and returns Comfy-friendly mask outputs.

Inputs:

- `model`: segmentation model
- `image`: input image

Options:

- `invert`: invert the merged mask
- `parts`: visual part rows in the node UI

Outputs:

- `preview`: input image with selected parts overlaid
- `merged_mask`: one mask merged from enabled rows
- `masks`: mask batch for selected labels
- `labels`: raw class-id data and label metadata

Segmentation part rows are intentionally compact:

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

### Sapiens2 Segmentation Advanced

Use this when you need the same segmentation output with extra control.

Additional options:

- `overlay_opacity`: preview overlay strength
- `preserve_background`: keep the original background outside the optional input mask
- `mask`: optional input mask limiting the output area

Additional outputs:

- `foreground_mask`: full foreground mask from the segmentation model
- `result`: raw Sapiens2 inference metadata

### Sapiens2 Normal

Runs normal estimation.

Inputs:

- `model`: normal model
- `image`: input image
- `preview_mode`: `result`, `overlay`, `side_by_side`, `source`
- `mask`: optional mask

Outputs:

- `normal_map`: normal visualization using the selected preview mode

### Sapiens2 Normal Advanced

Adds `foreground_mask`, raw `result`, and `preserve_background` for masked workflows.

### Sapiens2 Pointmap

Runs pointmap estimation and writes a GLB as points, splats, or a textured mesh.

Inputs:

- `model`: pointmap model
- `image`: input image
- `preview_mode`: `result`, `overlay`, `side_by_side`, `source`
- `camera_lens`: `default`, `wide`, or `telephoto`. `wide` compresses Z depth for wide-angle images that look too deep; `telephoto` expands Z depth for flatter telephoto images.
- `render_mode`: `points`, `splats`, or `mesh`
- `quality`: `low`, `mid`, `high`, or `super high`. Used to calculate the point budget for point/splat output and mesh stride for mesh output.
- `mask`: optional mask

Outputs:

- `preview`: pointmap/depth-style preview using the selected preview mode
- `pointmap_glb`: generated `.glb` path list, also shown in ComfyUI's 3D preview UI when available. The output honors the optional input mask, is centered around its bounding box, and is oriented for GLB viewers.

Basic mesh output applies light surface smoothing and uses a more permissive edge filter to preserve thin regions. Use the advanced node when you need unsmoothed mesh export and manual geometry controls.

### Sapiens2 Pointmap Mesh Advanced

Runs pointmap estimation and writes a GLB for 3D preview/export workflows, with additional controls for point, splat, or textured mesh output. Mesh mode produces connected triangles with UVs and the source image embedded as texture.

Inputs:

- `model`: pointmap model
- `image`: input image used for both inference and texture
- `preview_mode`: `result`, `overlay`, `side_by_side`, `source`
- `render_mode`: `points`, `splats`, or `mesh`
- `filename_prefix`: output filename prefix
- `mesh_stride`: mesh resolution step. `1` is highest detail, higher values are lighter.
- `rtol`: 3x3 depth-jump tolerance for removing silhouette/edge triangles
- `min_depth`, `max_depth`: valid Z range
- `center_mesh`: center vertices around their bounding box
- `flip_y`: flip the vertical axis for a conventional GLB Y-up view
- `flip_z`: flip the depth axis for conventional GLB front/back orientation
- `depth_scale`: scale Z depth before mesh or splat export. Values below `1.0` reduce exaggerated wide-angle depth.
- `xy_scale`: scale X/Y before export
- `depth_bias`: add a constant Z offset before export
- `max_points`: manual point budget used when `render_mode` is `points`
- `splat_size`: splat quad size in scene units. `0` uses automatic sizing.
- `splat_max_points`: point budget used when `render_mode` is `splats`
- `mesh_smooth_iterations`, `mesh_smooth_strength`: surface smoothing used when `render_mode` is `mesh`
- `mask`: optional foreground mask

Note: pointmap mesh export reconstructs the visible surface from a single image. It can connect and fill the front-facing pointmap, but it does not infer a true hidden backside or watertight full body/object volume.

Outputs:

- `preview`: pointmap/depth-style preview using the selected preview mode
- `glb_paths`: generated `.glb` path list
- `model_3d`: first generated `.glb` path, suitable for ComfyUI 3D preview nodes

### Sapiens2 Pose

Runs 308-keypoint pose estimation.

Inputs:

- `model`: pose model
- `image`: input image
- `target`: output format for `openpose_image` and `openpose_json`
  - `BODY_25`
  - `308-keypoint`
  - `COCO_18`
  - `OpenPose hand 21 + 21`
  - `OpenPose face 70`
- `bboxes`: optional person boxes. If omitted, people are detected automatically.

Outputs:

- `openpose_image`: black-background pose render for the selected target
- `preview`: selected target rendered over the source image
- `openpose_json`: OpenPose-style JSON string for the selected target, with raw 308-keypoint data retained as `sapiens_keypoints_2d`

### Sapiens2 Pose Advanced

Use this when you want OpenPose-compatible outputs but need detector and render controls.

Additional options:

- `keypoint_threshold`: minimum keypoint score used for rendering and masks
- `bbox_threshold`: person detector score threshold
- `nms_threshold`: detector non-maximum suppression threshold
- `radius`, `thickness`: render size controls
- `fallback_full_image_bbox`: use the full image if no person bbox is detected
- `flip_test`: run pose flip-test refinement
- `show_points`, `show_skeleton`: choose what the pose preview renders

Additional outputs:

- `keypoint_mask`: pose keypoint mask
- `result`: raw 308-keypoint pose result

## Install

Clone or place this repository under `ComfyUI/custom_nodes/`, then run the installer with the same Python environment ComfyUI uses:

```bash
python install.py
```

The installer:

- installs this node's Python requirements
- clones the official Sapiens2 source into `vendor/sapiens2`
- checks that important runtime packages import correctly
- protects the existing ComfyUI PyTorch/CUDA/xformers stack with temporary pip constraints

If you already manage dependencies yourself:

```bash
python install.py --no-deps
```

If you only want to clone or update the vendored Sapiens2 source:

```bash
python install.py --skip-deps
```

If you want to use an existing Sapiens2 checkout:

```bash
python install.py --skip-clone
export SAPIENS2_REPO=/path/to/facebookresearch/sapiens2
```

## PyTorch / CUDA Safety

Do **not** install the upstream Sapiens2 `requirements.txt` directly into your ComfyUI venv unless you know exactly what it will change. It may request a different PyTorch stack than the one ComfyUI is already using.

This repository's `install.py` creates a temporary constraints file that pins the currently installed versions of packages such as:

```text
torch, torchvision, torchaudio, xformers, triton, NVIDIA CUDA wheels
```

If pip tries to replace that stack, installation fails instead of silently breaking your ComfyUI environment.

The official Sapiens2 project currently lists Python 3.12+ and PyTorch 2.7+ as its supported environment. These nodes do not force-install PyTorch. Keep the ComfyUI PyTorch build that matches your CUDA/MPS setup.

## Hugging Face Models

`Sapiens2 Model Loader` maps task and size to the official Hugging Face repositories:

```text
segmentation -> facebook/sapiens2-seg-{0.4b,0.8b,1b,5b}
normal       -> facebook/sapiens2-normal-{0.4b,0.8b,1b,5b}
pointmap     -> facebook/sapiens2-pointmap-{0.4b,0.8b,1b,5b}
pose         -> facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}
```

Downloaded files are saved under:

```text
ComfyUI/models/sapiens2/<task>/
```

Pose detection uses a local RTMDet file when available:

```text
ComfyUI/models/sapiens2/detector/rtmdet_m.pth
```

If RTMDet is not available, pose falls back to Hugging Face DETR.

RTMDet support can require `mmdet`, `mmengine`, and `mmcv`. Those packages are not installed automatically because they may affect the PyTorch/CUDA stack.

## Why Easy?

Sapiens2 is powerful, but raw task-specific pipelines can be awkward inside ComfyUI. This project keeps the public node set small:

```text
Sapiens2 Model Loader
Sapiens2 Segmentation
Sapiens2 Normal
Sapiens2 Pointmap
Sapiens2 Pose
```

The goal is not to expose every internal knob on day one. The goal is to make the first workflow obvious, then leave enough detail for advanced masking, compositing, pose, and geometry workflows.

## License

This custom node repository's original adapter code and documentation are licensed under the MIT License. See [LICENSE.md](LICENSE.md).

That license applies only to this repository's wrapper code and documentation. It does not apply to Meta's Sapiens2 models, weights, upstream source code, algorithms, inference/training/fine-tuning code, or documentation.

Sapiens2 Materials are governed by Meta's Sapiens2 License:

- https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md

You are responsible for following Meta's license terms before downloading, using, modifying, or distributing Sapiens2 Materials.
