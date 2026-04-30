# ComfyUI-Sapiens2-Easy

ComfyUI custom nodes for Meta Sapiens2, designed around one idea:

**start easy, stay flexible when you need detail.**

This project wraps Sapiens2 human-centric vision models with a small, practical ComfyUI interface. Pick a task, pick a model size, pick a device, and the loader will use the local model if it exists or download it from Hugging Face when it does not.

## Highlights

- **One model loader for every task**
  Segmentation, normal, pointmap, and pose all use the same loader interface.

- **Auto download, auto reuse**
  Models are downloaded into `ComfyUI/models/sapiens2/` and reused on the next run. The loader reports where the model came from.

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
| Pointmap | Depth-style preview plus sampled `.glb` point cloud |
| Pose | 308-keypoint pose preview plus OpenPose-style JSON |

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

### Sapiens2 Normal

Runs normal estimation.

Inputs:

- `model`: normal model
- `image`: input image
- `mask`: optional mask

Outputs:

- `normal_map`: normal visualization image

### Sapiens2 Pointmap

Runs pointmap estimation and writes a sampled GLB point cloud.

Inputs:

- `model`: pointmap model
- `image`: input image
- `mask`: optional mask

Outputs:

- `preview`: pointmap/depth-style preview
- `pointmap_glb`: generated `.glb` path

### Sapiens2 Pose

Runs 308-keypoint pose estimation.

Inputs:

- `model`: pose model
- `image`: input image
- `bboxes`: optional person boxes. If omitted, people are detected automatically.

Outputs:

- `preview`: merged keypoint/bone preview
- `openpose_json`: OpenPose-style JSON string with Sapiens2 keypoints

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
