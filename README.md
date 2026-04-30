# Sapiens2 ComfyUI Custom Nodes

Simple ComfyUI nodes for Meta's Sapiens2 human-centric vision models.

Supported tasks:

- Body-part segmentation
- Surface normal estimation
- Pointmap preview and GLB point-cloud export
- 308-keypoint pose estimation

## Install

Place this folder under `ComfyUI/custom_nodes/`, then run the installer with the same Python environment ComfyUI uses:

```bash
python install.py
```

`install.py` installs this node's requirements with a temporary constraints file that pins the currently installed `torch`, `torchvision`, `torchaudio`, `xformers`, `triton`, and NVIDIA CUDA wheel versions. If pip tries to change that stack, installation fails instead of silently replacing the ComfyUI build.

After pip finishes, `install.py` imports the runtime packages listed in `requirements.txt` so missing packages such as `accelerate`, `transformers`, `timm`, or `tqdm` are caught during installation instead of later inside a node run.

`install.py` also clones the official Sapiens2 repo into `vendor/sapiens2`. You can skip that clone with `python install.py --skip-clone`, then set `SAPIENS2_REPO=/path/to/facebookresearch/sapiens2`.

The official Sapiens2 project currently lists Python 3.12+ and PyTorch 2.7+ as its supported environment. These nodes do not force-install PyTorch, so keep the ComfyUI PyTorch build that matches your CUDA/MPS setup.

`device = auto` prefers CUDA when available, otherwise CPU. It intentionally does not auto-select MPS because Sapiens2 segmentation can produce visibly incorrect label maps on MPS with some PyTorch builds. You can still choose `mps` explicitly if you want to test it.

### PyTorch / CUDA Safety Notes

Do not install the upstream Sapiens2 `requirements.txt` directly into your ComfyUI venv. It may request a different PyTorch stack than the one ComfyUI is already using.

Prefer:

```bash
python install.py
```

If ComfyUI already has most dependencies and you want the most conservative install, use:

```bash
python install.py --no-deps
```

`--no-deps` still installs the packages listed directly in this repository's `requirements.txt`, but it does not install their transitive dependencies. If the post-install import check reports a missing dependency, rerun without `--no-deps` or install the reported package manually in the ComfyUI venv.

Use `python install.py --skip-deps` only when you intentionally want to skip pip installation and just clone/update the local Sapiens2 repo path.

RTMDet pose detection is optional and uses `mmdet`/`mmengine`/`mmcv` when you provide `ComfyUI/models/sapiens2/detector/rtmdet_m.pth`. Those packages are not installed automatically because they can affect the PyTorch/CUDA stack. Without RTMDet, pose uses the Hugging Face DETR fallback.

## Nodes

### Sapiens2 Model Loader

Loads or downloads a model.

Inputs:

- `task`: `segmentation`, `normal`, `pointmap`, or `pose`
- `model_size`: `0.4b`, `0.8b`, `1b`, or `5b`
- `device`: `auto`, `cuda`, `mps`, or `cpu`

Output:

- `model`: a `SAPIENS2_MODEL` object carrying its task.

### Sapiens2 Segmentation

Runs body-part segmentation.

Inputs:

- `model`: segmentation model only
- `image`

Options:

- `invert`: inverts the merged mask
- `parts`: dynamic part rows in the UI. Click `+ add part`, pick a class, and enable/disable rows. If no rows are added, all foreground body parts are merged.

Outputs:

- `preview`: 29-class color overlay image
- `merged_mask`: one merged mask from selected parts
- `masks`: selected part masks as a Comfy mask batch
- `labels`: raw class-id label data

### Sapiens2 Normal

Runs normal estimation.

Inputs:

- `model`: normal model only
- `image`
- `mask` optional

Output:

- `normal_map`: normal visualization image

### Sapiens2 Pointmap

Runs pointmap estimation and exports a sampled GLB point cloud.

Inputs:

- `model`: pointmap model only
- `image`
- `mask` optional

Outputs:

- `preview`: depth preview image
- `pointmap_glb`: path to the generated `.glb` file

### Sapiens2 Pose

Runs 308-keypoint pose estimation.

Inputs:

- `model`: pose model only
- `image`
- `bboxes` optional. If omitted, the node detects people automatically.

Outputs:

- `preview`: merged bone/keypoint preview image
- `openpose_json`: OpenPose-style JSON string with Sapiens2 keypoints

## Hugging Face Mapping

`Sapiens2 Model Loader` auto-maps these official repos:

```text
segmentation -> facebook/sapiens2-seg-{0.4b,0.8b,1b,5b}
normal       -> facebook/sapiens2-normal-{0.4b,0.8b,1b,5b}
pointmap     -> facebook/sapiens2-pointmap-{0.4b,0.8b,1b,5b}
pose         -> facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}
pose detector -> local rtmdet_m.pth when present, otherwise facebook/detr-resnet-101-dc5
```

Downloaded files are saved under `ComfyUI/models/sapiens2/<task>/` by default. Pose detector files are saved under `ComfyUI/models/sapiens2/detector/`.

## License

This custom node repository's original adapter code is licensed under the MIT License. See [LICENSE.md](LICENSE.md).

That MIT license applies only to this repository's wrapper code and documentation. It does not apply to Meta's Sapiens2 models, weights, upstream source code, algorithms, inference/training/fine-tuning code, or documentation.

Sapiens2 Materials are governed by Meta's Sapiens2 License:

- https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md

You are responsible for following Meta's license terms before downloading, using, modifying, or distributing Sapiens2 Materials.
