# Sapiens2 ComfyUI Custom Nodes

ComfyUI wrapper nodes for Meta's Sapiens2 dense human-centric vision models.

Supported tasks:

- Body-part segmentation
- Surface normal estimation
- Pointmap depth visualization
- Albedo estimation
- 308-keypoint pose estimation

## Install

Place this folder under `ComfyUI/custom_nodes/`, then run the installer with the same Python environment ComfyUI uses:

```bash
python install.py --install-deps
```

`install.py --install-deps` installs this node's requirements with a temporary constraints file that pins the currently installed `torch`, `torchvision`, `torchaudio`, `xformers`, `triton`, and NVIDIA CUDA wheel versions. If pip tries to change that stack, installation fails instead of silently replacing the ComfyUI build.

`install.py` also clones the official Sapiens2 repo into `vendor/sapiens2`. You can skip that clone with `python install.py --install-deps --skip-clone`, then set `SAPIENS2_REPO=/path/to/facebookresearch/sapiens2` or fill `sapiens_repo_path` in the loader node.

The official Sapiens2 project currently lists Python 3.12+ and PyTorch 2.7+ as its supported environment. These nodes do not force-install PyTorch, so keep the ComfyUI PyTorch build that matches your CUDA/MPS setup.

### PyTorch / CUDA Safety Notes

Do not install the upstream Sapiens2 `requirements.txt` directly into your ComfyUI venv. It may request a different PyTorch stack than the one ComfyUI is already using.

Avoid running plain `pip install -r requirements.txt` unless you know how pip will resolve transitive dependencies in your environment. Prefer:

```bash
python install.py --install-deps
```

If ComfyUI already has most dependencies and you want the most conservative install, use:

```bash
python install.py --install-deps --no-deps
```

If `install.py --install-deps` says no `torch` package is detected, install the correct ComfyUI PyTorch/CUDA/MPS build first. The `--allow-torch-install` flag exists for fresh experimental environments, but it is not recommended for an existing ComfyUI venv.

When using an automatic custom-node manager, check whether it runs `pip install -r requirements.txt` before `install.py`. If it does, verify the resolved packages before applying the install, or install manually with `python install.py --install-deps` so the current PyTorch stack can be pinned.

## Checkpoints

Download task checkpoints from the Sapiens2 Hugging Face collection and place them under:

```text
ComfyUI/models/sapiens2/
```

Examples:

```text
ComfyUI/models/sapiens2/seg/sapiens2_1b_seg.safetensors
ComfyUI/models/sapiens2/normal/sapiens2_1b_normal.safetensors
ComfyUI/models/sapiens2/pointmap/sapiens2_1b_pointmap.safetensors
ComfyUI/models/sapiens2/albedo/sapiens2_1b_albedo.safetensors
ComfyUI/models/sapiens2/pose/sapiens2_1b_pose.safetensors
ComfyUI/models/sapiens2/detector/detr-resnet-101-dc5/
```

The loader also accepts an absolute `checkpoint_path`.

The local loader also supports `task = auto` and `arch = auto`; it inspects checkpoint keys and the backbone embedding dimension before building the model. Loaded models are cached by checkpoint path, file mtime/size, device, dtype, task, and arch.

## Quick Start

The nodes are designed so the first run needs only the obvious inputs. Leave advanced fields at their defaults unless you have a specific reason to change them.

Dense tasks:

1. Use `Load Sapiens2 from Hugging Face`.
2. Pick `task` and `model_size`.
3. Connect it to `Sapiens2 Dense Inference` with an `IMAGE`.

Local dense checkpoints:

1. Put checkpoints under `ComfyUI/models/sapiens2/<task>/`.
2. Use `Load Sapiens2 Dense Model`.
3. Leave `task`, `arch`, `device`, and `dtype` as `auto` for normal use.

Pose:

1. Use `Load Sapiens2 Pose from Hugging Face`.
2. Pick `model_size`.
3. Connect it to `Sapiens2 Pose Inference` with an `IMAGE`.

Advanced settings such as custom repo IDs, local paths, revision, token, device, dtype, thresholds, mask grow/blur, and pose rendering controls remain available as optional inputs.

## Nodes

- `Load Sapiens2 Dense Model`: loads a Sapiens2 checkpoint for one task.
- `Download Sapiens2 from Hugging Face`: downloads an official checkpoint and returns its local path.
- `Load Sapiens2 from Hugging Face`: downloads an official checkpoint, then loads it as `SAPIENS2_MODEL`.
- `Download Sapiens2 Pose Detector from Hugging Face`: downloads the DETR person detector used by the pose pipeline.
- `Load Sapiens2 Pose from Hugging Face`: downloads a pose checkpoint and optionally downloads the DETR detector, then loads a pose model.
- `Sapiens2 Dense Inference`: runs the loaded model on ComfyUI `IMAGE` tensors.
- `Load Sapiens2 Pose Model`: loads a local Sapiens2 pose checkpoint and local DETR detector snapshot.
- `Sapiens2 Pose Person Detection`: runs the pose detector only, previews person boxes, and returns reusable `SAPIENS2_BBOXES`.
- `Sapiens2 Pose Inference`: runs person detection, top-down 308-keypoint pose estimation, and returns a rendered pose image, keypoint mask, and raw pose data.
- `Sapiens2 Pose Group Masks`: splits pose keypoints into body, face, left hand, right hand, feet, and extra masks.
- `Sapiens2 Pose Select/Combine Group(s)`: selects or combines pose keypoint groups.
- `Sapiens2 Save Pose JSON`: writes per-image pose predictions, bboxes, keypoint scores, keypoint names, and skeleton links to JSON.
- `Sapiens2 Segmentation Part Masks`: splits body segmentation into 29 separate `MASK` outputs.
- `Sapiens2 Segmentation Combine Parts`: exposes 29 part toggles and merges selected parts into one `MASK`.
- `Sapiens2 Segmentation Select Part`: selects one body part and optionally grows, shrinks, blurs, or inverts it.
- `Sapiens2 Normal Channels`: splits normal output into x/y/z masks and a normal RGB image.
- `Sapiens2 Normal Select/Combine Channel(s)`: selects or combines normal channels.
- `Sapiens2 Pointmap Channels`: splits pointmap output into x/y/z-depth/valid masks and a depth image.
- `Sapiens2 Pointmap Select/Combine Channel(s)`: selects or combines pointmap channels.
- `Sapiens2 Pointmap Depth Range`: masks pixels by raw z-depth range.
- `Sapiens2 Albedo Channels`: splits albedo output into red/green/blue/luminance masks and an albedo image.
- `Sapiens2 Albedo Select/Combine Channel(s)`: selects or combines albedo channels.
- `Sapiens2 Process Mask`: generic threshold/grow/shrink/blur/invert utility for any mask.

For segmentation, the first output is an RGB overlay, `foreground_mask` is all non-background classes, and `aux_mask` contains class ids normalized to `0..1`.

For normal/albedo, the image output is the restored visualization. For pointmap, the image output is a grayscale depth preview and `aux_mask` is normalized depth.

`Sapiens2 Dense Inference` accepts an optional `mask`. With `preserve_background` enabled, masked-out pixels keep the original input image; otherwise they are cleared.

Pose inference uses the official top-down flow: DETR person boxes first, then Sapiens2 pose for each detected person. You can either let `Sapiens2 Pose Inference` run detection internally, or run `Sapiens2 Pose Person Detection` once and pass its `SAPIENS2_BBOXES` output into pose inference. If no person is detected, `fallback_full_image_bbox` can use the full image as a fallback bbox.

## Hugging Face Loader

The Hugging Face nodes auto-map these official Sapiens2 dense task repos:

```text
pretrain     -> facebook/sapiens2-pretrain-{0.1b,0.4b,0.8b,1b,1b-4k,5b}
segmentation -> facebook/sapiens2-seg-{0.4b,0.8b,1b,5b}
normal       -> facebook/sapiens2-normal-{0.4b,0.8b,1b,5b}
pointmap     -> facebook/sapiens2-pointmap-{0.4b,0.8b,1b,5b}
pose         -> facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}
pose detector -> facebook/detr-resnet-101-dc5
```

`Download Sapiens2 from Hugging Face` exposes `task` and `model_size` separately. `Load Sapiens2 from Hugging Face` exposes the same size choices for dense tasks that this node can run directly: segmentation, normal, pointmap, and custom albedo.

Pose has separate Hugging Face nodes because it also needs the DETR person detector snapshot. `Load Sapiens2 Pose from Hugging Face` can download both the pose checkpoint and detector in one node.

Downloaded files are saved under `ComfyUI/models/sapiens2/<task>/` by default. Use `repo_id` and `filename` to override the source, including for custom albedo checkpoints.

## Design Notes

Default settings favor the easiest working path: automatic device/dtype selection, automatic dense checkpoint task/architecture detection, built-in Hugging Face repo mapping, and pose detector download by default.

When you need more control, expose the optional inputs and override only the field you need. The code keeps those advanced paths separate from the common path so basic workflows stay short.

## License

This custom node repository's original adapter code is licensed under the MIT License. See [LICENSE.md](LICENSE.md).

That MIT license applies only to this repository's wrapper code and documentation. It does not apply to Meta's Sapiens2 models, weights, upstream source code, algorithms, inference/training/fine-tuning code, or documentation.

Sapiens2 Materials are governed by Meta's Sapiens2 License:

```text
https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
```

`install.py` can clone the upstream Sapiens2 repository into `vendor/sapiens2`, and the Hugging Face nodes can download Sapiens2 checkpoints. Those cloned or downloaded materials remain under the Sapiens2 License, not this repository's MIT license.

Before using or redistributing Sapiens2 Materials, review the Sapiens2 License carefully. It includes restrictions and obligations around redistribution, research acknowledgement, privacy and biometric-information laws, trade controls, prohibited uses, warranty, liability, termination, and audit rights. This README is only a summary and is not legal advice.
