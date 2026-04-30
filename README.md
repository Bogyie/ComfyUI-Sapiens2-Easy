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
python install.py
```

`install.py` installs this node's requirements with a temporary constraints file that pins the currently installed `torch`, `torchvision`, `torchaudio`, `xformers`, `triton`, and NVIDIA CUDA wheel versions. If pip tries to change that stack, installation fails instead of silently replacing the ComfyUI build.

After pip finishes, `install.py` imports the runtime packages listed in `requirements.txt` so missing packages such as `accelerate`, `transformers`, `timm`, or `tqdm` are caught during installation instead of later inside a node run.

`install.py` also clones the official Sapiens2 repo into `vendor/sapiens2`. You can skip that clone with `python install.py --skip-clone`, then set `SAPIENS2_REPO=/path/to/facebookresearch/sapiens2` or fill `sapiens_repo_path` in the loader node.

The official Sapiens2 project currently lists Python 3.12+ and PyTorch 2.7+ as its supported environment. These nodes do not force-install PyTorch, so keep the ComfyUI PyTorch build that matches your CUDA/MPS setup.

### PyTorch / CUDA Safety Notes

Do not install the upstream Sapiens2 `requirements.txt` directly into your ComfyUI venv. It may request a different PyTorch stack than the one ComfyUI is already using.

Avoid running plain `pip install -r requirements.txt` unless you know how pip will resolve transitive dependencies in your environment. Prefer:

```bash
python install.py
```

If ComfyUI already has most dependencies and you want the most conservative install, use:

```bash
python install.py --no-deps
```

`--no-deps` still installs the packages listed directly in this repository's `requirements.txt`, but it does not install their transitive dependencies. If the post-install import check reports a missing dependency, rerun without `--no-deps` or install the reported package manually in the ComfyUI venv.

If `install.py` says no `torch` package is detected, install the correct ComfyUI PyTorch/CUDA/MPS build first. The `--allow-torch-install` flag exists for fresh experimental environments, but it is not recommended for an existing ComfyUI venv.

Use `python install.py --skip-deps` only when you intentionally want to skip pip installation and just clone/update the local Sapiens2 repo path.

When using an automatic custom-node manager, check whether it runs `pip install -r requirements.txt` before `install.py`. If it does, verify the resolved packages before applying the install, or install manually with `python install.py` so the current PyTorch stack can be pinned.

RTMDet pose detection is optional and uses `mmdet`/`mmengine`/`mmcv` when you provide `rtmdet_m.pth`. Those packages are not installed automatically because they can affect the PyTorch/CUDA stack. Without RTMDet, the easy pose path uses the Hugging Face detector fallback.

## Model Loading

Use one node for every task:

- `Sapiens2 Model`

Pick `task` and `model_size`. The node first checks the local cache under `ComfyUI/models/sapiens2/<task>/`; if the file is missing, it downloads the matching Hugging Face checkpoint. Dense tasks are loaded through the official Sapiens2 config files and `init_model()` path. The `load_info` output tells you exactly where the checkpoint, config, and, for pose, detector were loaded from.

Use `Sapiens2 Model Advanced` if you need `source = local_only`, `source = download`, custom `repo_id`, `filename`, `checkpoint_path`, `revision`, `token`, `device`, `dtype`, or pose detector paths.

## Quick Start

The nodes are designed so the first run needs only the obvious inputs. Leave advanced fields at their defaults unless you have a specific reason to change them.

Basic workflow:

1. Add `Sapiens2 Model`.
2. Pick `task` and `model_size`.
3. Connect `model` and your `IMAGE` to `Sapiens2 Run`.

Task-specific convenience:

If you only need a mask or channel, connect the same `model` and `IMAGE` directly to a convenience node such as `Sapiens2 Segmentation`, `Sapiens2 Normal Channels`, `Sapiens2 Pointmap Depth Range`, or `Sapiens2 Pose Group Masks`. You do not need to wire a separate intermediate output first.

### Easy vs Advanced

Easy nodes keep the first run short:

- `Sapiens2 Model`: choose only task and model size. Missing checkpoints are downloaded, cached checkpoints are reused.
- `Sapiens2 Run`: run the loaded model with default visualization settings.
- `Sapiens2 Segmentation`: run body segmentation and return `masks`, `merged_mask`, `segm`, and `preview`.

Advanced nodes expose the knobs only when you need them:

- `Sapiens2 Model Advanced`: custom checkpoint path, Hugging Face repo/file, revision, token, device, dtype, pose detector, and upstream repo path.
- `Sapiens2 Run Advanced`: overlay opacity, background masking, pose thresholds, NMS, pose drawing size, and flip-test options.
- `Sapiens2 Segmentation Combine Parts`: 29 part toggles plus grow, blur, and invert while keeping the same `masks`, `merged_mask`, `segm`, and `preview` outputs.

## Nodes

- `Sapiens2 Model`: local-or-download model preparation for segmentation, normal, pointmap, pose, and custom albedo.
- `Sapiens2 Run`: unified inference. For dense models it returns the preview image, main mask, auxiliary mask, and result object. For pose it returns pose overlay, keypoint mask, and result object.
- `Sapiens2 Model Advanced`: model loading with explicit paths, Hugging Face overrides, dtype/device controls, and pose detector settings.
- `Sapiens2 Run Advanced`: unified inference with visualization and pose controls exposed.
- `Sapiens2 Segmentation`: simple body segmentation. Outputs all 29 part masks as a mask batch, the merged foreground mask, Impact Pack-compatible `SEGS`, and a preview image.
- `Sapiens2 Segmentation Part Masks`: runs segmentation and outputs all 29 class masks.
- `Sapiens2 Segmentation Combine Parts`: runs segmentation and merges selected class toggles. Outputs `masks`, `merged_mask`, `segm`, and `preview`.
- `Sapiens2 Segmentation Select Part`: runs segmentation for one selected class. Outputs the selected mask, `SEGS`, and a preview.
- `Sapiens2 Normal Channels`: runs normal estimation and outputs x/y/z masks plus normal image.
- `Sapiens2 Normal Select/Combine Channel(s)`: runs normal estimation and extracts or combines channels.
- `Sapiens2 Pointmap Channels`: runs pointmap estimation and outputs x/y/z-depth/valid masks plus depth image.
- `Sapiens2 Pointmap Select/Combine Channel(s)`: runs pointmap estimation and extracts or combines channels.
- `Sapiens2 Pointmap Depth Range`: runs pointmap estimation and masks a z-depth range.
- `Sapiens2 Albedo Channels`: runs albedo estimation and outputs red/green/blue/luminance masks plus albedo image.
- `Sapiens2 Albedo Select/Combine Channel(s)`: runs albedo estimation and extracts or combines channels.
- `Sapiens2 Pose Group Masks`: runs pose and outputs body, face, hands, feet, and extra masks.
- `Sapiens2 Pose Select/Combine Group(s)`: runs pose and extracts or combines pose groups.
- `Sapiens2 Save Pose JSON`: runs pose and writes per-image keypoints, bboxes, scores, names, and skeleton links to JSON.
- `Sapiens2 Process Mask`: generic threshold/grow/shrink/blur/invert utility.

Pose uses the same `SAPIENS2_MODEL` input as every other task. The detector is resolved inside `Sapiens2 Model` for pose models.

## Hugging Face Mapping

`Sapiens2 Model` auto-maps these official repos:

```text
pretrain     -> facebook/sapiens2-pretrain-{0.1b,0.4b,0.8b,1b,1b-4k,5b}
segmentation -> facebook/sapiens2-seg-{0.4b,0.8b,1b,5b}
normal       -> facebook/sapiens2-normal-{0.4b,0.8b,1b,5b}
pointmap     -> facebook/sapiens2-pointmap-{0.4b,0.8b,1b,5b}
pose         -> facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}
pose detector -> local rtmdet_m.pth when present, otherwise facebook/detr-resnet-101-dc5
```

Downloaded files are saved under `ComfyUI/models/sapiens2/<task>/` by default. Pose detector files are saved under `ComfyUI/models/sapiens2/detector/`. If `ComfyUI/models/sapiens2/detector/rtmdet_m.pth` exists, pose detection follows the reference RTMDet/MMDetection path. If it is missing, the easy node downloads and uses the lighter Hugging Face DETR detector path so first use still works.

## Design Notes

Default settings favor the easiest working path: local cache first, download only when missing, automatic device/dtype selection, built-in Hugging Face repo mapping, and pose detector resolution inside the same model node.

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
