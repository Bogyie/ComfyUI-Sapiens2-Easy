# ComfyUI-Sapiens2-Easy

[English](README.md) | [Korean](README.ko.md)

From one image to masks, normals, 3D pointmaps, and pose, without leaving ComfyUI.

[![GitHub stars](https://img.shields.io/github/stars/Bogyie/ComfyUI-Sapiens2-Easy?style=social)](https://github.com/Bogyie/ComfyUI-Sapiens2-Easy/stargazers)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![ComfyUI Registry](https://img.shields.io/badge/ComfyUI-Registry-111111)](https://registry.comfy.org/ko/nodes/comfyui-sapiens2-easy)
[![Sapiens2](https://img.shields.io/badge/Meta-Sapiens2-5b5bd6)](https://github.com/facebookresearch/sapiens2)

<p align="center">
  <img src="docs/images/sapiens2-workflow-overview.png" alt="Full Sapiens2 ComfyUI workflow" width="92%" />
</p>

Build a full human-analysis graph with segmentation, normal estimation, pointmap/GLB export, and pose outputs.

<table>
  <tr>
    <td width="50%"><img src="docs/images/sapiens2-pointmap-render-modes.png" alt="Sapiens2 pointmap render modes" /></td>
    <td width="50%"><img src="docs/images/sapiens2-pose-targets.png" alt="Sapiens2 pose target examples" /></td>
  </tr>
  <tr>
    <td align="center"><strong>Export pointmaps as points, splats, or textured mesh GLB.</strong></td>
    <td align="center"><strong>Generate pose outputs for BODY_25, COCO_18, 308-keypoint, hands, and face.</strong></td>
  </tr>
</table>

**ComfyUI-Sapiens2-Easy** turns Meta Sapiens2 into a small set of ComfyUI-native nodes. Pick a task, model size, and device, then chain the outputs into masks, normal maps, 3D previews, GLB files, or pose JSON.

## Why Use It

- **One loader for every task**: segmentation, normal, pointmap, and pose all share the same model loading flow.
- **Auto download and reuse**: official Hugging Face weights are saved under `ComfyUI/models/sapiens2/`.
- **Mask-first human workflows**: segment the person, then feed that mask into normal or pointmap nodes.
- **Single-image 3D export**: export pointmaps as points, splats, or textured `.glb` meshes.
- **Pose outputs for downstream tools**: generate `BODY_25`, `COCO_18`, `308-keypoint`, hand, and face targets.
- **ComfyUI-safe install path**: the installer avoids silently replacing your PyTorch/CUDA runtime stack.

## What It Supports

| Task | Output |
| --- | --- |
| **Segmentation** | Body-part masks, merged mask, selected-area preview, raw labels |
| **Normal** | Sapiens2 surface normal map |
| **Pointmap** | Depth-style preview and `.glb` export as points, splats, or mesh |
| **Pose** | 308-keypoint Sapiens2 pose plus OpenPose-style image and JSON outputs |

Supported model sizes: `0.4b`, `0.8b`, `1b`, `5b`

## Start Here

| Link | What It Is |
| --- | --- |
| [Install and setup](docs/GUIDE.md) | Full usage guide with install notes, model behavior, node details, and troubleshooting |
| [Korean guide](docs/GUIDE.ko.md) | 한국어 상세 가이드 |
| [ComfyUI Registry](https://registry.comfy.org/ko/nodes/comfyui-sapiens2-easy) | Registry listing for this node pack |

Recommended first graph:

```text
Image Load
  -> Sapiens2 Segmentation
  -> Sapiens2 Normal
  -> Sapiens2 Pointmap
  -> ComfyUI 3D Preview
```

Good first settings: `model_size = 1b`, `device = cuda`, `quality = high`, `mesh_smoothing = balanced`.

## Node Set

| Easy Nodes | Advanced Nodes |
| --- | --- |
| Sapiens2 Model Loader | Sapiens2 Segmentation Advanced |
| Sapiens2 Manual Model Loader | Sapiens2 Normal Advanced |
| Sapiens2 Segmentation | Sapiens2 Pointmap Mesh Advanced |
| Sapiens2 Normal | Sapiens2 Pose Advanced |
| Sapiens2 Pointmap |  |
| Sapiens2 Pose |  |

## Scope

This repository is a ComfyUI adapter around Meta Sapiens2. It does not train, modify, or redistribute the official Sapiens2 models. Model weights and upstream materials are governed by Meta's Sapiens2 license.

## Credits

- Meta Sapiens2: https://github.com/facebookresearch/sapiens2
- ComfyUI: https://github.com/comfyanonymous/ComfyUI

## License

This repository's original adapter code and documentation are licensed under the MIT License. See [LICENSE.md](LICENSE.md).

Meta's Sapiens2 models, weights, upstream source code, algorithms, and documentation are governed by Meta's Sapiens2 License: https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
