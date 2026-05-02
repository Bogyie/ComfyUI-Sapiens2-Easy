# ComfyUI-Sapiens2-Easy 가이드

[한국어](GUIDE.ko.md) | [English](GUIDE.md) | [README로 돌아가기](../README.ko.md)

이 문서는 설치, 모델 동작, 노드 세부 정보, 문제 해결을 README에서 분리해 정리한 가이드입니다.

## 설치

이 저장소를 `ComfyUI/custom_nodes/` 아래에 클론하고, ComfyUI를 실행하는 것과 같은 Python 환경에서 설치 스크립트를 실행합니다.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Bogyie/ComfyUI-Sapiens2-Easy.git
cd ComfyUI-Sapiens2-Easy
python install.py
```

설치 후 ComfyUI를 재시작하세요. 노드는 다음 카테고리에 나타납니다.

```text
Sapiens2
Sapiens2/Advanced
```

설치 스크립트는 다음 작업을 수행합니다.

- 이 노드에 필요한 Python 의존성을 설치합니다.
- 공식 Sapiens2 소스를 `vendor/sapiens2`에 클론합니다.
- 중요한 런타임 패키지가 정상적으로 import되는지 확인합니다.
- 임시 pip constraints로 기존 ComfyUI PyTorch/CUDA/xformers 스택을 보호합니다.

유용한 설치 옵션:

```bash
# Python 의존성을 설치하고 Sapiens2 소스를 클론
python install.py

# 이 노드의 직접 의존성만 설치
python install.py --no-deps

# 의존성 설치 없이 Sapiens2 소스만 클론
python install.py --skip-deps

# 직접 관리하는 Sapiens2 체크아웃 사용
python install.py --skip-clone
export SAPIENS2_REPO=/path/to/facebookresearch/sapiens2
```

기존 ComfyUI 환경에 upstream Sapiens2의 `requirements.txt`를 그대로 설치하지 마세요. PyTorch/CUDA 버전을 직접 관리하려는 경우가 아니라면 환경이 깨질 수 있습니다.

## 모델

지원 모델 크기:

```text
0.4b, 0.8b, 1b, 5b
```

지원 디바이스:

```text
auto, cuda, mps, cpu
```

`auto`는 CUDA가 있으면 CUDA를 우선 사용하고, 없으면 CPU를 사용합니다. 일부 PyTorch/MPS 조합에서 Sapiens2 segmentation label map이 잘못 나올 수 있어 MPS는 자동 선택하지 않습니다. 필요하면 `mps`를 직접 선택할 수 있습니다.

다운로드된 파일은 아래 경로에 저장됩니다.

```text
ComfyUI/models/sapiens2/<task>/
```

로더는 작업별로 공식 Hugging Face 저장소를 사용합니다.

```text
segmentation -> facebook/sapiens2-seg-{0.4b,0.8b,1b,5b}
normal       -> facebook/sapiens2-normal-{0.4b,0.8b,1b,5b}
pointmap     -> facebook/sapiens2-pointmap-{0.4b,0.8b,1b,5b}
pose         -> facebook/sapiens2-pose-{0.4b,0.8b,1b,5b}
```

## 추천 첫 워크플로

1. **Image Load**를 추가합니다.
2. **Sapiens2 Model Loader**를 추가하고 `task = segmentation`으로 설정합니다.
3. **Sapiens2 Segmentation**으로 foreground/person 마스크를 만듭니다.
4. 또 다른 **Sapiens2 Model Loader**를 추가하고 `task = normal`로 설정합니다.
5. **Sapiens2 Normal**에 이미지와 선택적 마스크를 연결합니다.
6. 또 다른 **Sapiens2 Model Loader**를 추가하고 `task = pointmap`으로 설정합니다.
7. **Sapiens2 Pointmap**에 image, mask, normal map을 연결합니다.
8. 생성된 `.glb`를 `points`, `splats`, `mesh` 중 원하는 모드로 프리뷰하거나 내보냅니다.

처음 써보기 좋은 설정:

```text
model_size: 1b
device: cuda
preview_mode: result
render_mode: points or mesh
quality: high
mesh_smoothing: balanced
normal_detail: balanced
```

그래프를 만들 때는 `0.4b` 또는 `0.8b`로 빠르게 반복하고, 최종 결과 품질이 중요할 때 `1b` 또는 `5b`로 올리는 것을 추천합니다.

## 노드

### 기본 노드

| 노드 | 용도 |
| --- | --- |
| **Sapiens2 Model Loader** | 작업별 Sapiens2 모델 다운로드 및 로드 |
| **Sapiens2 Manual Model Loader** | 로컬 체크포인트 수동 로드 |
| **Sapiens2 Segmentation** | 신체 부위 마스크와 선택 영역 프리뷰 생성 |
| **Sapiens2 Normal** | 선택적 마스크를 포함한 normal map 생성 |
| **Sapiens2 Pointmap** | pointmap 결과를 points, splats, mesh `.glb`로 내보내기 |
| **Sapiens2 Pose** | Sapiens2/OpenPose 스타일 pose 이미지와 JSON 생성 |

### 고급 노드

| 노드 | 추가 제어 |
| --- | --- |
| **Sapiens2 Segmentation Advanced** | overlay opacity, foreground mask, preserve background, raw result |
| **Sapiens2 Normal Advanced** | foreground mask, preserve background, raw result |
| **Sapiens2 Pointmap Mesh Advanced** | mesh stride, depth scale, smoothing, filters, splat size, normal texture options |
| **Sapiens2 Pose Advanced** | detector threshold, keypoint threshold, render size, fallback bbox, flip test |

## Segmentation Part 선택

Segmentation part row는 노드 UI에서 간결하게 다룰 수 있습니다.

```text
<enable> <part group> <detail> <remove>
```

예시:

- `Face / all`: face-neck + eyeglass + lip + teeth + tongue
- `Face / skin`: face-neck only
- `Mouth / all`: lip + teeth + tongue
- `Arm / left lower`: left lower arm
- `Leg / upper`: both upper legs
- `Clothing / upper`: upper clothing

part row를 추가하지 않으면 foreground 신체 부위 전체를 병합합니다.

## Pointmap Export

**Sapiens2 Pointmap**은 다음 형태로 내보낼 수 있습니다.

- `points`: 가벼운 colored point cloud
- `splats`: 더 꽉 차 보이는 point 기반 렌더링
- `mesh`: 선택적 normal detail을 포함한 textured triangle mesh

기본 노드는 실용적인 프리셋을 제공합니다.

- `camera_lens`: `default`, `wide`, `telephoto`
- `quality`: `low`, `mid`, `high`, `super high`
- `mesh_smoothing`: `off`, `light`, `balanced`, `strong`, `extra smooth`
- `normal_detail`: `off`, `subtle`, `balanced`, `strong`

`render_mode = mesh`일 때 **Sapiens2 Normal**의 normal map을 연결하면 mesh shading을 더 풍부하게 만들 수 있습니다.

참고: pointmap mesh export는 단일 이미지에서 보이는 표면을 재구성합니다. 앞쪽 표면을 연결하고 채울 수는 있지만, 보이지 않는 뒷면을 추론하거나 watertight full body/object volume을 만들지는 않습니다.

## Pose Target

**Sapiens2 Pose**는 다음 출력 target을 지원합니다.

- `BODY_25`
- `308-keypoint`
- `COCO_18`
- `OpenPose hand 21 + 21`
- `OpenPose face 70`

출력에는 black-background pose render, overlay preview, OpenPose 스타일 JSON이 포함됩니다. Raw Sapiens2 keypoint 데이터는 JSON의 `sapiens_keypoints_2d`에 보존됩니다.

Pose detection은 사용 가능한 경우 로컬 RTMDet 파일을 사용합니다.

```text
ComfyUI/models/sapiens2/detector/rtmdet_m.pth
```

RTMDet이 없으면 Hugging Face DETR fallback을 사용합니다. RTMDet 지원에는 `mmdet`, `mmengine`, `mmcv`가 필요할 수 있으며, 이 패키지들은 PyTorch/CUDA 스택에 영향을 줄 수 있어 자동 설치하지 않습니다.

## 팁

- 그래프를 만들 때는 `0.4b` 또는 `0.8b`로 시작하고, 최종 출력 때 더 큰 모델로 바꾸세요.
- pointmap export 전에 segmentation mask를 사용하면 배경 geometry를 줄일 수 있습니다.
- mesh pointmap export에서 표면 디테일을 높이고 싶다면 normal output을 `normal_map`으로 연결하세요.
- 광각 이미지에서 depth가 과장되어 보이면 `camera_lens = wide`를 시도해보세요.
- mesh output이 너무 거칠다면 advanced node로 가기 전에 quality를 낮추거나 smoothing을 올려보세요.

## 문제 해결

| 문제 | 시도해볼 것 |
| --- | --- |
| 모델 다운로드가 느림 | 첫 실행에는 큰 Sapiens2 가중치를 다운로드합니다. 이후에는 로컬 파일을 재사용합니다. |
| 설치 후 CUDA/PyTorch가 바뀜 | 올바른 ComfyUI torch stack을 다시 설치한 뒤 `python install.py`를 다시 실행하세요. |
| MPS에서 segmentation label이 이상함 | `cuda` 또는 `cpu`를 사용하거나 다른 PyTorch/MPS 빌드를 테스트하세요. |
| Pose detector 패키지가 없음 | DETR fallback을 사용하거나, 환경에 맞을 때만 RTMDet 의존성을 설치하세요. |
| Mesh에 배경이 너무 많음 | segmentation mask를 pointmap node에 연결하세요. |
