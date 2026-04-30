import torch


ARCH_SPECS = {
    "sapiens2_0.4b": {"embed_dim": 1024},
    "sapiens2_0.8b": {"embed_dim": 1280},
    "sapiens2_1b": {"embed_dim": 1536},
    "sapiens2_5b": {"embed_dim": 2432},
}
MODEL_SIZE_CHOICES = ("0.4b", "0.8b", "1b", "5b")

DEVICES = ("auto", "cuda", "mps", "cpu")
SAPIENS2_HF_ORG = "facebook"
POSE_DETECTOR_REPO = "facebook/detr-resnet-101-dc5"
POSE_RTMDET_FILENAME = "rtmdet_m.pth"
POSE_RTMDET_CONFIG_REL = "sapiens/pose/tools/vis/rtmdet_m_640-8xb32_coco-person.py"
POSE_KEYPOINT_COUNT = 308
POSE_CONFIG_DATASET = "shutterstock_goliath_3po"
POSE_CONFIG_RESOLUTION = "1024x768"

SEG_PARTS = (
    "Background",
    "Apparel",
    "Eyeglass",
    "Face_Neck",
    "Hair",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)

SEG_PART_OPTIONS = tuple(f"{idx:02d}: {name}" for idx, name in enumerate(SEG_PARTS))
SEG_CLASS_COUNT = len(SEG_PARTS)

SEG_PALETTE = torch.tensor(
    [
        [50, 50, 50],
        [255, 218, 0],
        [14, 204, 182],
        [128, 200, 255],
        [255, 0, 109],
        [189, 0, 204],
        [255, 0, 218],
        [0, 160, 204],
        [0, 255, 145],
        [204, 0, 131],
        [182, 0, 255],
        [255, 109, 0],
        [0, 255, 255],
        [72, 0, 255],
        [204, 131, 0],
        [255, 0, 0],
        [72, 255, 0],
        [189, 204, 0],
        [182, 255, 0],
        [102, 0, 204],
        [32, 72, 204],
        [0, 145, 255],
        [14, 204, 0],
        [0, 128, 72],
        [235, 205, 119],
        [115, 227, 112],
        [157, 113, 143],
        [132, 93, 50],
        [82, 21, 114],
    ],
    dtype=torch.float32,
) / 255.0
