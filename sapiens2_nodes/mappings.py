from .albedo import Sapiens2AlbedoChannels, Sapiens2AlbedoCombineChannels, Sapiens2AlbedoSelectChannel
from .huggingface import (
    Sapiens2HuggingFaceDownload,
    Sapiens2HuggingFaceModelLoader,
    Sapiens2PoseDetectorHuggingFaceDownload,
    Sapiens2PoseHuggingFaceModelLoader,
)
from .inference import Sapiens2DenseInference
from .model_loading import Sapiens2ModelLoader
from .normal import Sapiens2NormalChannels, Sapiens2NormalCombineChannels, Sapiens2NormalSelectChannel
from .pointmap import Sapiens2PointmapChannels, Sapiens2PointmapCombineChannels, Sapiens2PointmapDepthRange, Sapiens2PointmapSelectChannel
from .pose import (
    Sapiens2PoseCombineGroups,
    Sapiens2PoseGroupMasks,
    Sapiens2PoseInference,
    Sapiens2PoseModelLoader,
    Sapiens2PosePersonDetection,
    Sapiens2PoseSelectGroup,
    Sapiens2SavePoseJSON,
)
from .segmentation import Sapiens2SegmentationCombine, Sapiens2SegmentationPartMasks, Sapiens2SegmentationSelectPart
from .utilities import Sapiens2MaskProcess


NODE_CLASS_MAPPINGS = {
    "Sapiens2ModelLoader": Sapiens2ModelLoader,
    "Sapiens2HuggingFaceDownload": Sapiens2HuggingFaceDownload,
    "Sapiens2HuggingFaceModelLoader": Sapiens2HuggingFaceModelLoader,
    "Sapiens2PoseDetectorHuggingFaceDownload": Sapiens2PoseDetectorHuggingFaceDownload,
    "Sapiens2PoseHuggingFaceModelLoader": Sapiens2PoseHuggingFaceModelLoader,
    "Sapiens2DenseInference": Sapiens2DenseInference,
    "Sapiens2PoseModelLoader": Sapiens2PoseModelLoader,
    "Sapiens2PosePersonDetection": Sapiens2PosePersonDetection,
    "Sapiens2PoseInference": Sapiens2PoseInference,
    "Sapiens2PoseGroupMasks": Sapiens2PoseGroupMasks,
    "Sapiens2PoseSelectGroup": Sapiens2PoseSelectGroup,
    "Sapiens2PoseCombineGroups": Sapiens2PoseCombineGroups,
    "Sapiens2SavePoseJSON": Sapiens2SavePoseJSON,
    "Sapiens2SegmentationPartMasks": Sapiens2SegmentationPartMasks,
    "Sapiens2SegmentationCombine": Sapiens2SegmentationCombine,
    "Sapiens2SegmentationSelectPart": Sapiens2SegmentationSelectPart,
    "Sapiens2MaskProcess": Sapiens2MaskProcess,
    "Sapiens2NormalChannels": Sapiens2NormalChannels,
    "Sapiens2NormalSelectChannel": Sapiens2NormalSelectChannel,
    "Sapiens2NormalCombineChannels": Sapiens2NormalCombineChannels,
    "Sapiens2PointmapChannels": Sapiens2PointmapChannels,
    "Sapiens2PointmapSelectChannel": Sapiens2PointmapSelectChannel,
    "Sapiens2PointmapDepthRange": Sapiens2PointmapDepthRange,
    "Sapiens2PointmapCombineChannels": Sapiens2PointmapCombineChannels,
    "Sapiens2AlbedoChannels": Sapiens2AlbedoChannels,
    "Sapiens2AlbedoSelectChannel": Sapiens2AlbedoSelectChannel,
    "Sapiens2AlbedoCombineChannels": Sapiens2AlbedoCombineChannels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sapiens2ModelLoader": "Load Sapiens2 Dense Model",
    "Sapiens2HuggingFaceDownload": "Download Sapiens2 from Hugging Face",
    "Sapiens2HuggingFaceModelLoader": "Load Sapiens2 from Hugging Face",
    "Sapiens2PoseDetectorHuggingFaceDownload": "Download Sapiens2 Pose Detector from Hugging Face",
    "Sapiens2PoseHuggingFaceModelLoader": "Load Sapiens2 Pose from Hugging Face",
    "Sapiens2DenseInference": "Sapiens2 Dense Inference",
    "Sapiens2PoseModelLoader": "Load Sapiens2 Pose Model",
    "Sapiens2PosePersonDetection": "Sapiens2 Pose Person Detection",
    "Sapiens2PoseInference": "Sapiens2 Pose Inference",
    "Sapiens2PoseGroupMasks": "Sapiens2 Pose Group Masks",
    "Sapiens2PoseSelectGroup": "Sapiens2 Pose Select Group",
    "Sapiens2PoseCombineGroups": "Sapiens2 Pose Combine Groups",
    "Sapiens2SavePoseJSON": "Sapiens2 Save Pose JSON",
    "Sapiens2SegmentationPartMasks": "Sapiens2 Segmentation Part Masks",
    "Sapiens2SegmentationCombine": "Sapiens2 Segmentation Combine Parts",
    "Sapiens2SegmentationSelectPart": "Sapiens2 Segmentation Select Part",
    "Sapiens2MaskProcess": "Sapiens2 Process Mask",
    "Sapiens2NormalChannels": "Sapiens2 Normal Channels",
    "Sapiens2NormalSelectChannel": "Sapiens2 Normal Select Channel",
    "Sapiens2NormalCombineChannels": "Sapiens2 Normal Combine Channels",
    "Sapiens2PointmapChannels": "Sapiens2 Pointmap Channels",
    "Sapiens2PointmapSelectChannel": "Sapiens2 Pointmap Select Channel",
    "Sapiens2PointmapDepthRange": "Sapiens2 Pointmap Depth Range",
    "Sapiens2PointmapCombineChannels": "Sapiens2 Pointmap Combine Channels",
    "Sapiens2AlbedoChannels": "Sapiens2 Albedo Channels",
    "Sapiens2AlbedoSelectChannel": "Sapiens2 Albedo Select Channel",
    "Sapiens2AlbedoCombineChannels": "Sapiens2 Albedo Combine Channels",
}
