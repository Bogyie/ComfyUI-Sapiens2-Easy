from .albedo import Sapiens2AlbedoChannels, Sapiens2AlbedoCombineChannels, Sapiens2AlbedoSelectChannel
from .normal import Sapiens2NormalChannels, Sapiens2NormalCombineChannels, Sapiens2NormalSelectChannel
from .pointmap import Sapiens2PointmapChannels, Sapiens2PointmapCombineChannels, Sapiens2PointmapDepthRange, Sapiens2PointmapSelectChannel
from .pose import (
    Sapiens2PoseCombineGroups,
    Sapiens2PoseGroupMasks,
    Sapiens2PoseSelectGroup,
    Sapiens2SavePoseJSON,
)
from .segmentation import (
    Sapiens2Segmentation,
    Sapiens2SegmentationCombine,
    Sapiens2SegmentationPartMasks,
    Sapiens2SegmentationSelectPart,
)
from .unified import Sapiens2LoadModel, Sapiens2LoadModelAdvanced, Sapiens2Run, Sapiens2RunAdvanced
from .utilities import Sapiens2MaskProcess


NODE_CLASS_MAPPINGS = {
    "Sapiens2LoadModel": Sapiens2LoadModel,
    "Sapiens2LoadModelAdvanced": Sapiens2LoadModelAdvanced,
    "Sapiens2Run": Sapiens2Run,
    "Sapiens2RunAdvanced": Sapiens2RunAdvanced,
    "Sapiens2PoseGroupMasks": Sapiens2PoseGroupMasks,
    "Sapiens2PoseSelectGroup": Sapiens2PoseSelectGroup,
    "Sapiens2PoseCombineGroups": Sapiens2PoseCombineGroups,
    "Sapiens2SavePoseJSON": Sapiens2SavePoseJSON,
    "Sapiens2Segmentation": Sapiens2Segmentation,
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
    "Sapiens2LoadModel": "Sapiens2 Model",
    "Sapiens2LoadModelAdvanced": "Sapiens2 Model Advanced",
    "Sapiens2Run": "Sapiens2 Run",
    "Sapiens2RunAdvanced": "Sapiens2 Run Advanced",
    "Sapiens2PoseGroupMasks": "Sapiens2 Pose Group Masks",
    "Sapiens2PoseSelectGroup": "Sapiens2 Pose Select Group",
    "Sapiens2PoseCombineGroups": "Sapiens2 Pose Combine Groups",
    "Sapiens2SavePoseJSON": "Sapiens2 Save Pose JSON",
    "Sapiens2Segmentation": "Sapiens2 Segmentation",
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
