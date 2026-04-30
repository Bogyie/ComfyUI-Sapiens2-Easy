from .advanced import Sapiens2NormalAdvanced, Sapiens2PoseAdvanced, Sapiens2SegmentationAdvanced
from .easy import (
    Sapiens2ModelLoader,
    Sapiens2ModelLoaderManual,
    Sapiens2Normal,
    Sapiens2Pointmap,
    Sapiens2Pose,
    Sapiens2Segmentation,
)
from .pointmap_advanced import Sapiens2PointmapMeshAdvanced, Sapiens2PointmapToMesh


NODE_CLASS_MAPPINGS = {
    "Sapiens2ModelLoader": Sapiens2ModelLoader,
    "Sapiens2ModelLoaderManual": Sapiens2ModelLoaderManual,
    "Sapiens2Segmentation": Sapiens2Segmentation,
    "Sapiens2SegmentationAdvanced": Sapiens2SegmentationAdvanced,
    "Sapiens2Normal": Sapiens2Normal,
    "Sapiens2NormalAdvanced": Sapiens2NormalAdvanced,
    "Sapiens2Pointmap": Sapiens2Pointmap,
    "Sapiens2PointmapToMesh": Sapiens2PointmapToMesh,
    "Sapiens2PointmapMeshAdvanced": Sapiens2PointmapMeshAdvanced,
    "Sapiens2Pose": Sapiens2Pose,
    "Sapiens2PoseAdvanced": Sapiens2PoseAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sapiens2ModelLoader": "Sapiens2 Model Loader",
    "Sapiens2ModelLoaderManual": "Sapiens2 Manual Model Loader",
    "Sapiens2Segmentation": "Sapiens2 Segmentation",
    "Sapiens2SegmentationAdvanced": "Sapiens2 Segmentation Advanced",
    "Sapiens2Normal": "Sapiens2 Normal",
    "Sapiens2NormalAdvanced": "Sapiens2 Normal Advanced",
    "Sapiens2Pointmap": "Sapiens2 Pointmap",
    "Sapiens2PointmapToMesh": "Sapiens2 Pointmap To Mesh",
    "Sapiens2PointmapMeshAdvanced": "Sapiens2 Pointmap Mesh Advanced",
    "Sapiens2Pose": "Sapiens2 Pose",
    "Sapiens2PoseAdvanced": "Sapiens2 Pose Advanced",
}
