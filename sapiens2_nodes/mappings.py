from .easy import (
    Sapiens2ModelLoader,
    Sapiens2ModelLoaderManual,
    Sapiens2Normal,
    Sapiens2Pointmap,
    Sapiens2Pose,
    Sapiens2Segmentation,
)


NODE_CLASS_MAPPINGS = {
    "Sapiens2ModelLoader": Sapiens2ModelLoader,
    "Sapiens2ModelLoaderManual": Sapiens2ModelLoaderManual,
    "Sapiens2Segmentation": Sapiens2Segmentation,
    "Sapiens2Normal": Sapiens2Normal,
    "Sapiens2Pointmap": Sapiens2Pointmap,
    "Sapiens2Pose": Sapiens2Pose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sapiens2ModelLoader": "Sapiens2 Model Loader",
    "Sapiens2ModelLoaderManual": "Sapiens2 Manual Model Loader",
    "Sapiens2Segmentation": "Sapiens2 Segmentation",
    "Sapiens2Normal": "Sapiens2 Normal",
    "Sapiens2Pointmap": "Sapiens2 Pointmap",
    "Sapiens2Pose": "Sapiens2 Pose",
}
