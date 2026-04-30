from pathlib import Path


_FOLDER_SUBDIRS = {
    "sapiens2_seg": "seg",
    "sapiens2_normal": "normal",
    "sapiens2_pointmap": "pointmap",
    "sapiens2_albedo": "albedo",
    "sapiens2_pose": "pose",
    "sapiens2_detector": "detector",
}


def get_folder_paths_module():
    try:
        import folder_paths

        return folder_paths
    except ImportError:
        return None


def _add_model_folder_path(folder_paths, name: str, path: Path, extensions: set[str]) -> None:
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(name, str(path), is_default=True)
        return

    existing = folder_paths.folder_names_and_paths.get(name)
    if existing:
        paths, existing_extensions = existing
        paths = list(paths)
        if str(path) not in paths:
            paths.append(str(path))
        extensions = set(extensions)
        extensions.update(existing_extensions)
        folder_paths.folder_names_and_paths[name] = (paths, extensions)
        return

    folder_paths.folder_names_and_paths[name] = ([str(path)], set(extensions))


def register_model_folders() -> None:
    folder_paths = get_folder_paths_module()
    if folder_paths is None:
        return

    sapiens2_dir = Path(folder_paths.models_dir) / "sapiens2"
    sapiens2_dir.mkdir(parents=True, exist_ok=True)
    supported = set(getattr(folder_paths, "supported_pt_extensions", [".pt", ".pth"]))
    supported.update({".safetensors", ".bin"})

    _add_model_folder_path(folder_paths, "sapiens2", sapiens2_dir, supported)

    for name, subdir in _FOLDER_SUBDIRS.items():
        path = sapiens2_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        _add_model_folder_path(folder_paths, name, path, supported)


def get_model_root() -> Path:
    folder_paths = get_folder_paths_module()
    if folder_paths is not None:
        return Path(folder_paths.models_dir) / "sapiens2"
    return Path.cwd() / "models" / "sapiens2"
