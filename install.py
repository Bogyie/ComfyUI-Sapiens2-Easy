import argparse
import importlib
import re
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "sapiens2"
REQUIREMENTS = ROOT / "requirements.txt"

IMPORT_NAME_OVERRIDES = {
    "opencv-python": "cv2",
    "pillow": "PIL",
}

TORCH_STACK_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
    "triton",
    "pytorch-triton",
    "nvidia-cublas-cu11",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu11",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu11",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu11",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu11",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu11",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu11",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu11",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu11",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu11",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu11",
    "nvidia-nvtx-cu12",
)


def _installed_versions(package_names):
    versions = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _torch_stack_constraints(versions):
    return "\n".join(f"{name}=={version}" for name, version in sorted(versions.items())) + "\n"


def _check_torch_stack_unchanged(before, allow_new_torch_stack: bool = False):
    after = _installed_versions(TORCH_STACK_PACKAGES)
    changed = {
        name: (version, after.get(name))
        for name, version in before.items()
        if after.get(name) != version
    }
    if not allow_new_torch_stack:
        changed.update(
            {
                name: ("not installed", version)
                for name, version in after.items()
                if name not in before
            }
        )
    if changed:
        details = ", ".join(
            f"{name}: {old} -> {new or 'uninstalled'}"
            for name, (old, new) in sorted(changed.items())
        )
        raise RuntimeError(f"PyTorch stack changed during install: {details}")


def _requirement_package_name(line: str) -> str:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http://", "https://")):
        return ""
    line = line.split(";", 1)[0].strip()
    line = re.split(r"\s*(?:==|~=|!=|<=|>=|<|>)\s*", line, maxsplit=1)[0]
    return line.split("[", 1)[0].strip().lower()


def _required_imports():
    imports = []
    for line in REQUIREMENTS.read_text().splitlines():
        package = _requirement_package_name(line)
        if not package:
            continue
        imports.append((package, IMPORT_NAME_OVERRIDES.get(package, package.replace("-", "_"))))
    return imports


def _check_required_imports():
    missing = []
    for package, import_name in _required_imports():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            missing.append(f"{package} ({import_name}): {exc}")
    if missing:
        details = "\n  ".join(missing)
        raise RuntimeError(
            "Some runtime dependencies could not be imported after installation:\n"
            f"  {details}\n"
            "If you used --no-deps, rerun without --no-deps or install the reported "
            "package/import issue in the ComfyUI venv."
        )


def install_requirements(no_deps: bool = False, allow_torch_install: bool = False):
    if not REQUIREMENTS.exists():
        print(f"No requirements.txt found at {REQUIREMENTS}")
        return

    torch_stack = _installed_versions(TORCH_STACK_PACKAGES)
    if torch_stack:
        print("Keeping existing PyTorch/CUDA package versions:")
        for name, version in sorted(torch_stack.items()):
            print(f"  {name}=={version}")
    else:
        if not allow_torch_install:
            raise RuntimeError(
                "No existing torch package was detected. Install the correct ComfyUI "
                "PyTorch/CUDA/MPS build first, then rerun install.py. "
                "Use --allow-torch-install only if you intentionally want pip to choose torch."
            )
        print("No existing torch package was detected. pip may install torch as a dependency.")

    with tempfile.NamedTemporaryFile("w", prefix="sapiens2_torch_constraints_", suffix=".txt", delete=False) as handle:
        constraints_path = Path(handle.name)
        handle.write(_torch_stack_constraints(torch_stack))

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS),
        "-c",
        str(constraints_path),
    ]
    if no_deps:
        cmd.append("--no-deps")

    try:
        subprocess.check_call(cmd)
        _check_torch_stack_unchanged(
            torch_stack,
            allow_new_torch_stack=allow_torch_install and not torch_stack,
        )
        _check_required_imports()
    finally:
        constraints_path.unlink(missing_ok=True)


def clone_sapiens2():
    if VENDOR.exists():
        print(f"Sapiens2 already exists at {VENDOR}")
        return
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/facebookresearch/sapiens2.git",
            str(VENDOR),
        ]
    )
    print(f"Cloned Sapiens2 into {VENDOR}")


def main():
    parser = argparse.ArgumentParser(description="Install helper for Sapiens2 ComfyUI custom nodes.")
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Compatibility flag. Dependencies are installed by default unless --skip-deps is used.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip pip dependency installation and only clone the official Sapiens2 repo.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Pass --no-deps to pip while installing this node's direct requirements.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Do not clone facebookresearch/sapiens2 into vendor/sapiens2.",
    )
    parser.add_argument(
        "--allow-torch-install",
        action="store_true",
        help="Allow pip to install torch if none is present. Not recommended for an existing ComfyUI venv.",
    )
    args = parser.parse_args()

    if args.install_deps:
        print("--install-deps is no longer required; dependencies install by default.")

    if not args.skip_deps:
        install_requirements(no_deps=args.no_deps, allow_torch_install=args.allow_torch_install)
    else:
        print("Skipping pip dependency install because --skip-deps was passed.")

    if not args.skip_clone:
        clone_sapiens2()

    print("Done.")


if __name__ == "__main__":
    main()
