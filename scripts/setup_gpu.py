"""Install the Windows CUDA stack into the active virtual environment only."""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-directory", type=Path,
                        help="Use previously downloaded wheels; resolve remaining packages from cache")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script with .venv/Scripts/python.exe; a virtual environment is required.")
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    local_wheels = []
    if args.wheel_directory:
        directory = args.wheel_directory.resolve()
        if not directory.is_dir():
            parser.error("wheel-directory must exist")
        local_wheels = ["--find-links", str(directory)]
    if uv:
        base = [uv, "pip"]
        # faster-whisper declares the CPU distribution as a dependency, so do
        # the cleanup AFTER dependency resolution, then reinstall GPU without
        # resolving dependencies again. Both distributions share package files.
        subprocess.run(base + ["install", "--python", sys.executable,
                              "-r", "requirements-gpu.txt"]
                       + local_wheels + (["--offline"] if local_wheels else []),
                       cwd=root, check=True)
        subprocess.run(base + ["uninstall", "--python", sys.executable, "onnxruntime"], check=True)
        subprocess.run(base + ["install", "--python", sys.executable, "--no-deps",
                              "--reinstall-package", "onnxruntime-gpu", "onnxruntime-gpu==1.20.0"]
                       + local_wheels + (["--offline"] if local_wheels else []), check=True)
    else:
        base = [sys.executable, "-m", "pip"]
        subprocess.run(base + ["install", "-r", "requirements-gpu.txt"] + local_wheels, cwd=root, check=True)
        subprocess.run(base + ["uninstall", "-y", "onnxruntime"], check=True)
        subprocess.run(base + ["install", "--force-reinstall", "--no-deps", "onnxruntime-gpu==1.20.0"]
                       + local_wheels, check=True)
    print("GPU dependencies installed. Restart AutoEdit to use the new runtime.")


if __name__ == "__main__":
    main()
