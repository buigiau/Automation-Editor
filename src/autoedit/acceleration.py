"""Optional NVIDIA runtime discovery; changes only this Python process on Windows."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_DLL_HANDLES = []
_DLL_DIRECTORIES = set()
_PRELOADED = set()


def device_settings(device="auto", device_index=0):
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Inference device must be auto, cpu or cuda")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("CUDA device_index must be a nonnegative integer")
    return device, device_index


def is_accelerator_error(error):
    """Do not disguise bad model files or invalid inputs as GPU failures."""
    message = str(error).lower()
    return any(token in message for token in (
        "cuda", "cudnn", "cublas", "cufft", "nvrtc", "loadlibrary",
        "out of memory", "failed to load library", "not found or cannot be loaded",
        "compute type", "compute_type", "no kernel image",
    ))


def prepare_nvidia_runtime():
    """Find optional pip-installed NVIDIA DLLs without changing the system PATH.

    CTranslate2's native LoadLibrary calls also need a process-local PATH entry.
    Retain add_dll_directory handles for dependent DLLs throughout inference.
    Linux users supply their CUDA libraries via the normal loader configuration.
    """
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return
    for root in spec.submodule_search_locations or []:
        for component in ("cublas", "cuda_runtime", "cuda_nvrtc", "cufft", "nvjitlink", "cudnn"):
            for subdirectory in ("bin", "lib"):
                directory = str(Path(root) / component / subdirectory)
                if directory in _DLL_DIRECTORIES or not Path(directory).is_dir():
                    continue
                _DLL_HANDLES.append(os.add_dll_directory(directory))
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
                _DLL_DIRECTORIES.add(directory)
        # CTranslate2 wheels can bundle a different cuDNN dispatcher version.
        # Load this environment's dispatcher before importing CTranslate2, so
        # it matches the installed cuDNN component DLLs during deferred inference.
        dispatcher = Path(root) / "cudnn" / "bin" / "cudnn64_9.dll"
        if dispatcher.is_file() and str(dispatcher) not in _PRELOADED:
            import ctypes
            try:
                _DLL_HANDLES.append(ctypes.WinDLL(str(dispatcher)))
                _PRELOADED.add(str(dispatcher))
            except OSError:
                # Missing dependencies are handled by model initialization or
                # inference, where auto can recover and cuda can fail explicitly.
                pass
