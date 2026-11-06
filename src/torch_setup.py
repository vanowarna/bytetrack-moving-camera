"""Make PyTorch's bundled CUDA/cuDNN DLLs findable on Windows. Import BEFORE `import torch`.

cuDNN 9 loads its sub-libraries (e.g. cudnn_graph64_9.dll) by bare file name at runtime.
Windows then searches PATH, not torch/lib, and fails with
"Could not locate cudnn_graph64_9.dll" (or picks up another cuDNN installed elsewhere).
Fix: put torch/lib first on PATH and preload the cuDNN DLLs by full path, so later
lookups by name resolve to the already-loaded modules. No effect on Linux/macOS.
"""
import ctypes
import importlib.util
import os
import sys
from pathlib import Path


def _setup_windows_dlls():
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
    if not lib.is_dir():
        return
    os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(str(lib))
    except (OSError, AttributeError):
        pass
    # graph first: the other cuDNN sub-libraries depend on it
    for name in sorted(lib.glob("cudnn*64_9.dll"), key=lambda p: (p.name != "cudnn_graph64_9.dll", p.name)):
        try:
            ctypes.WinDLL(str(name))
        except OSError:
            pass


if sys.platform == "win32":
    _setup_windows_dlls()
