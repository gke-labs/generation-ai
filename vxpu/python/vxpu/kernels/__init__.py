# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The kernel library, and the hand-written CPU kernels (csrc/) built on first use.

This package root is the kernel library's home, organizing variants under
kernels/<op>/<variant>.py. It also loads the native C++ extension that later
CPU variants call.

load() compiles the extension with torch's JIT builder into
TORCH_EXTENSIONS_DIR (the executor image warms that cache at build
time, so a pod never compiles) and caches the module. available()
says whether a compiler and headers were there; the callers fall back to
the plain-aten path when they were not, so a machine without a
toolchain still runs, slowly.

VXPU_KERNELS=off disables them (the aten path is then the one run,
which is how the two are compared).
"""

# pyright: strict

import importlib
import os
import pkgutil
import threading
from typing import Any, Callable, TypeVar

from .. import plan_pb2
from ..plan import Applies, Numerics

_lock = threading.Lock()
_module: Any = None
_error: str | None = None
_enabled: bool = True

SOURCES: list[str] = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "csrc", name)
    for name in ("fp4_kernels.cpp", "node_server.cpp")
]

F = TypeVar("F", bound=Callable[..., Any])


def set_enabled(on: bool) -> None:
    """Switch the kernels off (and back on) at runtime: the ops then take
    the plain-aten paths. For A/B comparisons on one loaded engine."""
    global _enabled
    _enabled = bool(on)


def _has_liburing() -> bool:
    """Whether liburing headers are installed: links -luring and defines VXPU_HAVE_IO_URING."""
    for inc in ("/usr/include", "/usr/local/include"):
        if os.path.exists(os.path.join(inc, "liburing.h")):
            return True
    return False


def load() -> Any:
    """The extension module, or None if it cannot be built here (or the
    kernels are switched off)."""
    global _module, _error
    if not _enabled:
        return None
    if _module is not None or _error is not None:
        return _module
    with _lock:
        if _module is not None or _error is not None:
            return _module
        from ..plan import env_flag
        if not env_flag("VXPU_KERNELS", True, extra_false=("aten",))[0]:
            _error = "disabled by VXPU_KERNELS"
            return None
        try:
            import torch.utils.cpp_extension as cpp_extension
            import sys
            cflags: list[str] = ["-O3"]
            ldflags: list[str] = []
            if sys.platform != "darwin":  # gcc/clang with an OpenMP runtime; Apple clang has none
                cflags.append("-fopenmp")
                ldflags.append("-fopenmp")
            if _has_liburing():
                cflags.append("-DVXPU_HAVE_IO_URING=1")
                ldflags.append("-luring")
            cpp_ext: Any = cpp_extension
            _module = cpp_ext.load(name="vxpu_fp4_kernels", sources=SOURCES,
                                   extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)
        except Exception as e:  # noqa: BLE001  (no compiler, no headers: the aten path remains)
            _error = f"{type(e).__name__}: {e}"
            print(f"[vxpu] CPU kernels unavailable, using the plain-aten path: {_error.splitlines()[0][:200]}", flush=True)
            return None
        print(f"[vxpu] CPU kernels loaded (avx512: {_module.has_avx512()}, neon: {_module.has_neon()}, io_uring: {_module.has_io_uring()})", flush=True)
        return _module


def available() -> bool:
    return load() is not None


def why_not() -> str | None:
    """The build error, when available() is False."""
    load()
    return _error


def variant(
    devices: tuple[str, ...],
    numerics: Numerics,
    when: str,
    applies: Applies | None = None,
    site_applies: Callable[..., bool] | None = None,
    lowerable: bool = False,
    condition: plan_pb2.KernelCondition | dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorates a kernel variant function to register it with the op registry.

    Derives the op and variant name from the decorated function's module path
    (vxpu.kernels.<op>.<variant>). Calls plan.implementation to register it.
    """
    def register(fn: F) -> F:
        parts = fn.__module__.split(".")
        if len(parts) < 2:
            raise ValueError(f"cannot derive op and variant name from module {fn.__module__!r}")
        op = f"vxpu::{parts[-2]}"
        name = parts[-1]
        from .. import plan
        plan.implementation(
            op=op,
            name=name,
            devices=devices,
            numerics=numerics,
            when=when,
            applies=applies,
            site_applies=site_applies,
            lowerable=lowerable,
            condition=condition,
        )(fn)
        return fn

    return register


def discover() -> list[str]:
    """Imports every variant module under vxpu.kernels.<op>.<variant>."""
    imported: list[str] = []
    for op_info in pkgutil.iter_modules(__path__):
        if not op_info.ispkg or op_info.name.startswith("_"):
            continue
        op_pkg_name = f"{__name__}.{op_info.name}"
        op_pkg = importlib.import_module(op_pkg_name)
        op_path = getattr(op_pkg, "__path__", None)
        if op_path is None:
            continue
        for mod_info in pkgutil.iter_modules(op_path):
            if mod_info.ispkg or mod_info.name.startswith("_"):
                continue
            mod_name = f"{op_pkg_name}.{mod_info.name}"
            importlib.import_module(mod_name)
            imported.append(mod_name)
    return imported
