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

"""The hand-written CPU kernels (csrc/), built on first use.

``load()`` compiles the extension with torch's JIT builder into
``TORCH_EXTENSIONS_DIR`` (the executor image warms that cache at build
time, so a pod never compiles) and caches the module. ``available()``
says whether a compiler and headers were there; the callers fall back to
the plain-aten path when they were not, so a machine without a
toolchain still runs, slowly.

``VXPU_KERNELS=off`` disables them (the aten path is then the one run,
which is how the two are compared).
"""

import os
import threading

_lock = threading.Lock()
_module = None
_error = None
_enabled = True

SOURCES = [
    os.path.join(os.path.dirname(__file__), "csrc", "fp4_kernels.cpp"),
    os.path.join(os.path.dirname(__file__), "csrc", "fetch.cpp"),
]


def _has_libcurl():
    if os.environ.get("VXPU_FETCH", "").lower() in ("0", "off", "requests", "fallback"):
        return False
    import shutil
    import subprocess
    if shutil.which("curl-config") is not None:
        try:
            res = subprocess.run(["curl-config", "--libs"], capture_output=True, text=True)
            if res.returncode == 0:
                return True
        except Exception:
            pass
    if shutil.which("pkg-config") is not None:
        try:
            res = subprocess.run(["pkg-config", "--exists", "libcurl"], capture_output=True)
            if res.returncode == 0:
                return True
        except Exception:
            pass
    for inc in ["/usr/include", "/usr/local/include", "/opt/homebrew/include"]:
        if os.path.exists(os.path.join(inc, "curl", "curl.h")):
            return True
    return False


def set_enabled(on):
    """Switch the kernels off (and back on) at runtime: the ops then take
    the plain-aten paths. For A/B comparisons on one loaded engine."""
    global _enabled
    _enabled = bool(on)


def load():
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
        if os.environ.get("VXPU_KERNELS", "").lower() in ("0", "off", "false", "aten"):
            _error = "disabled by VXPU_KERNELS"
            return None
        try:
            from torch.utils.cpp_extension import load as jit_load
            import sys
            cflags, ldflags = ["-O3"], []
            if sys.platform != "darwin":  # gcc/clang with an OpenMP runtime; Apple clang has none
                cflags.append("-fopenmp")
                ldflags.append("-fopenmp")
            if _has_libcurl():
                cflags.append("-DHAVE_LIBCURL")
                ldflags.append("-lcurl")
            _module = jit_load(name="vxpu_fp4_kernels", sources=SOURCES,
                               extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)
        except Exception as e:  # noqa: BLE001  (no compiler, no headers: the aten path remains)
            _error = f"{type(e).__name__}: {e}"
            print(f"[vxpu] CPU kernels unavailable, using the plain-aten path: {_error.splitlines()[0][:200]}", flush=True)
            return None
        has_fetch_str = f", fetch: {_module.has_fetch()}" if hasattr(_module, "has_fetch") else ""
        print(f"[vxpu] CPU kernels loaded (avx512: {_module.has_avx512()}, neon: {_module.has_neon()}{has_fetch_str})", flush=True)
        return _module


def available():
    return load() is not None


def has_fetch():
    """Whether the C++ libcurl range fetcher is available."""
    mod = load()
    return mod is not None and getattr(mod, "has_fetch", lambda: False)()


def why_not():
    """The build error, when available() is False."""
    load()
    return _error
