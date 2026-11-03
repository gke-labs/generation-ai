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

"""Check pod environment, caches, scratch, and page cache before a load (#511)."""

import argparse
import ctypes
import glob
import json
import mmap
import os
import shutil
import sys
import tempfile

from .localblobs import _complete, external_files_dir, tensor_key
from . import plan

TOKEN_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
)


def check_blob_dir(manifest, env=os.environ):
    """Verify all checkpoint files in the manifest exist with complete sizes in VXPU_BLOB_DIR.

    Catches unpopulated blob directory from a restaged env.sh without VXPU_BLOB_DIR.
    """
    blob_dir = env.get("VXPU_BLOB_DIR")
    files = manifest.get("files", {})
    names = [entry.get("name", sha) for sha, entry in files.items()]
    if not blob_dir:
        return "FAIL", f"VXPU_BLOB_DIR unset (missing {len(names)} files: {', '.join(names)})"
    missing = [
        entry.get("name", sha)
        for sha, entry in files.items()
        if not _complete(os.path.join(blob_dir, sha), entry["size"])
    ]
    if missing:
        return "FAIL", f"missing {len(missing)} of {len(files)} files: {', '.join(missing)}"
    return "PASS", f"{len(files)} files present in {blob_dir}"


def _external_paths(manifest, binding=None, env=os.environ):
    """File names of external expert banks and n-gram tables expected on disk."""
    ext = (binding or {}).get("external") or manifest.get("external") or {}
    tensors = manifest.get("tensors", {})
    raw_range, _ = plan.env_text("VXPU_LAYER_RANGE", None, env=env)
    layer_range = plan.parse_layer_range(raw_range)
    in_range = lambda layer: layer_range is None or (
        layer >= layer_range[0] and (layer_range[1] is None or layer < layer_range[1])
    )
    names = []

    for layer, spec in ext.get("engram", {}).items():
        if not in_range(int(layer)):
            continue
        for key in ("weight", "scale"):
            if key in spec and spec[key] in tensors:
                ref = tensors[spec[key]]
                names.append(tensor_key(ref))

    in_ram, _ = plan.env_flag("VXPU_EXPERTS_IN_RAM", False, env=env)
    if not in_ram:
        for layer, specs in ext.get("experts", {}).items():
            layer_num = int(layer)
            if in_range(layer_num):
                for name in specs:
                    names.append(f"experts.{layer_num}.{name}")

    return names


def check_external_dir(manifest, binding=None, env=os.environ):
    """Verify external expert banks and n-gram tables exist under external_files_dir when expected.

    Catches missing external files on cold PVC before load (#504).
    Known gap: FileTable reads n-gram tables directly from blob files without
    extracting them under external_files_dir (engine.py when CPU kernels are
    available without a step graph and sources are file://).
    """
    ext = (binding or {}).get("external") or manifest.get("external") or {}
    has_experts = bool(ext.get("experts"))
    has_engram = bool(ext.get("engram"))
    if not has_experts and not has_engram:
        return "PASS", "no external tensors in artifact"

    in_ram, _ = plan.env_flag("VXPU_EXPERTS_IN_RAM", False, env=env)
    names = _external_paths(manifest, binding, env=env)
    if not names:
        if has_experts and in_ram:
            return "PASS", "experts held in RAM"
        return "PASS", "no external tensors in range"

    ext_dir = env.get("VXPU_EXTERNAL_DIR")
    if not ext_dir:
        return "FAIL", f"VXPU_EXTERNAL_DIR unset (missing {len(names)} files)"

    target_dir = external_files_dir(ext_dir, manifest)
    missing = [name for name in names if not os.path.exists(os.path.join(target_dir, name))]
    if missing:
        return "FAIL", f"missing {len(missing)} files: {', '.join(missing)}"

    if has_experts and in_ram:
        return "PASS", f"{len(names)} files present in {ext_dir} (experts held in RAM)"
    return "PASS", f"{len(names)} files present in {ext_dir}"


def check_tokenizer(manifest, env=os.environ):
    """Verify HF_HOME contains tokenizer files for the model repository.

    Catches missing tokenizer cache before model initialization.
    """
    hf_home = env.get("HF_HOME")
    repo = manifest.get("source", {}).get("repo", "")
    if not hf_home:
        return "FAIL", "HF_HOME unset"
    if os.path.isdir(repo):
        if any(os.path.exists(os.path.join(repo, f)) for f in TOKEN_FILES):
            return "PASS", f"repo {repo}"
        return "FAIL", f"repo {repo}"
    norm = repo.replace("/", "--")
    found = any(
        any(f in TOKEN_FILES for f in files)
        for root, _, files in os.walk(hf_home)
        if not repo or norm in root
    )
    if found:
        return "PASS", f"tokenizer present in {hf_home} for {repo}"
    return "FAIL", f"tokenizer files for {repo} missing from {hf_home}"


def check_environment(
    env=os.environ,
    max_map_count_path="/proc/sys/vm/max_map_count",
    core_pattern_path="/proc/sys/kernel/core_pattern",
):
    """Verify offline mode, scratch paths, vm.max_map_count, and core dump limits.

    Catches dlopen exhaustion at default max_map_count and crashes filling boot disk.
    """
    offline = (env.get("HF_HUB_OFFLINE") or "").strip().upper()
    if offline not in ("1", "ON", "YES", "TRUE"):
        return "FAIL", f"HF_HUB_OFFLINE is {env.get('HF_HUB_OFFLINE')!r}, expected 1"

    scratch = os.path.realpath(
        env.get("VXPU_SCRATCH_DIR")
        or ("/mnt/ssd" if os.path.isdir("/mnt/ssd") else "/tmp")
    )
    bad = []
    for var in (
        "TMPDIR",
        "VXPU_AOTI_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        val = env.get(var)
        if not val:
            bad.append(f"{var} unset")
            continue
        try:
            real_val = os.path.realpath(val)
            if os.path.commonpath([real_val, scratch]) != scratch:
                bad.append(f"{var}={val}")
        except (ValueError, OSError):
            bad.append(f"{var}={val}")
    if bad:
        return "FAIL", f"not under scratch {scratch}: {', '.join(bad)}"

    if os.path.exists(max_map_count_path):
        try:
            with open(max_map_count_path) as f:
                max_map = int(f.read().strip())
            if max_map < 1048576:
                return "FAIL", f"vm.max_map_count is {max_map} (minimum 1048576)"
        except (OSError, ValueError) as e:
            return "FAIL", f"cannot read {max_map_count_path}: {e}"

    core_disabled = False
    try:
        import resource

        soft, _ = resource.getrlimit(resource.RLIMIT_CORE)
        if soft == 0:
            core_disabled = True
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    if not core_disabled and os.path.exists(core_pattern_path):
        try:
            with open(core_pattern_path) as f:
                pattern = f.read().strip()
                if pattern in ("/dev/null", "|/bin/false"):
                    core_disabled = True
        except OSError:
            pass

    if not core_disabled:
        return "FAIL", "core dumps enabled (need RLIMIT_CORE soft limit 0 or core_pattern /dev/null)"

    return "PASS", f"offline, paths under {scratch}, max_map_count and core dump limits ok"


def _scan_strays(env=os.environ):
    """Total bytes of lingering temporary files from interrupted operations."""
    strays = 0

    tmpdir = env.get("TMPDIR") or tempfile.gettempdir()
    if os.path.isdir(tmpdir):
        for entry in os.scandir(tmpdir):
            if entry.name.startswith("vxpu-compile-"):
                try:
                    if entry.is_dir():
                        strays += sum(
                            f.stat().st_size
                            for f in os.scandir(entry.path)
                            if f.is_file()
                        )
                    else:
                        strays += entry.stat().st_size
                except OSError:
                    pass

    ext_dir = env.get("VXPU_EXTERNAL_DIR")
    if ext_dir and os.path.isdir(ext_dir):
        for root, _, files in os.walk(ext_dir):
            for filename in files:
                if ".partial." in filename:
                    try:
                        strays += os.path.getsize(os.path.join(root, filename))
                    except OSError:
                        pass

    blob_dir = env.get("VXPU_BLOB_DIR")
    if blob_dir:
        incoming_dir = os.path.join(blob_dir, ".incoming")
        if os.path.isdir(incoming_dir):
            for entry in os.scandir(incoming_dir):
                try:
                    if entry.is_dir():
                        strays += sum(
                            f.stat().st_size
                            for f in os.scandir(entry.path)
                            if f.is_file()
                        )
                    else:
                        strays += entry.stat().st_size
                except OSError:
                    pass

    aoti_dir = env.get("VXPU_AOTI_CACHE_DIR") or os.path.expanduser("~/.cache/vxpu/aoti")
    if os.path.isdir(aoti_dir):
        for entry in os.scandir(aoti_dir):
            if entry.name.endswith(".tmp.pt2"):
                try:
                    strays += entry.stat().st_size
                except OSError:
                    pass

    return strays


def check_scratch(env=os.environ):
    """Verify free space on scratch directory and report lingering temporary files.

    Catches disk exhaustion from interrupted extractions and large temporary artifacts.
    """
    scratch = env.get("VXPU_SCRATCH_DIR") or (
        "/mnt/ssd" if os.path.isdir("/mnt/ssd") else tempfile.gettempdir()
    )
    if not os.path.exists(scratch):
        return "FAIL", f"scratch directory {scratch} missing"
    usage = shutil.disk_usage(scratch)
    free_gb = usage.free / 1e9
    used_pct = (usage.used / usage.total) * 100
    strays = _scan_strays(env)
    msg = f"{free_gb:.1f} GB free ({used_pct:.1f}% used), {strays / 1e9:.1f} GB temporaries"
    if used_pct >= 95.0 or free_gb < 10.0:
        return "FAIL", msg
    if strays > 5e9 or used_pct >= 90.0:
        return "WARN", msg
    return "PASS", msg


def check_caches(env=os.environ):
    """Verify AOTI cache directory is writable and TorchInductor and Triton caches exist.

    Catches swept or missing compilation caches before running compiled segments.
    """
    aoti = env.get("VXPU_AOTI_CACHE_DIR") or os.path.expanduser("~/.cache/vxpu/aoti")
    if not os.path.isdir(aoti):
        return "FAIL", f"AOTI cache {aoti} missing"
    try:
        probe = os.path.join(aoti, f".probe.{os.getpid()}")
        with open(probe, "w"):
            pass
        os.remove(probe)
    except OSError as e:
        return "FAIL", f"AOTI probe not writable: {e}"

    inductor_dir = env.get("TORCHINDUCTOR_CACHE_DIR")
    if not inductor_dir:
        try:
            from torch._inductor.runtime.cache_dir_utils import default_cache_dir

            inductor_dir = default_cache_dir()
        except Exception:
            import getpass

            inductor_dir = os.path.join(
                tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
            )
    if not os.path.isdir(inductor_dir):
        return "FAIL", f"inductor cache missing ({inductor_dir})"

    triton_dir = env.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    if not os.path.isdir(triton_dir):
        return "FAIL", f"triton cache missing ({triton_dir})"

    return "PASS", "AOTI probe writable, inductor and Triton caches present"


def check_page_cache(manifest, binding=None, env=os.environ, samples=1000):
    """Sample memory residency of external bank and table files using mincore.

    Catches cold node residency (<90%) before load (#504).
    """
    ext_dir = env.get("VXPU_EXTERNAL_DIR")
    if not ext_dir:
        return "PASS", "no bank table files to check"
    target_dir = external_files_dir(ext_dir, manifest)
    names = _external_paths(manifest, binding, env=env)
    paths = [
        os.path.join(target_dir, name)
        for name in names
        if os.path.exists(os.path.join(target_dir, name))
    ]
    if not paths:
        return "PASS", "no bank table files to check"
    try:
        libc = ctypes.CDLL(None)
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    except Exception:
        return "WARN", "mincore unavailable"

    resident = 0
    total = 0
    vec = (ctypes.c_char * 1)()
    for p in paths:
        size = os.path.getsize(p)
        if not size:
            continue
        pages = (size + 4095) // 4096
        sample_count = min(samples, pages)
        with open(p, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mm:
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                stride = pages // sample_count
                for i in range(sample_count):
                    offset = i * stride * 4096
                    if libc.mincore(addr + offset, 4096, vec) == 0:
                        resident += ord(vec[0]) & 1
                total += sample_count

    if not total:
        return "PASS", "no pages sampled"

    pct = (resident / total) * 100
    msg = f"{pct:.1f}% resident across {len(paths)} bank files"
    if pct < 90.0:
        return "WARN", f"cold node, only {msg} (#504)"
    return "PASS", msg


def check_gpus(env=os.environ):
    """Verify GPU availability and device memory via torch.cuda, and check runner locks.

    Catches missing torch environment and colliding timed test runners.
    """
    locks = glob.glob(env.get("VXPU_LOCK_PATTERN", "/root/hero/h15/LOCK-*"))
    lock_msg = f"; held by {locks[0]}" if locks else ""
    try:
        import torch

        if not torch.cuda.is_available():
            return "WARN", f"CUDA unavailable{lock_msg}"
        count = torch.cuda.device_count()
        free_gbs = [f"{torch.cuda.mem_get_info(i)[0] / 1e9:.1f} GB" for i in range(count)]
        status = "WARN" if locks else "PASS"
        return status, f"{count} GPUs ({', '.join(free_gbs)} free){lock_msg}"
    except Exception as e:
        return "FAIL", f"GPU check failed ({e}){lock_msg}"


def run_doctor(artifact_dir=None, env=os.environ, log=print):
    """Run all pre-load diagnostic checks against the given artifact and environment."""
    artifact = artifact_dir or env.get("VXPU_ARTIFACT_DIR") or "."
    manifest_path = (
        os.path.join(artifact, "manifest.json")
        if os.path.isdir(artifact)
        else artifact
    )
    with open(manifest_path) as f:
        manifest = json.load(f)

    binding_path = os.path.join(os.path.dirname(manifest_path), "binding.json")
    binding = None
    if os.path.exists(binding_path):
        with open(binding_path) as f:
            binding = json.load(f)

    checks = [
        ("blob_dir", lambda: check_blob_dir(manifest, env)),
        ("external_dir", lambda: check_external_dir(manifest, binding, env)),
        ("tokenizer", lambda: check_tokenizer(manifest, env)),
        ("environment", lambda: check_environment(env)),
        ("scratch", lambda: check_scratch(env)),
        ("caches", lambda: check_caches(env)),
        ("page_cache", lambda: check_page_cache(manifest, binding, env)),
        ("gpus", lambda: check_gpus(env)),
    ]

    failed = False
    for name, fn in checks:
        try:
            status, msg = fn()
        except plan.ConfigError as e:
            status, msg = "FAIL", str(e)
        log(f"{status:<4}  {name}: {msg}")
        if status == "FAIL":
            failed = True

    return 1 if failed else 0


def main(argv=None):
    """Check pod environment and caches before a load."""
    parser = argparse.ArgumentParser(
        description="Check pod environment and caches before a load"
    )
    parser.add_argument(
        "artifact",
        nargs="?",
        default=None,
        help="artifact directory (default: $VXPU_ARTIFACT_DIR, else .)",
    )
    args = parser.parse_args(argv)
    sys.exit(run_doctor(args.artifact))


if __name__ == "__main__":
    main()
