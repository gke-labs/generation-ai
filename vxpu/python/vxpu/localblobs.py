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


"""The executor machine's local weight tier (#357).

An executor on a machine of its own (a GPU VM, a Mac) keeps weights on
its local disk, one file per checkpoint file named by its sha256: the
machine's persistent cache, which survives executor restarts and serves
the next load of any model sharing those files. The plan's blob_dir
(VXPU_BLOB_DIR) is the directory; blob_bucket (VXPU_BLOB_BUCKET,
gs://BUCKET) is the weight bucket modelstore keeps (gs://BUCKET/_blobs/
<sha256>), from which missing files are copied first (`gcloud storage
cp`, in parallel, at the machine's network speed: ~510 GB in ~5 minutes
in-region). The manifest's sources are then pointed at the local files,
so rehydration reads local disk.

Without a bucket, files not already local keep their manifest sources
(modelstore, through the router's rewrite). A copy lands in a staging
directory of its own under blob_dir/.incoming/ (one per call, so
concurrent loads never share one) and is renamed into place only once its
size matches the manifest, so an interrupted copy is never mistaken for a
complete file. Every complete file is kept even when others failed to
copy, so a retry copies only what is still missing. Staging left by a
process that no longer runs (a killed executor skips the cleanup) is
removed by the next copy, and on Linux the copy itself dies with the
process that started it.
"""

import os
import shutil
import subprocess
import tempfile
import time


def _complete(path, size):
    return os.path.exists(path) and os.path.getsize(path) == size


def localize(files, blob_dir, bucket=None, log=lambda msg: print(msg, flush=True)):
    """Localize checkpoint files into blob_dir.

    The files argument is a manifest's mapping of sha256 to file entries with
    size, name, and source. This function returns a copy whose sources are
    file:// paths under blob_dir for every file held locally, copying any
    missing files from bucket when provided. It also returns a summary with
    local, files, local_files, missing_files, copied, copied_bytes, and
    seconds.
    """
    os.makedirs(blob_dir, exist_ok=True)
    missing = [sha for sha, entry in files.items() if not _complete(os.path.join(blob_dir, sha), entry["size"])]
    copied, copied_bytes, started = 0, 0, time.time()
    if missing and bucket:
        root = os.path.join(blob_dir, ".incoming")
        os.makedirs(root, exist_ok=True)
        _sweep(root)
        incoming = tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=root)
        try:
            urls = "".join(f"{bucket.rstrip('/')}/_blobs/{sha}\n" for sha in missing)
            total = sum(files[sha]["size"] for sha in missing)
            log(f"[vxpu] copying {len(missing)} weight files ({total / 1e9:.1f} GB) from {bucket} to {blob_dir}")
            result = subprocess.run(_with_parent(["gcloud", "storage", "cp", "-I", incoming + "/"]), input=urls,
                                    text=True, capture_output=True)
            # every complete file first: gcloud finishes the other copies before reporting a failed one
            for sha in missing:
                src = os.path.join(incoming, sha)
                if _complete(src, files[sha]["size"]):
                    os.replace(src, os.path.join(blob_dir, sha))
                    copied += 1
                    copied_bytes += files[sha]["size"]
            if result.returncode != 0:
                tail = "\n".join(line for line in result.stderr.splitlines() if not line.startswith("Copying"))[-2000:]
                raise RuntimeError(f"gcloud storage cp from {bucket} failed ({copied} of {len(missing)} files "
                                   f"copied and kept): {tail}")
        finally:
            shutil.rmtree(incoming, ignore_errors=True)
    out, local = {}, 0
    local_files, missing_files = [], []
    for sha, entry in files.items():
        path = os.path.join(blob_dir, sha)
        if _complete(path, entry["size"]):
            out[sha] = {**entry, "source": "file://" + os.path.abspath(path)}
            local += 1
            local_files.append({"name": entry["name"], "size": entry["size"], "sha": sha})
        else:
            out[sha] = dict(entry)
            missing_files.append({"name": entry["name"], "size": entry["size"], "sha": sha})
    summary = {"local": local, "files": len(files), "local_files": local_files,
               "missing_files": missing_files, "copied": copied, "copied_bytes": copied_bytes,
               "seconds": round(time.time() - started, 1)}
    if copied:
        log(f"[vxpu] copied {copied} files ({copied_bytes / 1e9:.1f} GB) in {summary['seconds']} s")
    return out, summary


def release(files, blob_dir, exclude_shas=None):
    """Ask the kernel to drop the page cache of the local blob files once
    the engine has loaded from them: their bytes now live in RAM (the
    expert banks), on the GPU or in the external tables' own files, so
    cached blob pages only push out what the engine reads every token
    (the memory-mapped or file-backed n-gram tables). Tables read directly
    from checkpoint files via io_uring are excluded (#452)."""
    released = 0
    for sha in files:
        if exclude_shas and sha in exclude_shas:
            continue
        path = os.path.join(blob_dir, sha)
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            released += 1
        except (OSError, AttributeError):  # no posix_fadvise (macOS): nothing to release
            pass
        finally:
            os.close(fd)
    return released


def _sweep(root):
    """Remove the staging directories (named <pid>-...) of processes that
    no longer run: a killed executor never ran its cleanup."""
    for name in os.listdir(root):
        pid = name.split("-", 1)[0]
        if pid.isdigit() and int(pid) != os.getpid() and not _alive(int(pid)):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # running, as another user
        return True
    return True


def _with_parent(cmd):
    """cmd, run so that it is sent SIGTERM if this process dies (Linux,
    util-linux's setpriv): a killed executor's copy would otherwise run
    on, orphaned, into a directory nothing reads. preexec_fn could set
    the same flag without the extra program, but is unsafe in a
    process with threads, as the executor is."""
    setpriv = shutil.which("setpriv")
    return [setpriv, "--pdeathsig", "TERM", "--"] + cmd if setpriv else cmd
