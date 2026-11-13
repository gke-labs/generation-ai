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

"""Rehydrate a vXPU artifact into runnable modules.

bound tensors are fetched by HTTP range request, one per tensor,
streaming straight to the target device (flat memory profile; warm
reloads come from the server's digest keep-alive). derived
tensors are recomputed from config using transformers' own formula
registry (exact RoPE for every rope_type, including per-layer-type
hybrids). state tensors are zero-initialized per session.

The graphs were traced on the meta device; after placement they are
retargeted to the executor's device, and device-specific compat passes
are applied (e.g. histc(int) has no CPU kernel; for expert counting it
is exactly bincount).
"""

import email.utils
import fcntl
import hashlib
import math
import os
import random
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from torch.export.passes import move_to_device_pass

from .localblobs import tensor_key

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}


_local = threading.local()


def _session():
    """One keep-alive HTTP session per thread.

    A model is tens of thousands of range reads; a new connection per
    read exhausts the ephemeral ports (errno 99) on a fast node.
    """
    session = getattr(_local, "session", None)
    if session is None:
        session = _local.session = requests.Session()
        retries = requests.adapters.Retry(total=3, respect_retry_after_header=False)
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


# A tensor larger than this is read as several concurrent ranges into
# one buffer (a 98 GB embedding table is one checkpoint tensor).
PART_BYTES = 1 << 30
PART_WORKERS = 8
CHUNK = 8 << 20

RETRY_STATUSES = {408, 425, 429, 499, 500, 502, 503, 504}
MAX_WAIT = 120.0
BACKOFF_BASE = 0.5
REQUEST_TIMEOUT = 60.0


def _parse_retry_after(value):
    """Parse a Retry-After header into seconds to wait, or None."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _get_range(url, offset, end, stop_event=None, log=lambda msg: print(msg, flush=True),
               max_wait=MAX_WAIT, backoff_base=BACKOFF_BASE, timeout=REQUEST_TIMEOUT):
    """Fetch a byte range [offset, end] with retry on transient HTTP statuses.

    Returns the response object with status 206, or raises an exception.
    The caller is responsible for closing the returned response.
    """
    deadline = time.monotonic() + max_wait
    attempts = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("load stopped")
        attempts += 1
        resp = _session().get(url, headers={"Range": f"bytes={offset}-{end}"},
                              stream=True, timeout=timeout)
        if resp.status_code in RETRY_STATUSES:
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            backoff = min(backoff_base * (2 ** (attempts - 1)), 30.0) * (0.5 + random.random() * 0.5)
            wait = max(retry_after, backoff) if retry_after is not None else backoff
            if time.monotonic() + wait > deadline:
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as err:
                    raise type(err)(f"{err} (after {attempts} attempts)", response=err.response) from err
                finally:
                    resp.close()
            resp.close()
            log(f"[vxpu] retry HTTP {resp.status_code} for {url} offset {offset}: waiting {wait:.2f}s")
            if stop_event is not None and stop_event.wait(wait):
                raise RuntimeError("load stopped")
            elif stop_event is None:
                time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(f"expected HTTP 206 Partial Content, got {resp.status_code}")
            return resp
        except Exception:
            resp.close()
            raise


def _read_range(url, offset, length, into, stop_event=None, log=lambda msg: print(msg, flush=True),
                max_wait=None, backoff_base=None):
    """Stream bytes [offset, offset + length) of url into the memoryview
    `into`, in chunks, so nothing is held twice. Retries transient HTTP
    statuses (408, 425, 429, 499, 500, 502, 503, 504) with backoff and jitter."""
    end = offset + length - 1
    if url.startswith("file://"):  # a local checkpoint directory
        with open(url[len("file://"):], "rb") as f:
            f.seek(offset)
            got = f.readinto(into)
    else:
        max_wait = MAX_WAIT if max_wait is None else max_wait
        backoff_base = BACKOFF_BASE if backoff_base is None else backoff_base
        resp = _get_range(url, offset, end, stop_event=stop_event, log=log,
                          max_wait=max_wait, backoff_base=backoff_base)
        try:
            got = 0
            for chunk in resp.iter_content(CHUNK):
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("load stopped")
                n = len(chunk)
                if got + n > length:
                    raise RuntimeError(f"server sent more than {length} bytes")
                into[got:got + n] = chunk
                got += n
        finally:
            resp.close()
    if got != length:
        raise RuntimeError(f"expected {length} bytes, got {got} bytes")


def advise_random(tensor):
    """madvise(MADV_RANDOM) over a memory-mapped tensor's pages: a fault
    reads just its page, with no readahead. For tables read row by row at
    random (the n-gram tables, #364), readahead fetches up to 128 KiB per
    few-hundred-byte row and evicts other rows to hold it. Returns whether
    the advice was taken (not on platforms without madvise)."""
    import ctypes
    import mmap
    MADV_RANDOM = getattr(mmap, "MADV_RANDOM", None)  # the platform's value, where Python knows one
    if MADV_RANDOM is None:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        madvise = libc.madvise
    except (OSError, AttributeError):
        return False
    storage = tensor.untyped_storage()
    addr, length = storage.data_ptr(), storage.nbytes()
    start = addr - addr % mmap.PAGESIZE
    madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    return madvise(start, length + (addr - start), MADV_RANDOM) == 0


def _clean_strays(dirname, basename):
    """Remove unlocked temporary files left behind by interrupted loads."""
    try:
        entries = os.listdir(dirname)
    except OSError:
        return
    prefix = basename + ".partial."
    for name in entries:
        if name.startswith(prefix):
            file_path = os.path.join(dirname, name)
            try:
                sfd = os.open(file_path, os.O_RDWR)
                try:
                    fcntl.flock(sfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.remove(file_path)
                except (BlockingIOError, OSError):
                    pass
                finally:
                    os.close(sfd)
            except OSError:
                pass


def file_backed(path, length):
    """A uint8 tensor of `length` bytes whose storage is the file at
    `path`, memory-mapped shared: writes land in the page cache and reach
    the file, reads come from the page cache or the disk. A tensor placed
    this way costs no committed memory; the kernel keeps the hot pages
    resident and evicts the rest, which is the RAM-over-SSD tier of
    hierarchical placement. A complete file of the right size is reused.

    Returns (tensor, fresh, commit). If a complete file of the right size
    already exists at path, file_backed maps it and returns fresh=False.
    Otherwise it creates a temporary file next to path, locked so that no
    other process deletes it, and returns fresh=True. Fill the tensor,
    then call commit(), which flushes the data and renames the temporary
    file to path.
    """
    dirname = os.path.dirname(path) or "."
    basename = os.path.basename(path)
    os.makedirs(dirname, exist_ok=True)
    fresh = not (os.path.exists(path) and os.path.getsize(path) == length)
    if not fresh:
        _clean_strays(dirname, basename)
        return torch.from_file(path, shared=True, size=length, dtype=torch.uint8), False, lambda: None

    _clean_strays(dirname, basename)
    fd, tmp = tempfile.mkstemp(dir=dirname, prefix=basename + ".partial.")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(fd, length)
    tensor = torch.from_file(tmp, shared=True, size=length, dtype=torch.uint8)
    fin = weakref.finalize(tensor, os.close, fd)

    def commit():
        os.fsync(fd)
        try:
            os.replace(tmp, path)
        finally:
            fin()
        _clean_strays(dirname, basename)

    return tensor, True, commit


def fetch_tensor(ref, files, cas_dir=None, mmap_dir=None, stop_event=None):
    """One tensor, read by range into a buffer allocated once.

    Large tensors are read as several concurrent ranges of the same
    buffer. With cas_dir set, bytes are also kept on disk under a
    content-derived key (for persistent volumes); warm reloads otherwise
    come from the engine keep-alive (digest reuse). With mmap_dir set,
    the buffer itself is a file there (see file_backed): the tensor is
    memory-mapped rather than held, and a file already complete from an
    earlier load is used without fetching.

    Returns (tensor, bytes_downloaded); 0 when served from the cache.
    """
    url = files[ref["file_sha256"]]["source"]
    length = ref["length"]
    key = tensor_key(ref)
    path = os.path.join(cas_dir, key) if cas_dir else None
    commit = lambda: None
    if mmap_dir:
        buf, fresh, commit = file_backed(os.path.join(mmap_dir, key), length)
        if not fresh:
            return buf.view(DTYPES[ref["dtype"]]).view(ref["shape"]), 0
    else:
        buf = torch.empty(length, dtype=torch.uint8)
    view = memoryview(buf.numpy())
    downloaded = 0
    if path is not None and os.path.exists(path):
        with open(path, "rb") as f:
            if f.readinto(view) != length:
                raise RuntimeError(f"cache entry {path} is short")
    else:
        parts = max(1, min(PART_WORKERS, -(-length // PART_BYTES)))
        if parts == 1:
            _read_range(url, ref["offset"], length, view, stop_event=stop_event)
        else:
            step = -(-length // parts)
            with ThreadPoolExecutor(max_workers=parts) as pool:
                futures = [pool.submit(_read_range, url, ref["offset"] + i * step,
                                       min(step, length - i * step), view[i * step:min(length, (i + 1) * step)],
                                       stop_event=stop_event)
                           for i in range(parts) if i * step < length]
                for f in futures:
                    f.result()
        downloaded = length
        if path is not None:
            with open(path, "wb") as f:
                f.write(view)
    commit()
    tensor = buf.view(DTYPES[ref["dtype"]]).view(ref["shape"])
    return tensor, downloaded


def fetch_stacked(spec, manifest_tensors, files, cas_dir=None, workers=16):
    """Assemble a parameter from several checkpoint tensors.

    ``spec["parts"]`` is a list of rows; each row is a list of manifest
    tensor names concatenated along ``spec["cat_dim"]`` (a gated MLP
    expert's gate and up projections become one block). With
    ``spec["stack"]`` (the default) the rows are stacked along a new
    leading dimension (an expert bank); otherwise there is one row (a
    table stored in shards). The result is allocated once and filled
    part by part, so assembling a 50 GB table costs 50 GB, not twice
    that. Parts are fetched concurrently: a 128-expert layer is 384
    range reads, and a model has dozens of layers.

    Returns (tensor, bytes_downloaded).
    """
    rows, cat_dim, stack = spec["parts"], spec["cat_dim"], spec.get("stack", True)
    # Shapes and dtype come from the manifest, so the result can be
    # allocated before anything is fetched.
    first = manifest_tensors[rows[0][0]]
    row_shape = list(first["shape"])
    row_shape[cat_dim] = sum(manifest_tensors[n]["shape"][cat_dim] for n in rows[0])
    out_shape = ([len(rows)] if stack else []) + row_shape
    dtype = DTYPES[first["dtype"]]
    out = torch.empty(out_shape, dtype=dtype)

    # (row, offset, name) in assembly order; fetched through a bounded
    # window so at most `workers` parts are in flight (a sharded 50 GB
    # table must not be held twice).
    slots = []
    for r, row in enumerate(rows):
        offset = 0
        for n in row:
            slots.append((r, offset, n))
            offset += manifest_tensors[n]["shape"][cat_dim]
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for i, (r, offset, n) in enumerate(slots):
            pending[i] = pool.submit(fetch_tensor, manifest_tensors[n], files, cas_dir)
            if len(pending) >= 2 * workers:
                j = min(pending)
                _place_part(out, slots[j], pending.pop(j).result(), cat_dim, stack)
                downloaded += _last_bytes
        for j in sorted(pending):
            _place_part(out, slots[j], pending.pop(j).result(), cat_dim, stack)
            downloaded += _last_bytes
    return out, downloaded


_last_bytes = 0


def _place_part(out, slot, fetched, cat_dim, stack):
    global _last_bytes
    r, offset, _ = slot
    part, _last_bytes = fetched
    target = out[r] if stack else out
    target.narrow(cat_dim, offset, part.shape[cat_dim]).copy_(part)


def derived_tensor(fqn, meta, config):
    """Recompute a config-derived tensor to the shape the graph expects.

    RoPE tables use transformers' own ROPE_INIT_FUNCTIONS registry,
    keyed by rope_type and (for hybrid-attention models) layer type —
    exact by construction. Embedding scales are sqrt(hidden).
    """
    text_config = (config.get_text_config()
                   if hasattr(config, "get_text_config") else config)
    shape = tuple(meta.shape)
    dtype = (meta.dtype if meta.dtype.is_floating_point
             else torch.float32)

    if "embed_scale" in fqn:
        hidden = text_config.hidden_size
        if "per_layer" in fqn and getattr(
                text_config, "hidden_size_per_layer_input", 0):
            hidden = text_config.hidden_size_per_layer_input
        return torch.full(shape or (1,), math.sqrt(hidden),
                          dtype=dtype).reshape(shape)

    # Mirror transformers' own rope init (modeling_rope_utils + the
    # per-model rotary modules): select the layer's rope_type, call the
    # registered init fn, and pass the exact kwargs transformers passes
    # (e.g. head_dim_key="global_head_dim" for Gemma 4 proportional
    # full-attention RoPE). Reproducing this rather than re-deriving
    # formulas keeps hybrid/partial-rotary models exact.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    layer_type = None
    for candidate in ("sliding_attention", "full_attention"):
        if candidate in fqn:
            layer_type = candidate
    rope_params = getattr(text_config, "rope_parameters", None) or {}
    if layer_type and isinstance(rope_params.get(layer_type), dict):
        params = rope_params[layer_type]
    else:
        params = rope_params if isinstance(rope_params, dict) else {}
    rope_type = params.get("rope_type", "default")

    kwargs = {}
    if layer_type:
        kwargs["layer_type"] = layer_type
        if layer_type == "full_attention" and rope_type == "proportional":
            kwargs["head_dim_key"] = "global_head_dim"
    init_fn = ROPE_INIT_FUNCTIONS.get(rope_type)
    if init_fn is not None:
        inv_freq, _ = init_fn(text_config, **kwargs)
        want = shape[-1] if shape else inv_freq.numel()
        if inv_freq.numel() == want:
            return inv_freq.to(dtype).reshape(shape)
    # Fallback: plain geometric series over the dims the graph expects.
    length = shape[-1] if shape else 32
    theta = params.get("rope_theta",
                       getattr(text_config, "rope_theta", 10000.0))
    exponent = torch.arange(0, length, dtype=torch.float32) / max(length, 1)
    return (1.0 / (theta ** exponent)).to(dtype).reshape(shape)


def cpu_compat(exported):
    """Rewrite ops whose CPU kernels lack integer support.

    histc(int) has no CPU kernel; for integer inputs counted over
    [0, bins) it is exactly bincount (sliced to bins, cast back).
    Graphs from the forked transformers count in float off the GPU and
    need no rewrite; this remains for artifacts exported before that.
    """
    graph = exported.graph_module.graph
    replaced = 0
    for node in list(graph.nodes):
        if (node.op == "call_function"
                and str(node.target) == "aten.histc.default"):
            x, bins = node.args[0], node.args[1]
            val = x.meta.get("val") if hasattr(x, "meta") else None
            if val is not None and val.is_floating_point():
                continue  # float histc has a CPU kernel; leave the graph alone
            with graph.inserting_before(node):
                flat = graph.call_function(
                    torch.ops.aten.reshape.default, (x, [-1]))
                cast = graph.call_function(
                    torch.ops.aten._to_copy.default, (flat,),
                    {"dtype": torch.int64})
                counts = graph.call_function(
                    torch.ops.aten.bincount.default, (cast,),
                    {"minlength": bins})
                sliced = graph.call_function(
                    torch.ops.aten.slice.Tensor, (counts, 0, 0, bins))
                out = graph.call_function(
                    torch.ops.aten._to_copy.default, (sliced,),
                    {"dtype": torch.int32})
            node.replace_all_uses_with(out)
            graph.erase_node(node)
            replaced += 1
    exported.graph_module.recompile()
    return replaced


def strip_asserts(module):
    """Remove _assert_tensor_metadata guard nodes.

    They are eager-mode sanity checks Dynamo cannot re-trace when
    compiling on CUDA, and ep.module() regenerates some in submodules —
    strip every graph in the tree.
    """
    removed = 0
    for _, sub in module.named_modules():
        if not hasattr(sub, "graph"):
            continue
        for node in list(sub.graph.nodes):
            if (node.op == "call_function"
                    and "_assert_tensor_metadata" in str(node.target)):
                sub.graph.erase_node(node)
                removed += 1
        sub.recompile()
    return removed


def load_program(graph_path, tensors, device):
    """Load an exported program or reuse an existing one. Place real tensors into it, and retarget them to the device."""
    if isinstance(graph_path, torch.export.ExportedProgram):
        exported = graph_path
    else:
        exported = torch.export.load(graph_path)

    def place(fqn, tensor):
        # The checkpoint's dtype may differ from the graph's parameter
        # dtype (Qwen stores fp8 block scales as bf16; transformers'
        # fp8 modules declare float32). Cast as from_pretrained would.
        expected = exported.state_dict.get(fqn, exported.constants.get(fqn))
        bytes_like = (torch.int8, torch.uint8)
        if (isinstance(expected, torch.Tensor) and expected.dtype != tensor.dtype
                and expected.dtype in bytes_like and tensor.dtype in bytes_like):
            # Packed fp4 weights: the checkpoint says int8, the graph
            # keeps them as uint8 bytes. Same bytes, no cast.
            tensor = tensor.view(expected.dtype)
        elif (isinstance(expected, torch.Tensor)
                and expected.dtype != tensor.dtype
                and expected.is_floating_point()
                and tensor.is_floating_point()):
            tensor = tensor.to(expected.dtype)
        if fqn in exported.state_dict:
            exported.state_dict[fqn] = torch.nn.Parameter(
                tensor, requires_grad=False)
        elif fqn in exported.constants:
            exported.constants[fqn] = tensor
        # A graph need not lift every artifact tensor (prefill vs
        # decode differ slightly); extras are simply unused here.

    for fqn, tensor in tensors.items():
        place(fqn, tensor)

    # Example inputs were recorded on meta; materialize placeholders so
    # the device pass can walk them.
    ex_args, ex_kwargs = exported.example_inputs
    fix = lambda t: (torch.zeros(t.shape, dtype=t.dtype)  # noqa: E731
                     if isinstance(t, torch.Tensor) and t.is_meta else t)
    exported._example_inputs = (
        tuple(fix(t) for t in ex_args),
        {k: fix(v) for k, v in ex_kwargs.items()})

    exported = move_to_device_pass(exported, device)
    if device == "cpu":
        cpu_compat(exported)
    return exported


def share_state(src_module, dst_module, state_fqns):
    """Alias state buffers so prefill and decode share one cache.

    move_to_device_pass copies tensors per program; re-point the decode
    module's state buffers at the prefill module's tensors.
    """
    src = dict(src_module.named_buffers())
    shared = 0
    for fqn in state_fqns:
        if fqn in src:
            parent, name = (fqn.rsplit(".", 1)
                            if "." in fqn else ("", fqn))
            dst_parent = (dst_module.get_submodule(parent)
                          if parent else dst_module)
            if name in dst_parent._buffers:
                dst_parent._buffers[name] = src[fqn]
                shared += 1
    return shared
