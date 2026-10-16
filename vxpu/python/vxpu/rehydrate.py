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

import hashlib
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import requests
import torch
from torch.export.passes import move_to_device_pass

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
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


# In-process cache for direct GCS signed URLs, resolved once per file.
_direct_url_cache = {}
_direct_url_lock = threading.Lock()
_url_locks = {}
_locks_lock = threading.Lock()


def _get_url_lock(url):
    with _locks_lock:
        if url not in _url_locks:
            _url_locks[url] = threading.Lock()
        return _url_locks[url]


class _CachedDirectURL:
    def __init__(self, url, expires_at=None):
        self.url = url
        self.expires_at = expires_at

    def is_expired(self, now=None):
        if self.expires_at is None:
            return False
        if now is None:
            now = time.time()
        return now >= self.expires_at


def _parse_url_expiry(url):
    """Extract expiry timestamp from GCS V4 pre-signed URL query parameters."""
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "X-Goog-Date" in qs and "X-Goog-Expires" in qs:
            date_str = qs["X-Goog-Date"][0]
            expires_sec = int(qs["X-Goog-Expires"][0])
            dt = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return dt.timestamp() + expires_sec
    except Exception:
        pass
    return None


def _ask_direct_url(source_url):
    """Ask modelstore for a direct GCS signed URL via HEAD or GET ?redirect=1."""
    sess = _session()
    # 1. Try HEAD request on source_url (the existing hook returning X-Modelstore-Direct-Url)
    try:
        resp = sess.head(source_url)
        if resp.status_code == 200:
            direct_url = resp.headers.get("X-Modelstore-Direct-Url")
            if direct_url:
                return direct_url
            return source_url
        if resp.status_code == 307:
            direct_url = resp.headers.get("Location") or resp.headers.get("X-Modelstore-Direct-Url")
            if direct_url:
                return direct_url
    except Exception:
        pass

    # 2. Try GET with ?redirect=1 (answering 307 to the signed URL)
    try:
        sep = "&" if "?" in source_url else "?"
        resp = sess.get(f"{source_url}{sep}redirect=1", allow_redirects=False, stream=True)
        resp.close()
        if resp.status_code == 307 and "Location" in resp.headers:
            return resp.headers["Location"]
        if "X-Modelstore-Direct-Url" in resp.headers:
            return resp.headers["X-Modelstore-Direct-Url"]
    except Exception:
        pass

    # No direct URL (e.g. local blobstore or no signer configured); read through source_url
    return source_url


def _resolve_url(source_url, force_refresh=False):
    """Return the cached direct URL for source_url, refreshing if expired or missing."""
    if not source_url.startswith(("http://", "https://")):
        return source_url
    with _direct_url_lock:
        if not force_refresh and source_url in _direct_url_cache:
            entry = _direct_url_cache[source_url]
            if not entry.is_expired():
                return entry.url

    with _get_url_lock(source_url):
        with _direct_url_lock:
            if not force_refresh and source_url in _direct_url_cache:
                entry = _direct_url_cache[source_url]
                if not entry.is_expired():
                    return entry.url
            _direct_url_cache.pop(source_url, None)

        direct_url = _ask_direct_url(source_url)
        expires_at = _parse_url_expiry(direct_url) if direct_url != source_url else None
        entry = _CachedDirectURL(direct_url, expires_at)
        with _direct_url_lock:
            _direct_url_cache[source_url] = entry
        return direct_url


def clear_url_cache():
    """Clear in-process cached direct URLs (useful for tests)."""
    with _direct_url_lock:
        _direct_url_cache.clear()
    with _locks_lock:
        _url_locks.clear()


# A tensor larger than this is read as several concurrent ranges into
# one buffer (a 98 GB embedding table is one checkpoint tensor).
PART_BYTES = 1 << 30
PART_WORKERS = 8
CHUNK = 8 << 20


def _read_range(url, offset, length, into):
    """Stream bytes [offset, offset + length) of url into the memoryview
    `into`, in chunks, so nothing is held twice. Asks for direct URLs once per
    file, caches in-process, and retries on 403/expired."""
    end = offset + length - 1
    if url.startswith("file://"):  # a local checkpoint directory
        with open(url[len("file://"):], "rb") as f:
            f.seek(offset)
            got = f.readinto(into)
    else:
        for attempt in range(2):
            fetch_url = _resolve_url(url, force_refresh=(attempt > 0))
            resp = _session().get(fetch_url, headers={"Range": f"bytes={offset}-{end}"}, stream=True)
            if resp.status_code == 403 and attempt == 0 and fetch_url != url:
                resp.close()
                continue
            resp.raise_for_status()
            if resp.status_code != 206:
                raise RuntimeError(f"expected HTTP 206 Partial Content, got {resp.status_code}")
            got = 0
            for chunk in resp.iter_content(CHUNK):
                n = len(chunk)
                if got + n > length:
                    raise RuntimeError(f"server sent more than {length} bytes")
                into[got:got + n] = chunk
                got += n
            break
    if got != length:
        raise RuntimeError(f"expected {length} bytes, got {got} bytes")


def file_backed(path, length):
    """A uint8 tensor of `length` bytes whose storage is the file at
    `path`, memory-mapped shared: writes land in the page cache and reach
    the file, reads come from the page cache or the disk. A tensor placed
    this way costs no committed memory; the kernel keeps the hot pages
    resident and evicts the rest, which is the RAM-over-SSD tier of
    hierarchical placement. A complete file of the right size is reused."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fresh = not (os.path.exists(path) and os.path.getsize(path) == length)
    if fresh:
        with open(path, "wb") as f:
            f.truncate(length)
    return torch.from_file(path, shared=True, size=length, dtype=torch.uint8), fresh


def fetch_tensor(ref, files, cas_dir=None, mmap_dir=None):
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
    key = hashlib.sha256(f"{ref['file_sha256']}:{ref['offset']}:{length}".encode()).hexdigest()
    path = os.path.join(cas_dir, key) if cas_dir else None
    if mmap_dir:
        buf, fresh = file_backed(os.path.join(mmap_dir, key), length)
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
            _read_range(url, ref["offset"], length, view)
        else:
            step = -(-length // parts)
            with ThreadPoolExecutor(max_workers=parts) as pool:
                futures = [pool.submit(_read_range, url, ref["offset"] + i * step,
                                       min(step, length - i * step), view[i * step:min(length, (i + 1) * step)])
                           for i in range(parts) if i * step < length]
                for f in futures:
                    f.result()
        downloaded = length
        if path is not None:
            with open(path, "wb") as f:
                f.write(view)
    tensor = buf.view(DTYPES[ref["dtype"]]).view(ref["shape"])
    return tensor, downloaded


def fetch_stacked(spec, manifest_tensors, files, cas_dir=None, workers=16, mmap_path=None):
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
    if mmap_path:
        # the bank lives in a file (see file_backed): assembled once,
        # memory-mapped on every load after
        nbytes = torch.empty(0, dtype=dtype).element_size()
        for d in out_shape:
            nbytes *= d
        buf, fresh = file_backed(mmap_path, nbytes)
        out = buf.view(dtype).view(out_shape)
        if not fresh:
            return out, 0
    else:
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
    """Load a .pt2, place real tensors into it, retarget to device."""
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
