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

import collections
import concurrent.futures
import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

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


GAP_BYTES = 1 << 20          # 1 MB gap coalescence
RUN_PART_BYTES = 256 << 20   # split runs > 256 MB into parts for parallelism
MAX_IN_FLIGHT_BYTES = 8 << 30 # 8 GB in-flight buffer memory limit
MAX_IN_FLIGHT_RUNS = 64
CHUNK = 8 << 20


def _read_range(url, offset, length, into):
    """Stream bytes [offset, offset + length) of url into the memoryview
    `into`, in chunks, so nothing is held twice."""
    end = offset + length - 1
    if url.startswith("file://"):  # a local checkpoint directory
        got = 0
        with open(url[len("file://"):], "rb") as f:
            f.seek(offset)
            while got < length:
                n = f.readinto(into[got:])
                if not n:
                    break
                got += n
    else:
        resp = _session().get(url, headers={"Range": f"bytes={offset}-{end}"}, stream=True)
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


class ReadItem:
    """A tensor slice to be read from a file range into a destination tensor."""

    def __init__(self, file_sha256, offset, length, target,
                 cat_dim=None, slot_offset=0, shape=None, dtype=None, cas_path=None):
        self.file_sha256 = file_sha256
        self.offset = offset
        self.length = length
        self.target = target
        self.cat_dim = cat_dim
        self.slot_offset = slot_offset
        self.shape = shape if shape is not None else list(target.shape)
        self.dtype = dtype if dtype is not None else target.dtype
        self.cas_path = cas_path


class _Run:
    """A contiguous or coalesced run of bytes to fetch from one file."""

    def __init__(self, file_sha256, offset, length, items):
        self.file_sha256 = file_sha256
        self.offset = offset
        self.length = length
        self.items = items
        self.buf = None
        self.view = None
        self.buf_bytes = 0


def _can_map_straight(run):
    if len(run.items) != 1:
        return False
    item = run.items[0]
    if item.offset != run.offset or item.length != run.length:
        return False
    if item.cat_dim is not None:
        return False
    if item.target.device.type != "cpu":
        return False
    if not item.target.is_contiguous():
        return False
    return True


def _scatter_run(run):
    if _can_map_straight(run):
        item = run.items[0]
        if item.cas_path is not None:
            os.makedirs(os.path.dirname(item.cas_path), exist_ok=True)
            with open(item.cas_path, "wb") as f:
                f.write(run.view)
        return

    buf = run.buf
    for item in run.items:
        start = item.offset - run.offset
        end = start + item.length
        slice_bytes = buf[start:end]
        part = slice_bytes.view(item.dtype).view(item.shape)
        if item.cat_dim is None:
            item.target.copy_(part)
        else:
            item.target.narrow(item.cat_dim, item.slot_offset, part.shape[item.cat_dim]).copy_(part)
        if item.cas_path is not None:
            os.makedirs(os.path.dirname(item.cas_path), exist_ok=True)
            with open(item.cas_path, "wb") as f:
                f.write(memoryview(slice_bytes.contiguous().numpy()))


def plan_and_fetch_ranges(ranges, files, cas_dir=None, workers=16,
                          gap_bytes=GAP_BYTES, part_bytes=RUN_PART_BYTES,
                          max_in_flight_bytes=MAX_IN_FLIGHT_BYTES):
    """Plan, coalesce, fetch and scatter ranges into destination tensors.

    Merges ranges that touch or sit within `gap_bytes` (1 MB) into runs,
    splits runs larger than `part_bytes` (256 MB) into parts for parallelism,
    fetches the parts concurrently into one buffer per run (or straight into
    the destination tensor when a run maps to one), and scatters into
    destination tensors with narrow().copy_() (or copy_()). Preserves cas_dir
    behaviour.

    Returns (bytes_downloaded, requests_issued).
    """
    needed_items = []
    for item in ranges:
        if cas_dir:
            key = hashlib.sha256(f"{item.file_sha256}:{item.offset}:{item.length}".encode()).hexdigest()
            path = os.path.join(cas_dir, key)
            if os.path.exists(path):
                if item.cat_dim is None:
                    view = memoryview(item.target.contiguous().reshape(-1).view(torch.uint8).numpy())
                    with open(path, "rb") as f:
                        if f.readinto(view) != item.length:
                            raise RuntimeError(f"cache entry {path} is short")
                else:
                    part = torch.empty(item.shape, dtype=item.dtype)
                    view = memoryview(part.contiguous().reshape(-1).view(torch.uint8).numpy())
                    with open(path, "rb") as f:
                        if f.readinto(view) != item.length:
                            raise RuntimeError(f"cache entry {path} is short")
                    item.target.narrow(item.cat_dim, item.slot_offset, part.shape[item.cat_dim]).copy_(part)
                continue
            item.cas_path = path
        needed_items.append(item)

    if not needed_items:
        return 0, 0

    items_by_file = collections.defaultdict(list)
    for it in needed_items:
        items_by_file[it.file_sha256].append(it)

    runs = []
    for file_sha256, file_items in items_by_file.items():
        file_items.sort(key=lambda it: it.offset)
        cur_run = None
        for it in file_items:
            if cur_run is None:
                cur_run = _Run(file_sha256, it.offset, it.length, [it])
            else:
                cur_end = cur_run.offset + cur_run.length
                if it.offset <= cur_end + gap_bytes:
                    new_end = max(cur_end, it.offset + it.length)
                    cur_run.length = new_end - cur_run.offset
                    cur_run.items.append(it)
                else:
                    runs.append(cur_run)
                    cur_run = _Run(file_sha256, it.offset, it.length, [it])
        if cur_run is not None:
            runs.append(cur_run)

    run_queue = collections.deque(runs)
    active_runs = {}
    future_to_info = {}
    in_flight_bytes = 0
    total_downloaded = 0
    requests_issued = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while run_queue or future_to_info:
            while run_queue and (in_flight_bytes + (0 if _can_map_straight(run_queue[0]) else run_queue[0].length) <= max_in_flight_bytes or len(active_runs) == 0) and len(active_runs) < MAX_IN_FLIGHT_RUNS:
                run = run_queue.popleft()
                url = files[run.file_sha256]["source"]
                if _can_map_straight(run):
                    run.buf = None
                    run.view = memoryview(run.items[0].target.contiguous().reshape(-1).view(torch.uint8).numpy())
                    run.buf_bytes = 0
                else:
                    run.buf = torch.empty(run.length, dtype=torch.uint8)
                    run.view = memoryview(run.buf.numpy())
                    run.buf_bytes = run.length
                in_flight_bytes += run.buf_bytes

                n_parts = 0
                for p_off in range(0, run.length, part_bytes):
                    p_len = min(part_bytes, run.length - p_off)
                    fut = pool.submit(_read_range, url, run.offset + p_off, p_len, run.view[p_off : p_off + p_len])
                    future_to_info[fut] = (run, p_len)
                    requests_issued += 1
                    total_downloaded += p_len
                    n_parts += 1
                active_runs[run] = n_parts

            if future_to_info:
                done, _ = concurrent.futures.wait(list(future_to_info.keys()), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    fut.result()
                    run, p_len = future_to_info.pop(fut)
                    active_runs[run] -= 1
                    if active_runs[run] == 0:
                        del active_runs[run]
                        _scatter_run(run)
                        run.buf = None
                        run.view = None
                        in_flight_bytes -= run.buf_bytes

    return total_downloaded, requests_issued


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
    length = ref["length"]
    dtype = DTYPES[ref["dtype"]]
    shape = ref["shape"]
    if mmap_dir:
        key = hashlib.sha256(f"{ref['file_sha256']}:{ref['offset']}:{length}".encode()).hexdigest()
        buf, fresh = file_backed(os.path.join(mmap_dir, key), length)
        tensor = buf.view(dtype).view(shape)
        if not fresh:
            return tensor, 0
    else:
        tensor = torch.empty(shape, dtype=dtype)
    item = ReadItem(
        file_sha256=ref["file_sha256"],
        offset=ref["offset"],
        length=length,
        target=tensor,
        cat_dim=None,
        shape=shape,
        dtype=dtype,
    )
    downloaded, _ = plan_and_fetch_ranges([item], files, cas_dir=cas_dir)
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

    items = []
    for r, row in enumerate(rows):
        target_row = out[r] if stack else out
        offset = 0
        for n in row:
            ref = manifest_tensors[n]
            part_shape = ref["shape"]
            part_dtype = DTYPES[ref["dtype"]]
            items.append(ReadItem(
                file_sha256=ref["file_sha256"],
                offset=ref["offset"],
                length=ref["length"],
                target=target_row,
                cat_dim=cat_dim,
                slot_offset=offset,
                shape=part_shape,
                dtype=part_dtype,
            ))
            offset += part_shape[cat_dim]

    downloaded, _ = plan_and_fetch_ranges(items, files, cas_dir=cas_dir, workers=workers)
    return out, downloaded


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
