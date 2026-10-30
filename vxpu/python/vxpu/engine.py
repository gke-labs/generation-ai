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

"""Serving engine: shared weights and compiled graphs, per-session KV.

The binding's three categories map directly onto serving:

    bound   -> fetched once, shared read-only across sessions
    derived -> computed once from config, shared
    state   -> allocated fresh per session; the graphs mutate it in
               place, so a session IS its state tensors

The two graphs (prefill, decode) are built and compiled exactly once
per model. Compilation is the expensive step (tens of seconds), and the
decode graph is identical across sessions — only the state buffers it
reads and mutates differ. So a session owns just its state tensors, and
each turn binds them into the shared modules (a reference swap the
compiled graph re-reads at call time — no recompile). One GPU serves
turns serially anyway, so a lock around bind+run costs nothing real.

Multi-turn conversation costs only the new tokens: the prefill graph
has a dynamic sequence dimension and explicit cache positions, so a
follow-up turn prefills only the tokens beyond the cached prefix.
"""

import collections
import concurrent.futures
import contextlib
import hashlib
import json
import os
import queue
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import torch

from typing import NamedTuple

from . import kernels, lower, ops, segments, vxpu_pb2  # noqa: F401  (registers vxpu::* before a graph that calls them is loaded)
from .plan import Plan
from .plan import record as plan_record
from .telemetry import Recorder, segment_info
from .rehydrate import (DTYPES, advise_random, derived_tensor, fetch_stacked, fetch_tensor,
                        load_program, share_state, strip_asserts)


class FileTable:
    """An n-gram table read straight from checkpoint files via io_uring (#452)."""

    def __init__(self, fd, sha, weight_offset, weight_row_bytes,
                 scale_offset, scale_row_bytes, rows, weight_dtype, scale_dtype):
        self.fd = fd
        self.sha = sha
        self.weight_offset = weight_offset
        self.weight_row_bytes = weight_row_bytes
        self.scale_offset = scale_offset
        self.scale_row_bytes = scale_row_bytes
        self.rows = rows
        self.weight_dtype = weight_dtype
        self.scale_dtype = scale_dtype


def configure_inductor(plan):
    """Set Inductor settings the executor relies on under this plan.

    The artifact carries reference semantics (an fp8 linear is cast,
    scale, matmul); the engine's job is to make that fast here. On CUDA,
    inductor's prologue fusion folds the dequantization into the GEMM's
    operand load, so fp8 weights are read once at fp8 width instead of
    being materialized in bf16 every step: measured on an L4, that takes
    an 8192x8192 fp8 linear from 2.6x slower than bf16 to 1.9x faster,
    at the fp8 bandwidth floor. max-autotune is what enables the GEMM
    templates that fusion needs; it costs compile time once per shape.
    The mode itself is a choice of the engine's plan (plan.Plan:
    compile_mode, which VXPU_COMPILE_MODE overrides).
    """
    if plan["prologue_fusion"]:
        import torch._inductor.config as inductor_config
        inductor_config.prologue_fusion = True


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "tokenizer.model", "vocab.json", "spiece.model")


def external_files_dir(external_dir, manifest):
    """Where an artifact's memory-mapped external tensors live under the
    plan's external_dir (None: none). The engine and a node server use
    the same. The files are keyed by the model's manifest, so two
    artifacts of one checkpoint share them and another checkpoint cannot."""
    if not external_dir:
        return None
    return os.path.join(external_dir, hashlib.sha256(json.dumps(
        manifest["source"], sort_keys=True).encode()).hexdigest()[:16])


def fetch_engram_tensors(manifest, specs_by_layer, cas_dir, mmap_root, layers=None, stop_event=None):
    """Fetch the raw engram table tensors for layers, and return them as
    {layer: (weight, scale)} with the bytes fetched. Stop early, before
    the next tensor, if stop_event is set."""
    out, fetched = {}, 0
    for key, spec in specs_by_layer.items():
        if stop_event is not None and stop_event.is_set():
            return None, fetched
        layer = int(key)
        if layers is not None and layer not in layers:
            continue
        weight, n = fetch_tensor(manifest["tensors"][spec["weight"]], manifest["files"], cas_dir, mmap_dir=mmap_root)
        fetched += n
        if stop_event is not None and stop_event.is_set():
            return None, fetched
        scale, n = fetch_tensor(manifest["tensors"][spec["scale"]], manifest["files"], cas_dir, mmap_dir=mmap_root)
        fetched += n
        if mmap_root:
            advise_random(weight)
            advise_random(scale)
        out[layer] = (weight, scale)
    return out, fetched


def load_engram_tables(manifest, specs_by_layer, cas_dir, mmap_root, layers=None):
    """The n-gram tables of an artifact's layers (binding external.engram;
    `layers` limits them), each as the model's own embedding module on the
    host: ({layer: module}, bytes fetched). Memory-mapped from files under
    mmap_root when set, read row by row at random (no readahead, #364).
    The engine and a node server (nodeserver.py) load them the same way."""
    from . import deepseek_v41 as ds
    tensors_by_layer, fetched = fetch_engram_tensors(manifest, specs_by_layer, cas_dir, mmap_root, layers=layers)
    tables = {}
    for layer, (weight, scale) in tensors_by_layer.items():
        spec = specs_by_layer[str(layer)]
        tables[layer] = ds.host_engram(layer, spec["num_embeddings"], spec["dim"], weight, scale)
        ops.ENGRAM_DIM[layer] = spec["dim"]
    return tables, fetched


def fetch_expert_banks(manifest, specs_by_layer, cas_dir, mmap_root, in_ram, layers=None, stop_event=None):
    """Fetch the raw expert bank tensors for layers, and return them as
    {layer: {name: tensor}} with the bytes fetched. Stop early, before the
    next bank, if stop_event is set."""
    out, fetched = {}, 0
    for layer, specs in specs_by_layer.items():
        if stop_event is not None and stop_event.is_set():
            return None, fetched
        if layers is not None and int(layer) not in layers:
            continue
        banks = {}
        for name, spec in specs.items():
            if stop_event is not None and stop_event.is_set():
                return None, fetched
            tensor, downloaded = fetch_stacked(
                spec, manifest["tensors"], manifest["files"], cas_dir,
                mmap_path=os.path.join(mmap_root, f"experts.{layer}.{name}") if (mmap_root and not in_ram) else None)
            banks[name] = tensor
            fetched += downloaded
        out[int(layer)] = banks
    return out, fetched


def load_expert_layers(manifest, specs_by_layer, args, cas_dir, mmap_root, in_ram, layers=None):
    """The expert banks of an artifact's layers (binding external.experts;
    `layers` limits them), each assembled as the model's own MoE module
    on the host: ({layer: module}, bytes fetched). Memory-mapped from
    files under mmap_root unless in_ram. The engine and the expert
    server (nodeserver.py) load them the same way."""
    from . import deepseek_v41 as ds
    banks_by_layer, fetched = fetch_expert_banks(manifest, specs_by_layer, cas_dir, mmap_root, in_ram, layers)
    out = {}
    for layer, banks in banks_by_layer.items():
        out[layer] = ds.host_experts(layer, args, banks)
    return out, fetched


def _has_tokenizer_files(repo):
    """False only for a local checkpoint directory with none of the
    usual tokenizer files; a Hub repo is always attempted."""
    if not os.path.isdir(repo):
        return True
    return any(os.path.exists(os.path.join(repo, f)) for f in TOKENIZER_FILES)


class Engine:
    _copy_worker_hook = None

    def __init__(self, artifact_dir, device="cpu", compile_decode=False,
                 cas_dir=None, external_dir=None, plan_given=None,
                 _copy_worker_hook=None):
        from transformers import AutoTokenizer
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING)

        self.artifact_dir = artifact_dir
        self.device = device
        self.compile_decode = compile_decode
        self._copy_worker_hook = _copy_worker_hook or self._copy_worker_hook
        # The physical plan: every choice about how this engine runs the
        # artifact, made here once and reported by explain() (plan.py).
        self.plan = Plan(device, compile_decode=compile_decode, external_dir=external_dir, given=plan_given)
        # Where the tensors the graphs call out for (expert banks, n-gram
        # tables) are placed: None keeps them in host memory; a directory
        # (local SSD) memory-maps them from files there, the page cache
        # being the RAM tier. VXPU_EXTERNAL_DIR sets it for the executor.
        self.external_dir = self.plan["external_dir"]
        with open(os.path.join(artifact_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        # this machine's local weight tier: read checkpoint files from local
        # disk, copying the missing ones from the weight bucket first
        if self.plan["blob_dir"]:
            from . import localblobs
            self.manifest["files"], summary = localblobs.localize(
                self.manifest["files"], self.plan["blob_dir"], self.plan["blob_bucket"])
            print(f"[vxpu] local weight tier: {summary['local']} of {summary['files']} files on local disk", flush=True)
        with open(os.path.join(artifact_dir, "binding.json")) as f:
            self.binding = json.load(f)
        if cas_dir:
            os.makedirs(cas_dir, exist_ok=True)

        model_type = self.manifest["config"].get("model_type")
        if model_type in CONFIG_MAPPING:
            self.config = CONFIG_MAPPING[model_type].from_dict(self.manifest["config"])
        else:
            # A model transformers does not implement (DeepSeek-V4.1 runs
            # from the model maker's module): a plain config carries what
            # generation needs (eos ids, vocab size).
            from transformers import PretrainedConfig
            self.config = PretrainedConfig.from_dict(self.manifest["config"])
        self.max_cache_len = int(self.binding.get("max_cache_len", 1024))
        # Tokenizer/processor files are not yet part of the manifest.
        # Generate (ids in, ids out) needs none; Chat does. A local
        # checkpoint without tokenizer files (the test fixtures) loads
        # without one; for a Hub repo a failed fetch fails the load, so
        # a transient Hub error is reported as one rather than as a
        # model with no tokenizer.
        repo = self.manifest["source"]["repo"]
        if _has_tokenizer_files(repo):
            self.tokenizer = AutoTokenizer.from_pretrained(repo)
        else:
            print(f"[vxpu] {repo} has no tokenizer files; "
                  "Chat is unavailable, Generate works", flush=True)
            self.tokenizer = None
        self.generation_config = self._generation_config()
        self.eos_ids = self._eos_ids()
        if self.tokenizer is not None and getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'] | capitalize }}: {{ message['content'] }}\\n"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "Assistant:"
                "{% endif %}"
            )

        # Weights move to the device once: both programs then
        # alias the same storage (device pass no-ops on tensors already
        # in place), so a session costs cache memory, not a weight copy.
        self.shared = {}
        self.bytes_fetched = 0
        self.external_bytes_fetched = 0
        self._timing_bound = 0.0
        self._timing_banks = 0.0
        self._timing_stacked = 0.0
        self._timing_graph = 0.0
        # What the graphs call out for (ops.py), owned here: it is freed
        # with the engine and seen by the ops only while this engine runs
        # a graph (external_scope).
        self.external = ops.External()
        self.expert_mover = None  # experts_gpu.Mover, when hot experts move with the traffic
        self.external.plan = self.plan  # the ops dispatch by this engine's plan
        # Names the graphs use for a tensor that is also known under a
        # canonical name (a cache tensor lifted both as a wrapper buffer
        # and as a constant); bound to the same tensor as the canonical.
        self.aliases = self.binding.get("aliases", {})
        self.min_prefill_len = self.binding.get("min_prefill_len", 1)

        self.sessions = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._decode_split = None
        self._resident = None  # set when the split decode keeps one bound state
        self._resident_owner = None
        self._ranges = []  # the ranges of layers after this engine's, on other engines (attach_ranges)
        self._current = None  # the session being generated for

        # Start fetching dense weights and external tensors first. Fetching
        # needs only the manifest and the binding, not the graph (#465).
        # The engine deserializes the decode probe while worker threads read weights.
        def fetch_dense():
            self._fetch_bound(cas_dir, stop_event=stop_event)
            if stop_event is not None and stop_event.is_set():
                return
            if self.binding.get("stacked"):
                self._fetch_stacked(cas_dir, stop_event=stop_event)

        has_external = bool(self.binding.get("external"))
        stop_event = threading.Event()
        pool = ThreadPoolExecutor(max_workers=2 if has_external else 1)
        try:
            f_dense = pool.submit(fetch_dense)
            f_external = None
            if has_external:
                from . import deepseek_v41 as ds
                args = ds.model_args(self.manifest["source"]["repo"], self.max_cache_len)
                mm = external_files_dir(self.external_dir, self.manifest)
                self.external.remote_nodes = self._connect_remote_nodes()
                self._remote_layers = {i for kind, i in self.external.remote_nodes if kind == "experts"}
                remote_tables = {i for kind, i in self.external.remote_nodes if kind == "engram"}
                specs = self.binding["external"].get("experts", {})
                if self.plan["layer_range"]:
                    self._remote_layers |= {int(layer) for layer in specs if not self._in_range(int(layer))}
                    remote_tables |= {int(k) for k in self.binding["external"].get("engram", {}) if not self._in_range(int(k))}
                local = {int(layer) for layer in specs} - self._remote_layers
                local_tables = {int(k) for k in self.binding["external"].get("engram", {})} - remote_tables
                f_external = pool.submit(
                    self._fetch_external_tensors, self.binding["external"], cas_dir, mm, local, local_tables, stop_event)

            # Shapes for derived and state tensors come from the decode program itself.
            t_probe_start = time.time()
            probe = torch.export.load(
                os.path.join(artifact_dir, "decode.pt2"))
            probe_meta = {**probe.state_dict, **probe.constants}
            t_probe = time.time() - t_probe_start

            # Check for early failure on external tensors, and wait for dense weights.
            if f_external is not None and f_external.done() and f_external.exception() is not None:
                raise f_external.exception()
            f_dense.result()

            for fqn in self.binding["derived"]:
                self.shared[fqn] = derived_tensor(
                    fqn, probe_meta[fqn], self.config).to(device)
            # Lifted but unused by the graph (other towers' tables, and the
            # banks the graph calls out for instead of reading): the graph
            # never touches them, so an empty tensor of the dtype stands in.
            # Full-size zeros would be 272 GB for DeepSeek's expert banks.
            for fqn in self.binding.get("unused", []):
                meta = probe_meta[fqn]
                self.shared[fqn] = torch.empty(0, dtype=meta.dtype, device=device)
            self.state_specs = {
                fqn: (probe_meta[fqn].shape, probe_meta[fqn].dtype)
                for fqn in self.binding["state"]}

            scratch = {
                fqn: torch.zeros(shape, dtype=dtype, device=self.device)
                for fqn, (shape, dtype) in self.state_specs.items()}

            # Build the graphs while another thread fetches external tensors.
            # The graphs do not hold external banks as attributes, so _build
            # runs concurrently with fetching expert banks and n-gram tables.
            if has_external:
                f_build = pool.submit(self._build, probe, scratch)
                done, not_done = concurrent.futures.wait(
                    [f_build, f_external], return_when=concurrent.futures.FIRST_EXCEPTION)
                for f in done:
                    if f.exception() is not None:
                        raise f.exception()
                if not_done:
                    done2, _ = concurrent.futures.wait(not_done)
                    for f in done2:
                        if f.exception() is not None:
                            raise f.exception()
                fetched_experts, fetched_engram = f_external.result()
                self._assemble_external(self.binding["external"], args, mm, local, remote_tables, fetched_experts, fetched_engram)
            else:
                self._build(probe, scratch)
        except Exception:
            stop_event.set()
            pool.shutdown(wait=True)
            raise
        pool.shutdown(wait=True)

        if self.plan["blob_dir"]:
            # the weights are loaded: the blob files' cached pages would only
            # crowd out what the engine reads every token (tables read every
            # token keep their pages)
            from . import localblobs
            keep_shas = {t.file_backed.sha for t in getattr(self.external, "engram", {}).values()
                         if getattr(t, "file_backed", None) is not None} if hasattr(self, "external") else set()
            released = localblobs.release(self.manifest["files"], self.plan["blob_dir"], exclude_shas=keep_shas)
            print(f"[vxpu] released the page cache of {released} local blob files", flush=True)

        parts = []
        concurrent_fetch = []
        if self._timing_bound > 0 or self.binding.get("bound"):
            concurrent_fetch.append(f"bound {self._timing_bound:.1f}s")
        if self._timing_stacked > 0 or self.binding.get("stacked"):
            concurrent_fetch.append(f"stacked {self._timing_stacked:.1f}s")
        if concurrent_fetch:
            parts.append(
                f"concurrent phases: {', '.join(concurrent_fetch)}, probe {t_probe:.1f}s")
        else:
            parts.append(f"probe {t_probe:.1f}s")

        if self.binding.get("external"):
            parts.append(
                f"concurrent phases: external {self._timing_banks:.1f}s, "
                f"graph {self._timing_graph:.1f}s")
        else:
            parts.append(f"graph {self._timing_graph:.1f}s")

        self.load_breakdown = ", ".join(parts)
        self.plan.note(f"load: {self.load_breakdown}")

        print("[vxpu] " + self.plan.summary(self.plan_ops()).replace("\n", "\n[vxpu] "), flush=True)

    def _fetch_bound(self, cas_dir, stop_event=None):
        """Fetch bound tensors concurrently. Copy them to the device on a
        worker thread, using a dedicated CUDA stream when running on CUDA.
        Stop early if stop_event is set."""
        bound = list(self.binding.get("bound", {}).items())
        if not bound:
            self._timing_bound = 0.0
            return
        t_start = time.time()
        q = queue.Queue(maxsize=32)
        device = self.device
        stream = torch.cuda.Stream(device=device) if str(device).startswith("cuda") else None
        copy_error = []

        def device_copy_worker():
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                while True:
                    item = q.get()
                    if item is None:
                        q.task_done()
                        break
                    if stop_event is not None and stop_event.is_set():
                        q.task_done()
                        continue
                    if not copy_error:
                        try:
                            if self._copy_worker_hook:
                                self._copy_worker_hook(item)
                            fqn, tensor, downloaded = item
                            if stream:
                                tensor = tensor.pin_memory()
                                self.shared[fqn] = tensor.to(device, non_blocking=True)
                            else:
                                self.shared[fqn] = tensor.to(device)
                            with self._lock:
                                self.bytes_fetched += downloaded
                        except Exception as e:  # noqa: BLE001
                            copy_error.append(e)
                    q.task_done()
            if stream and not copy_error:
                try:
                    stream.synchronize()
                except Exception as e:  # noqa: BLE001
                    copy_error.append(e)

        copy_thread = threading.Thread(target=device_copy_worker, daemon=True)
        copy_thread.start()

        def put_item(item):
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                if copy_error:
                    raise copy_error[0]
                try:
                    q.put(item, timeout=0.1)
                    return
                except queue.Full:
                    pass

        try:
            with ThreadPoolExecutor(max_workers=16) as pool:
                pending = {}
                for i, (fqn, ref) in enumerate(bound):
                    if stop_event is not None and stop_event.is_set():
                        break
                    if copy_error:
                        raise copy_error[0]
                    pending[i] = pool.submit(fetch_tensor, ref, self.manifest["files"], cas_dir)
                    if len(pending) >= 32:
                        j = min(pending)
                        tensor, downloaded = pending.pop(j).result()
                        if stop_event is not None and stop_event.is_set():
                            break
                        put_item((bound[j][0], tensor, downloaded))
                for j in sorted(pending):
                    if stop_event is not None and stop_event.is_set():
                        break
                    if copy_error:
                        raise copy_error[0]
                    tensor, downloaded = pending.pop(j).result()
                    if stop_event is not None and stop_event.is_set():
                        break
                    put_item((bound[j][0], tensor, downloaded))
        finally:
            q.put(None)
            copy_thread.join()
        if stop_event is not None and stop_event.is_set():
            return
        if copy_error:
            raise copy_error[0]
        self._timing_bound = time.time() - t_start

    def _fetch_stacked(self, cas_dir, stop_event=None):
        """Fetch stacked parameters assembled from checkpoint per-expert tensors.
        Stop early if stop_event is set."""
        stacked = self.binding.get("stacked", {})
        if not stacked:
            self._timing_stacked = 0.0
            return
        t_start = time.time()
        for fqn, spec in stacked.items():
            if stop_event is not None and stop_event.is_set():
                return
            tensor, downloaded = fetch_stacked(
                spec, self.manifest["tensors"], self.manifest["files"], cas_dir)
            if stop_event is not None and stop_event.is_set():
                return
            self.shared[fqn] = tensor.to(self.device)
            with self._lock:
                self.bytes_fetched += downloaded
        self._timing_stacked = time.time() - t_start

    def _fetch_external_tensors(self, external, cas_dir, mm, local_experts, local_tables, stop_event=None):
        """Fetch expert banks and n-gram table tensors into host memory without
        constructing modules. Stop early if stop_event is set."""
        t_start = time.time()
        specs = external.get("experts", {})
        banks_in_ram = self.plan["experts_placement"] == "ram"
        fetched_experts, downloaded = fetch_expert_banks(
            self.manifest, specs, cas_dir, mm, in_ram=banks_in_ram,
            layers=local_experts, stop_event=stop_event)
        with self._lock:
            self.bytes_fetched += downloaded
            self.external_bytes_fetched += downloaded
        if stop_event is not None and stop_event.is_set():
            return None, None

        use_file_tables = not self.plan["step_graph"] and kernels.available()
        if self.plan["step_graph"] and external.get("engram"):
            self.plan.note("n-gram tables copied: step graph native packages require resident tensors (#470)")

        fetched_engram = {}
        for key, spec in external.get("engram", {}).items():
            layer = int(key)
            if local_tables is not None and layer not in local_tables:
                continue
            if stop_event is not None and stop_event.is_set():
                return None, None
            w_ref = self.manifest["tensors"][spec["weight"]]
            s_ref = self.manifest["tensors"][spec["scale"]]
            w_url = self.manifest["files"][w_ref["file_sha256"]]["source"]
            s_url = self.manifest["files"][s_ref["file_sha256"]]["source"]
            if (use_file_tables
                    and w_url.startswith("file://") and s_url.startswith("file://")
                    and w_ref["file_sha256"] == s_ref["file_sha256"]):
                path = w_url[len("file://"):]
                fd = os.open(path, os.O_RDONLY)
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
                    except OSError:
                        pass
                rows = w_ref["shape"][0]
                file_table = FileTable(
                    fd=fd,
                    sha=w_ref["file_sha256"],
                    weight_offset=w_ref["offset"],
                    scale_offset=s_ref["offset"],
                    weight_row_bytes=w_ref["length"] // rows,
                    scale_row_bytes=s_ref["length"] // rows,
                    rows=rows,
                    weight_dtype=DTYPES[w_ref["dtype"]],
                    scale_dtype=DTYPES[s_ref["dtype"]],
                )
                weakref.finalize(file_table, os.close, fd)
                fetched_engram[layer] = file_table
                continue
            weight, downloaded = fetch_tensor(w_ref, self.manifest["files"], cas_dir, mmap_dir=mm)
            with self._lock:
                self.bytes_fetched += downloaded
                self.external_bytes_fetched += downloaded
            if stop_event is not None and stop_event.is_set():
                return None, None
            scale, downloaded = fetch_tensor(s_ref, self.manifest["files"], cas_dir, mmap_dir=mm)
            with self._lock:
                self.bytes_fetched += downloaded
                self.external_bytes_fetched += downloaded
            if mm:  # memory-mapped, read row by row at random: no readahead (#364)
                advise_random(weight)
                advise_random(scale)
            fetched_engram[layer] = (weight, scale)
        self._timing_banks = time.time() - t_start
        return fetched_experts, fetched_engram

    def _assemble_external(self, external, args, mm, local_experts, remote_tables, fetched_experts, fetched_engram):
        """Assemble host-side external modules from fetched tensors.

        The host_experts and host_engram functions construct modules under
        deepseek_v41.bf16_default(), which sets PyTorch's default dtype to
        bfloat16 process-wide. Run this method only after _build finishes on
        the other thread. That prevents tensors created during graph construction
        or lowering from silently turning into bfloat16.
        """
        from . import deepseek_v41 as ds
        from . import ops
        specs = external.get("experts", {})
        if fetched_experts:
            for layer, banks in fetched_experts.items():
                self.external.experts[layer] = ds.host_experts(layer, args, banks)
        for layer in sorted({int(layer) for layer in specs} & self._remote_layers):
            self.external.experts[layer] = ds.host_experts(layer, args, {})
        if self.plan["gpu_experts"] and self.external.experts:
            self._place_hot_experts(self.plan["gpu_experts"])
        if self.plan["expert_mailboxes"] and self.external.experts:
            self._attach_mailboxes()
        if fetched_engram:
            engram = external.get("engram", {})
            has_file_tables = False
            for layer, item in fetched_engram.items():
                spec = engram[str(layer)]
                if isinstance(item, FileTable):
                    has_file_tables = True
                    embed = ds.host_engram(int(layer), spec["num_embeddings"], spec["dim"])
                    embed.file_backed = item
                    self.external.engram[int(layer)] = embed
                else:
                    weight, scale = item
                    self.external.engram[int(layer)] = ds.host_engram(
                        int(layer), spec["num_embeddings"], spec["dim"], weight, scale)
                ops.ENGRAM_DIM[int(layer)] = spec["dim"]
            if has_file_tables:
                k = kernels.load()
                if k is not None and k.io_uring_ready():
                    self.plan.note("n-gram tables: rows read straight from checkpoint files with io_uring")
                else:
                    self.plan.note("n-gram tables: rows read straight from checkpoint files with pread (io_uring unavailable)")
        for key in remote_tables:
            ops.ENGRAM_DIM[key] = external.get("engram", {})[str(key)]["dim"]

    def _bind_external(self, external, cas_dir):
        """Tensors the graphs call out for (ops.py): the expert banks and
        the n-gram tables. They stay on the host, assembled as the model's
        own modules so the fork's code computes on them; the graphs, on
        whatever device they run, reach them through the ops. A device
        copy of 272 GB of experts is exactly what this avoids."""
        from . import deepseek_v41 as ds
        args = ds.model_args(self.manifest["source"]["repo"], self.max_cache_len)
        mm = external_files_dir(self.external_dir, self.manifest)
        self.external.remote_nodes = self._connect_remote_nodes()
        self._remote_layers = {i for kind, i in self.external.remote_nodes if kind == "experts"}
        remote_tables = {i for kind, i in self.external.remote_nodes if kind == "engram"}
        specs = external.get("experts", {})
        if self.plan["layer_range"]:  # another executor runs the other ranges, with their experts (#387)
            self._remote_layers |= {int(layer) for layer in specs if not self._in_range(int(layer))}
            remote_tables |= {int(k) for k in external.get("engram", {}) if not self._in_range(int(k))}
        local = {int(layer) for layer in specs} - self._remote_layers
        local_tables = {int(k) for k in external.get("engram", {})} - remote_tables
        fetched_experts, fetched_engram = self._fetch_external_tensors(external, cas_dir, mm, local, local_tables)
        self._assemble_external(external, args, mm, local, remote_tables, fetched_experts, fetched_engram)

    def _in_range(self, layer):
        """Whether this engine runs `layer` (all of them without a layer range)."""
        rng = self.plan["layer_range"]
        return rng is None or (layer >= rng[0] and (rng[1] is None or layer < rng[1]))

    def _connect_remote_nodes(self):
        """{(collection, index): ops.RemoteNode} for the plan's remote_nodes
        (name=tcp://host:port/index, ...): the host node of each name, from
        the decode graph split as it is split everywhere, and a connection
        per server."""
        from . import kernels, nodes
        spec = self.plan["remote_nodes"]
        if not spec:
            return {}
        calls = nodes.host_calls(nodes.decode_split(self.artifact_dir))
        k = kernels.load()
        if k is None:
            raise RuntimeError(f"remote_nodes: the node client needs the CPU kernels: {kernels.why_not()}")
        clients, remote = {}, {}
        for item in spec.split(","):
            name, address = item.split("=", 1)
            if name not in calls:
                raise ValueError(f"remote_nodes: {name} is not a host node of this artifact's decode step")
            if not address.startswith("tcp://"):
                raise ValueError(f"remote_nodes: {address}: only tcp:// mailboxes so far")
            server, index = address[len("tcp://"):].rsplit("/", 1)
            if server not in clients:
                host, port = server.rsplit(":", 1)
                clients[server] = k.NodeClient(host, int(port))
            remote[calls[name]] = ops.RemoteNode(clients[server], int(index), address)
        print(f"[vxpu] remote nodes: {', '.join(f'{k}.{i}' for k, i in sorted(remote))}", flush=True)
        return remote

    def _attach_mailboxes(self):
        """Decode's experts as mailboxes (mailboxes.py): per layer, the GPU
        pools of its residency, and the CPU here for the rest. (A layer a
        node server serves never reaches its mailboxes.)"""
        from . import mailboxes
        owners = collections.Counter()
        for layer, moe in self.external.experts.items():
            if layer in self._remote_layers:
                continue
            res = getattr(moe, "_vxpu_hot", None)
            moe._vxpu_mailboxes = mailboxes.Layer(moe, layer, residency=res)
            owners["cpu"] += 1
            for pool in (res.pools if res else []):
                owners[str(pool.device)] += 1
        msg = "expert mailboxes, layers per owner: " + ", ".join(f"{k} {v}" for k, v in sorted(owners.items()))
        print(f"[vxpu] {msg}", flush=True)
        self.plan.note(msg)

    def _place_hot_experts(self, spec):
        """Hold the placement's hot experts on the GPUs (experts_gpu.py)."""
        from . import experts_gpu
        gpus = experts_gpu.load_placement(spec)
        devices = experts_gpu.devices_for(self.device)
        if len(gpus) > len(devices):
            raise ValueError(f"the placement names {len(gpus)} GPUs; {len(devices)} are visible")
        def place(item):
            layer, moe = item
            by_device = [(dev, gpu.get(layer)) for dev, gpu in zip(devices, gpus)]
            if any(ids for _, ids in by_device):
                moe._vxpu_hot = experts_gpu.Residency(moe, by_device)

        # a node server holds a remote layer's banks: none here to copy from
        layers = [(layer, moe) for layer, moe in self.external.experts.items() if layer not in self._remote_layers]
        started = time.time()
        # Layers are copied a few at a time (#466): each copy gathers the held
        # experts on the host and copies them to a GPU, both outside the GIL,
        # so copies to both GPUs overlap. Each layer in flight holds a host
        # copy of its held experts (about 3.6 GB for DeepSeek-V4.1-Flash), so
        # only a few run at once.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(place, layers))
        copy_s = time.time() - started
        placed = [m._vxpu_hot for m in self.external.experts.values() if getattr(m, "_vxpu_hot", None)]
        if self.plan["gpu_experts_moves"] and self.plan["step_graph"]:
            self.plan.note("hot experts stay where the placement put them: moving them would rewrite slot maps "
                           "while the step graph replays")
        elif self.plan["gpu_experts_moves"]:
            self.expert_mover = experts_gpu.Mover(self.plan["gpu_experts_moves"],
                                                  self.plan["gpu_experts_request_weight"])
            counts = experts_gpu.load_counts(spec)
            for layer, moe in sorted(self.external.experts.items()):
                if getattr(moe, "_vxpu_hot", None):
                    self.expert_mover.attach(moe._vxpu_hot, counts.get(layer))
        total = sum(m.n_local_experts for m in self.external.experts.values())
        per_device = []
        for dev in devices:
            pools = [p for r in placed for p in r.pools if p.device == dev]
            if pools:
                per_device.append(f"{dev} {sum(p.count for p in pools)} ({sum(p.bytes for p in pools) / 1e9:.1f} GB)")
        msg = (f"hot experts on {', '.join(per_device)} (copied in {copy_s:.1f}s): {sum(r.count for r in placed)} of {total}, "
               f"in {len(placed)} layers" + (f"; moving with the traffic, {self.plan['gpu_experts_moves']} at once, "
                                             f"request weight {self.plan['gpu_experts_request_weight']:g}"
                                             if self.expert_mover else ""))
        print(f"[vxpu] {msg}", flush=True)
        self.plan.note(msg)

    def external_scope(self):
        """The context in which this engine's graphs must run: the ops
        answer from this engine's external tensors. generate() enters it;
        a caller invoking the programs directly (a per-layer probe) must."""
        return ops.bound(self.external)

    def _generation_config(self):
        """The model's generation_config.json as a dict (may be empty).

        It carries the ids a chat model actually stops on (Gemma 4:
        <end_of_turn>, <turn|>, and <|tool_response> after a tool call)
        and the sampling defaults transformers' generate() would use.
        Like the tokenizer, it is fetched from the source repo until it
        is part of the manifest.
        """
        from transformers import GenerationConfig
        try:
            return GenerationConfig.from_pretrained(
                self.manifest["source"]["repo"]).to_dict()
        except Exception as e:  # noqa: BLE001
            print(f"[vxpu] no generation_config: {e}", flush=True)
            return {}

    def _eos_ids(self):
        """Every id that ends a turn.

        Chat models stop on more than the tokenizer's eos: Gemma emits
        <end_of_turn> (config eos_token_id lists several ids). Union the
        model config, its text config, and the tokenizer.
        """
        sources = [self.manifest["config"],
                   self.manifest["config"].get("text_config") or {},
                   self.generation_config]
        ids = set()
        for source in sources:
            eos = source.get("eos_token_id")
            if isinstance(eos, int):
                ids.add(eos)
            elif isinstance(eos, (list, tuple)):
                ids.update(int(i) for i in eos)
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return ids

    def _build(self, probe, scratch):
        """Build and compile the prefill and decode graphs once with real weights in place.
        Bind scratch state now and overwrite it per session at chat time."""
        t_start = time.time()
        tensors = {**self.shared, **scratch}
        # Aliases must be placed too (an unplaced constant stays on the
        # meta device); _bind below re-points them at the canonical
        # tensors once the programs are on the device.
        tensors.update({alias: tensors[canonical]
                        for alias, canonical in self.aliases.items()
                        if canonical in tensors})
        self._prefill = load_program(
            os.path.join(self.artifact_dir, "prefill.pt2"),
            tensors, self.device).module()
        self._decode = load_program(
            probe,
            tensors, self.device).module()
        shared = share_state(self._prefill, self._decode,
                             self.binding["state"])
        assert shared > 0, "prefill/decode share no cache state"
        self._bind(scratch)  # aliases -> the canonical tensors, both graphs
        self._decode_run = self._decode
        self._prefill_run = self._prefill
        if self.compile_decode:
            strip_asserts(self._decode)
            configure_inductor(self.plan)
            mode = self.plan["compile_mode"]
            if self.plan["split_decode"]:
                # cut at the host ops; each dense segment its own compiled
                # unit (a CUDA graph on CUDA: the plan's mode defaults to
                # reduce-overhead there), the host ops eager between
                if self.plan["lower_decode"]:
                    report = lower.lower(self._decode, self.plan)
                    msg = lower.describe(report)
                    print(f"[vxpu] {msg}", flush=True)
                    self.plan.note(msg)
                self._decode_split = segments.compile_segments(
                    self._decode, self.device, mode,
                    split_compile=self.plan["split_compile"], direct=self.plan["direct_host_ops"],
                    runtime=self.plan["segment_runtime"], aoti_cache_dir=self.plan["aoti_cache_dir"])
                self._segment_info = segment_info(self._decode_split.split)
                total, compiled = self._decode_split.vxpu_segments
                print(f"[vxpu] decode graph split into {total} segments, {compiled} compiled", flush=True)
                self._decode_run = self._decode_split
                # The compiled segments mutate the caches in place; a CUDA
                # graph allows that only on tensors whose address never
                # changes. So the scratch state stays bound for good (the
                # resident state) and sessions are copied in and out of it
                # when the active session changes (see _bind).
                self._resident = scratch
                self._resident_owner = None
                if str(self.device).startswith("cuda"):
                    for tensor in scratch.values():
                        torch._dynamo.mark_static_address(tensor)
            else:
                ops.prepare_compile()
                self._decode_run = torch.compile(self._decode, mode=mode)
        if self.plan["dataflow"]:
            self._run_as_dataflow(scratch)
        if self.plan["remote_ranges"]:  # other executors run the layers after this engine's range (#387)
            from .rangeserver import RemoteRange
            ranges = []
            for item in self.plan["remote_ranges"].split(","):
                span, address = item.split("=", 1)
                ranges.append(RemoteRange(address, self.device))
                print(f"[vxpu] layers {span} on {address}", flush=True)
            self.attach_ranges(ranges)
        self._timing_graph = time.time() - t_start

    def _run_as_dataflow(self, scratch):
        """Run decode through the dataflow runner (dataflow.py): over the
        compiled split when there is one, else over the split alone (the
        segments eager). Either way the graphs stay bound to one resident
        state, as the compiled split's are (see _bind)."""
        from . import dataflow
        split = self._decode_split
        if split is None:
            split = segments.compile_segments(self._decode, self.device, None, split_compile=False,
                                              direct=self.plan["direct_host_ops"])
            self._resident = scratch
            self._resident_owner = None
        runner = dataflow.Runner(split.split, *split.vxpu_segments, device=self.device)
        if self.plan["layer_range"]:  # only this engine's range of layers, decode and prefill (#387)
            runner = dataflow.Runner(split.split, *split.vxpu_segments, device=self.device,
                                     graph=self._range_slice(runner.graph))
            prefill, _ = segments.split_at_host_ops(self._prefill)
            graph = self._range_slice(dataflow.Graph.from_split(prefill))
            self._prefill_run = dataflow.Runner(prefill, len(graph.nodes), 0, device=self.device, graph=graph)
            print(f"[vxpu] layers {self._range_text()}: decode {len(runner.graph.nodes)} nodes, "
                  f"prefill {len(graph.nodes)}", flush=True)
        if self.plan["dataflow_placement"]:
            kinds = dict(item.split("=", 1) for item in self.plan["dataflow_placement"].split(","))
            runner.place({n.name: kinds[n.kind] for n in runner.graph.nodes if n.kind in kinds})
        self._decode_split = runner
        if self.plan["step_graph"]:
            from .stepgraph import StepGraph
            self._decode_split = StepGraph(runner)
        self._segment_info = segment_info(self._decode_split.split)
        self._decode_run = self._decode_split
        msg = runner.graph.describe() + (", the step one CUDA graph" if self.plan["step_graph"] else "")
        print(f"[vxpu] {msg}", flush=True)
        self.plan.note(msg)

    def _range_slice(self, graph):
        """graph's slice for this engine's layer range."""
        first, end = self.plan["layer_range"]
        start = graph.cut_position(first) if first > 0 else 0
        # an END with no layer after it (END = the layer count) is the last range: it keeps the head
        last = end is None or not any(n.host == ("experts", end) for n in graph.nodes)
        stop = len(graph.nodes) if last else graph.cut_position(end)
        return graph.slice(start, stop)

    def _range_text(self):
        first, end = self.plan["layer_range"]
        return f"{first}-" + (str(end - 1) if end is not None else "last")

    def range_inputs(self, phase):
        """The names of the values this engine's range takes in a step of
        phase ("decode" or "prefill"): step inputs, and what earlier
        ranges made for it (cross:...)."""
        run = self._decode_run if phase == "decode" else self._prefill_run
        graph = getattr(run, "runner", run).graph
        return list(graph.inputs)

    def range_numbers(self, phase):
        """The names of this range's inputs that are plain numbers rather than
        tensors, such as a prefill's length. The wire format carries only
        tensors, so the range server turns these back into numbers when they
        arrive."""
        run = self._decode_run if phase == "decode" else self._prefill_run
        runner = getattr(run, "runner", run)
        numbers = set()
        for node in runner.graph.nodes:
            sub = getattr(runner.split, node.name)
            phs = [n for n in getattr(sub, "_orig_mod", sub).graph.nodes if n.op == "placeholder"]
            for r, ph in zip(node.args, phs):
                if getattr(r, "kind", None) == "input" and not isinstance(ph.meta.get("val"), torch.Tensor):
                    numbers.add(r.name)
        return numbers

    def run_range(self, session_id, phase, values):
        """Run this engine's range of layers for session_id (#387): its
        state bound (a session this engine has not seen starts empty), the
        decode or prefill slice run on values. Returns what the ranges after
        it need (by name), or the step's output from the last range."""
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = self.sessions[session_id] = {
                    "id": session_id, "messages": [], "cached_ids": [],
                    "state": {fqn: torch.zeros(shape, dtype=dtype, device=self.device)
                              for fqn, (shape, dtype) in self.state_specs.items()}}
            self._bind(session["state"])
            values = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in values.items()}
            with self.external_scope():
                return (self._decode_run if phase == "decode" else self._prefill_run)(**values)

    def reset_range(self, session_id):
        """Zero a session's state here: the head engine started it over."""
        with self._lock:
            if session_id in self.sessions:
                self._reset_state(self.sessions[session_id])

    def attach_ranges(self, ranges):
        """Run the layers after this engine's range on ranges, in order
        (#387): each takes the step's inputs and what the ranges before it
        made, the last returns the step's output. A range is an object with
        inputs(phase), run(session_id, phase, values) and reset(session_id):
        another engine in this process (LocalRange), or one on another
        machine."""
        self._ranges = list(ranges)
        self._decode_run = RangeChain(self, self._decode_run, "decode", self._ranges)
        self._prefill_run = RangeChain(self, self._prefill_run, "prefill", self._ranges)

    def plan_ops(self):
        """The vxpu ops this engine's graphs call."""
        found = set()
        for module in (self._prefill, self._decode):
            for _, sub in module.named_modules():  # a split module's ops are in its segments
                graph = getattr(sub, "graph", None)
                for node in graph.nodes if isinstance(graph, torch.fx.Graph) else ():
                    target = str(node.target)
                    if node.op == "call_function" and target.startswith("vxpu."):
                        found.add("vxpu::" + target.split(".")[1])
        return found

    def explain(self):
        """This engine's physical plan (plan.Plan.explain), limited to the
        ops its graphs call, with the implementations selected so far."""
        return self.plan.explain(self.plan_ops())

    def _reset_state(self, session):
        """Zero a session's state tensors in place (the graphs stay bound),
        here and in the ranges after this engine's (#387)."""
        for r in self._ranges:
            r.reset(session["id"])
        for tensor in session["state"].values():
            tensor.zero_()
        if self._resident is not None and self._resident_owner is session["state"]:
            for tensor in self._resident.values():
                tensor.zero_()

    def _bind(self, state):
        """Make `state` the one the graphs read and write.

        With a resident state (the split decode, see _build) the graphs
        stay bound to it: the previous session's contents are copied out
        to its own tensors and this session's copied in, only when the
        active session changes. Otherwise the graphs' buffers are pointed
        at this session's tensors, as below."""
        if self._resident is not None and state is not self._resident:
            if self._resident_owner is state:
                return
            if self._resident_owner is not None:
                for fqn, tensor in self._resident.items():
                    self._resident_owner[fqn].copy_(tensor)
            for fqn, tensor in self._resident.items():
                tensor.copy_(state[fqn])
            self._resident_owner = state
            return
        self._bind_pointers(state)

    def _bind_pointers(self, state):
        """Point both graphs' state buffers at this session's tensors.

        A reference swap, O(number of buffers); the compiled decode
        re-reads its buffers on each call, so this triggers no recompile.
        """
        state = {**state, **{alias: state[canonical]
                             for alias, canonical in self.aliases.items()
                             if canonical in state}}
        for module in (self._prefill, self._decode):
            for fqn, tensor in state.items():
                parent, name = (fqn.rsplit(".", 1) if "." in fqn
                                else ("", fqn))
                target = (module.get_submodule(parent) if parent
                          else module)
                if name in target._buffers:
                    target._buffers[name] = tensor
                elif isinstance(target.__dict__.get(name), torch.Tensor):
                    # A lifted constant (a cache tensor the model reaches
                    # through its cache object) is a plain attribute of
                    # the unlifted module, not a registered buffer.
                    setattr(target, name, tensor)

    def new_session(self):
        with self._lock:
            session_id = f"s{self._next_id}"
            self._next_id += 1
            self.sessions[session_id] = {
                "id": session_id,
                "state": {
                    fqn: torch.zeros(shape, dtype=dtype,
                                     device=self.device)
                    for fqn, (shape, dtype) in self.state_specs.items()},
                "messages": [],
                # Token ids whose KV entries are in the cache, kept in
                # step with every prefill/decode so an interrupted
                # stream never leaves it describing a cache that is not
                # there.
                "cached_ids": [],
            }
        return session_id

    def chat(self, session_id, text, max_new_tokens=96, raw_prompt=False,
             telemetry=vxpu_pb2.TELEMETRY_NONE):
        """One templated (or raw) text turn; greedy, like the CLI expects.
        With ``telemetry=vxpu_pb2.TELEMETRY_SUMMARY``, the reply carries the turn's
        execution telemetry under ``"telemetry"``. ``telemetry=vxpu_pb2.TELEMETRY_TRACE``
        also records each decode step's expert selections, in order. Without
        either, nothing is recorded."""
        with self._lock:
            return self._chat_locked(session_id, text, max_new_tokens,
                                     raw_prompt, telemetry)

    def _prompt_ids(self, session, text, raw_prompt):
        """Token ids of the whole conversation so far plus this turn.

        Templated mode: the engine owns the transcript and renders it
        with the tokenizer's chat template. Raw mode: the client owns
        the transcript and sends the complete rendered prompt (e.g.
        with tools); it is tokenized verbatim, adding BOS only if the
        rendered text does not already start with it.
        """
        if self.tokenizer is None:
            raise ValueError("this model has no tokenizer; use Generate with token ids")
        if raw_prompt:
            bos = self.tokenizer.bos_token or ""
            add_special = not (bos and text.startswith(bos))
            return self.tokenizer(
                text, add_special_tokens=add_special,
                return_tensors="pt")["input_ids"][0].tolist()
        messages = session["messages"] + [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)["input_ids"][0].tolist()

    def _chat_locked(self, session_id, text, max_new_tokens, raw_prompt,
                     telemetry=vxpu_pb2.TELEMETRY_NONE):
        session = self._session(session_id)
        ids = self._prompt_ids(session, text, raw_prompt)
        token_ids, stats = [], None
        for chunk in self._generate_locked(
                session, ids, max_new_tokens, SamplingParams(do_sample=False), telemetry=telemetry):
            if chunk["token_ids"]:
                token_ids.extend(chunk["token_ids"])
            if chunk["done"]:
                stats = chunk

        # Text replies exclude the end-of-turn id itself. Raw clients
        # own the format and need the model's other control tokens
        # (e.g. Gemma's <|tool_call> ... <tool_call|> delimiters);
        # templated replies are plain text.
        if token_ids and token_ids[-1] in self.eos_ids:
            token_ids = token_ids[:-1]
        reply = self.tokenizer.decode(token_ids,
                                      skip_special_tokens=not raw_prompt)
        if not raw_prompt:
            session["messages"].append({"role": "user", "content": text})
            session["messages"].append(
                {"role": "assistant", "content": reply})
        return {
            "text": reply,
            "session_tokens": len(session["cached_ids"]),
            "new_prompt_tokens": stats["prefilled_tokens"],
            "generated": stats["generated"],
            "prefill_ms": stats["prefill_ms"],
            "ms_per_token": stats["ms_per_token"],
            **({"telemetry": stats["telemetry"]} if "telemetry" in stats else {}),
        }

    def generate(self, session_id, input_ids, max_new_tokens, params,
                 extra_eos_ids=(), logits_sink=None,
                 telemetry=vxpu_pb2.TELEMETRY_NONE):
        """transformers' generate() loop, streamed.

        Yields dicts {"token_ids": [...], "done": bool, ...}; the last
        one has done=True and the statistics. The generation lock is
        held for the whole stream (one accelerator serves turns
        serially anyway). ``logits_sink(position, logits)``, if given,
        sees the next-token logits (a 1-D tensor) before each pick,
        position 0 being the first generated token; used to record
        test vectors. With ``telemetry=vxpu_pb2.TELEMETRY_SUMMARY``, the final
        event carries the call's execution telemetry under ``"telemetry"``.
        ``telemetry=vxpu_pb2.TELEMETRY_TRACE`` also records each decode step's expert
        selections, in order. Without either, nothing is recorded.
        """
        with self._lock:
            session = self._session(session_id)
            yield from self._generate_locked(
                session, list(input_ids), max_new_tokens, params,
                extra_eos_ids, logits_sink, telemetry)

    def _session(self, session_id):
        if session_id not in self.sessions:
            raise ValueError(f"unknown session_id: {session_id}")
        return self.sessions[session_id]

    def _generate_locked(self, session, ids, max_new_tokens, params,
                         extra_eos_ids=(), logits_sink=None,
                         telemetry=vxpu_pb2.TELEMETRY_NONE):
        if isinstance(telemetry, bool) or telemetry not in vxpu_pb2.Telemetry.values():
            raise ValueError(f"telemetry accepts only Telemetry enum values, got {telemetry!r}")
        if self.expert_mover is not None:  # generate and chat: the hot experts follow this request's selections too
            self.expert_mover.begin_request()
        recorder = None
        if telemetry != vxpu_pb2.TELEMETRY_NONE:
            recorder = Recorder(self.device, getattr(self, "_segment_info", None)
                                if self._decode_split is not None else None,
                                trace=(telemetry == vxpu_pb2.TELEMETRY_TRACE))
            self.external.telemetry = recorder
            if self._decode_split is not None:
                self._decode_split.set_recorder(recorder)
        try:
            with self.external_scope():
                yield from self._generate_scoped(session, ids, max_new_tokens, params,
                                                 extra_eos_ids, logits_sink, recorder)
        finally:
            if recorder is not None:
                self.external.telemetry = None
                if self._decode_split is not None:
                    self._decode_split.set_recorder(None)

    def _generate_scoped(self, session, ids, max_new_tokens, params,
                         extra_eos_ids, logits_sink, recorder=None):
        self._bind(session["state"])
        self._current = session
        total_len = len(ids)
        if total_len == 0:
            raise ValueError("empty prompt")
        if total_len > self.max_cache_len:
            raise ValueError(
                f"prompt length ({total_len} tokens) exceeds maximum "
                f"cache capacity ({self.max_cache_len} tokens)")
        eos_ids = self.eos_ids | set(int(i) for i in extra_eos_ids)
        sampler = params.resolve(self.generation_config)

        # Reuse the cache only for the prefix that is really there:
        # re-rendering a transcript need not reproduce the generated
        # ids token-for-token, and a client may rewrite history.
        # Prefill from the first differing token: the caches are
        # position-addressed (explicit cache_position, causal masks), so
        # entries from that position on are simply overwritten and
        # nothing stale beyond it is ever attended.
        cached = session["cached_ids"]
        start = 0
        while (start < len(cached) and start < total_len
               and cached[start] == ids[start]):
            start += 1
        if start == total_len:
            # Everything is cached, including the last prompt token;
            # recompute its logits by re-prefilling just that token.
            start = total_len - 1
        if self.binding.get("prefill_from_zero") and (
                start < len(cached)
                or (start > 0 and total_len - start >= self.min_prefill_len)):
            # This model's prefill graph starts at position 0 (its ring
            # buffers and compressed caches are seeded by the prefill),
            # so it cannot extend a cached prefix: a rewritten prefix,
            # and a multi-token continuation (the next turn of a chat),
            # mean a fresh state and a full prefill. A single new token
            # goes through the decode graph below, which takes a position.
            self._reset_state(session)
            start = 0

        allowed_tokens = self.max_cache_len - total_len
        max_tokens_to_generate = max(0, min(max_new_tokens, allowed_tokens))
        new_ids = torch.tensor([ids[start:]], dtype=torch.long,
                               device=self.device)

        # The prefill overwrites cache positions from `start` on; until
        # it has finished only the shared prefix is known to be there.
        session["cached_ids"] = cached[:start]
        prefill_started = time.perf_counter()
        if new_ids.shape[1] < self.min_prefill_len:
            # The prefill graph's length starts at min_prefill_len; a
            # single new token (a re-asked prompt, a one-token turn) is
            # exactly what the decode graph computes.
            logits = self._decode_run(
                input_ids=new_ids,
                cache_position=torch.tensor([start], device=self.device))
        else:
            logits = self._prefill_run(
                input_ids=new_ids,
                cache_position=torch.arange(start, total_len,
                                            device=self.device))
        prefill_s = time.perf_counter() - prefill_started
        cached = session["cached_ids"] = list(ids)
        if recorder is not None:
            recorder.prefill_ms = prefill_s * 1e3
            recorder.prefill_tokens = total_len - start
            recorder.phase = "decode"

        # The returned sequence matches transformers: the end-of-turn
        # id that stops generation is included (and counted) but, as
        # with transformers, never fed through the model, so it is not
        # part of the cache.
        token_ids = []
        finish_reason = "length"
        if logits_sink is not None:
            logits_sink(0, logits[0, -1])
        next_id = sampler.pick(logits[0, -1], ids)
        loop_started = time.perf_counter()
        for step in range(max_tokens_to_generate):
            token_ids.append(next_id)
            yield {"token_ids": [next_id], "done": False}
            if next_id in eos_ids:
                finish_reason = "eos"
                break
            if len(cached) >= self.max_cache_len:
                break
            step_started = time.perf_counter() if recorder is not None else 0.0
            logits = self._decode_run(
                input_ids=torch.tensor([[next_id]], device=self.device),
                cache_position=torch.tensor([len(cached)],
                                            device=self.device))
            cached.append(next_id)
            if logits_sink is not None:
                logits_sink(step + 1, logits[0, -1])
            next_id = sampler.pick(logits[0, -1], cached)
            if recorder is not None:
                recorder.step_ms.append((time.perf_counter() - step_started) * 1e3)
        loop_s = time.perf_counter() - loop_started

        done = {
            "token_ids": [],
            "done": True,
            "finish_reason": finish_reason,
            "prompt_tokens": total_len,
            "prefilled_tokens": total_len - start,
            "generated": len(token_ids),
            "prefill_ms": round(prefill_s * 1e3),
            "ms_per_token": round(
                loop_s / max(len(token_ids), 1) * 1e3, 1),
        }
        if recorder is not None:
            done["telemetry"] = recorder.result(plan_record(self.explain()))
        yield done


class SamplingParams:
    """Generation parameters as transformers' GenerationConfig names
    them. None means "use the model's generation_config default", so a
    bare request samples exactly as model.generate(input_ids) would."""

    FIELDS = ("do_sample", "temperature", "top_k", "top_p",
              "repetition_penalty")

    def __init__(self, do_sample=None, temperature=None, top_k=None,
                 top_p=None, repetition_penalty=None, seed=None):
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed

    def resolve(self, generation_config):
        defaults = {"do_sample": False, "temperature": 1.0, "top_k": 50,
                    "top_p": 1.0, "repetition_penalty": 1.0}
        values = {}
        for name in self.FIELDS:
            value = getattr(self, name)
            if value is None:
                value = generation_config.get(name)
            if value is None:
                value = defaults[name]
            values[name] = value
        return Sampler(seed=self.seed, **values)


class Sampler:
    """The logits processors transformers applies for these settings,
    in its order: repetition penalty, temperature, top-k, top-p, then
    multinomial sampling (or argmax when do_sample is false)."""

    def __init__(self, do_sample, temperature, top_k, top_p,
                 repetition_penalty, seed=None):
        # The same bounds transformers' processors enforce.
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be > 0 when sampling")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be in [0, 1]")
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_k = int(top_k or 0)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(int(seed))

    def pick(self, logits, context_ids):
        logits = logits.detach().float().cpu()
        if self.repetition_penalty != 1.0 and context_ids:
            ids = torch.tensor(sorted(set(context_ids)), dtype=torch.long)
            scores = logits[ids]
            logits[ids] = torch.where(scores < 0,
                                      scores * self.repetition_penalty,
                                      scores / self.repetition_penalty)
        if not self.do_sample:
            return int(logits.argmax())
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if 0 < self.top_k < logits.numel():
            cutoff = torch.topk(logits, self.top_k).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=False)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative <= (1 - self.top_p)
            remove[-1] = False  # always keep the most likely token
            logits = logits.masked_fill(
                remove.scatter(0, sorted_idx, remove), float("-inf"))
        probs = logits.softmax(-1)
        return int(torch.multinomial(probs, 1, generator=self.generator))


class RangeChain(torch.nn.Module):
    """A head engine's step with the layers after its range on other
    engines (#387): its own slice, then each range in turn, each given the
    step's inputs and what the ranges before it made, for the session
    being generated."""

    def __init__(self, engine, local, phase, ranges):
        super().__init__()
        self.engine, self.local, self.phase, self.ranges = engine, local, phase, ranges

    def forward(self, **step):
        made = self.local(**step)
        session = self.engine._current["id"]
        for r in self.ranges:
            out = r.run(session, self.phase, {n: step[n] if n in step else made[n] for n in r.inputs(self.phase)})
            if not isinstance(out, dict):
                return out
            made.update(out)
        return made


class LocalRange:
    """Another engine in this process running a range of layers, as a
    range of a head engine's (Engine.attach_ranges): for tests, and the
    shape of a range on another machine."""

    def __init__(self, engine, device):
        self.engine, self.device = engine, device

    def inputs(self, phase):
        return self.engine.range_inputs(phase)

    def run(self, session_id, phase, values):
        out = self.engine.run_range(session_id, phase, values)
        if isinstance(out, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in out.items()}
        return out.to(self.device)

    def reset(self, session_id):
        self.engine.reset_range(session_id)

