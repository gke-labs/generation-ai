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

from . import checks, kernels, lower, ops, rehydrate, segments, vxpu_pb2  # noqa: F401  (registers vxpu::* before a graph that calls them is loaded)
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


def sum_expert_stats(graphs):
    """The per-layer distinct-expert counts of several step graphs added up (None where no graph
    counts: the statistics are off, or the graphs have no experts, as a drafter's)."""
    total = None
    for g in graphs:
        counts = g.expert_stats() if g is not None and hasattr(g, "expert_stats") else None
        if counts is None:
            continue
        total = total or {}
        for layer, n in counts.items():
            total[layer] = total.get(layer, 0) + n
    return total


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
        from . import physplan
        physplan.apply_env()  # VXPU_PLAN=path: the plan file's ranges and kernel table, as the switches below
        self.plan = Plan(device, compile_decode=compile_decode, external_dir=external_dir, given=plan_given)
        self.checks = checks.Checks(self)  # the standing self-checks (checks.py), by the plan's switches
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
        # tokens per row in a decode step: more than one verifies drafts (_Rows)
        self.step_tokens = self.binding.get("step_tokens", 1)
        self._draft = self._draft_run = None  # the model's own drafter (draft.pt2), when exported

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

        # the weights are loaded: the blob files' cached pages would only
        # crowd out what the engine reads every token (tables read every
        # token keep their pages); local files are those in blob_dir or
        # named by a file:// source in the manifest
        from . import localblobs
        keep_shas = {t.file_backed.sha for t in getattr(self.external, "engram", {}).values()
                     if getattr(t, "file_backed", None) is not None} if hasattr(self, "external") else set()
        released = localblobs.release(self.manifest["files"], self.plan["blob_dir"], exclude_shas=keep_shas)
        if released:
            print(f"[vxpu] released the page cache of {released} local blob files", flush=True)
        if str(self.device).startswith("cuda") and self._resident is not None:
            self._warm_up()

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
        self.checks.golden()  # the model's golden prompts, decoded and compared; a mismatch refuses the load

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
            # a layer whose every expert now sits on a GPU reads its host banks no more: their pages
            # leave the cache, so the n-gram tables read every step keep theirs
            paths, nbytes = [], 0
            for moe in self.external.experts.values():
                hot = getattr(moe, "_vxpu_hot", None)
                if hot is None or hot.count < moe.n_local_experts:
                    continue
                for _, param in moe.named_parameters(recurse=False):
                    path = rehydrate.file_of(param)
                    if path is not None:
                        paths.append(path)
                        nbytes += param.numel() * param.element_size()
            if paths:
                released = rehydrate.release_files(paths)
                print(f"[vxpu] released the page cache of {released} expert bank files ({nbytes / 1e9:.0f} GB): "
                      f"every expert of those layers is on a GPU", flush=True)
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
                    # the tables are read every step: their pages come in now, not lookup by lookup
                    took = [rehydrate.populate(t) for t in (weight, scale)]
                    if all(x is not None for x in took):
                        print(f"[vxpu] n-gram table of layer {layer} resident: "
                              f"{(weight.numel() * weight.element_size() + scale.numel() * scale.element_size()) / 1e9:.1f} GB "
                              f"in {sum(took):.1f} s", flush=True)
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
        if "draft" in self.binding:
            self._draft = load_program(os.path.join(self.artifact_dir, "draft.pt2"), tensors, self.device).module()
            share_state(self._prefill, self._draft, self.binding["state"])
            self._draft_run = self._draft
        self._bind(scratch)  # aliases -> the canonical tensors, both graphs
        self._decode_run = self._decode
        self._prefill_run = self._prefill
        self.checks.keep_logical(self._decode)  # the graph as exported, before the lowering: the reference check's
        if self.compile_decode:
            strip_asserts(self._decode)
            configure_inductor(self.plan)
            mode = self.plan["compile_mode"]
            if self.plan["split_decode"]:
                # cut at the host ops; each dense segment its own compiled
                # unit (a CUDA graph on CUDA: the plan's mode defaults to
                # reduce-overhead there), the host ops eager between
                if self.plan["lower_decode"]:
                    fenced = segments.mark_islands(self._decode)  # before the lowering renames the ops' nodes
                    report = lower.lower(self._decode, self.plan,
                                         layer_device=self._layer_device if self.plan["layer_devices"] else None)
                    for msg in (lower.describe(report), lower.describe_measured(report)):
                        if msg:
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
                if self._draft is not None:
                    strip_asserts(self._draft)
                    self._draft_run = segments.compile_segments(
                        self._draft, self.device, mode,
                        split_compile=self.plan["split_compile"], direct=self.plan["direct_host_ops"],
                        runtime=self.plan["segment_runtime"], aoti_cache_dir=self.plan["aoti_cache_dir"])
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
        if self.plan["layer_devices"]:
            self._shard_layers()
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

    def _layer_device(self, layer):
        """The GPU of `layer` under layer_devices (START:END=cuda:N,...), or None."""
        for item in self.plan["layer_devices"].split(","):
            span, device = item.split("=", 1)
            first, end = span.split(":")
            if int(first) <= layer and (end == "" or layer < int(end)):
                return torch.device(device)
        return None

    def _mtp_base(self):
        """The layer id of the drafter's first stage: one past the backbone's last layer (the id
        the placement and the expert tensors use for the stages, deepseek_v41.external_spec)."""
        if getattr(self, "_mtp_base_value", None) is None:
            layers = [l for seg in segment_info(self._decode_split.split) for l in seg["layers"]]
            self._mtp_base_value = (max(layers) + 1) if layers else 0
        return self._mtp_base_value

    def _graph(self, split):
        """The dataflow graph of a split, with the drafter's stages counted as layers after the
        backbone's so layer sharding places them."""
        from . import dataflow
        return dataflow.Graph.from_split(split, self._mtp_base())

    def _shard_layers(self):
        """Layer sharding: move each dense segment's resident tensors (its layers' weights and
        caches) to the GPU layer_devices gives its layers, in the decode, prefill and draft
        programs alike (their state is shared; their weights may be copies per program). A
        segment spanning two layers (one layer's tail and the next's attention) goes with the
        later layer; a segment with no layer (the head) goes where the segment before it went;
        the drafter's stages count as the layers after the backbone's. A tensor read by segments
        on two GPUs is owned by the one that writes it (the decode's last segment writes the
        drafter's rings; a group's first layer writes the compressed KV its later layers read),
        else by the first to read it, across all the programs; the dataflow runner copies it for
        the others (dataflow.Runner._arg: a state tensor's replica is refreshed once a step from
        the owner's). A tensor written on two GPUs is an error: the replica rule copies the
        owner's writes out, never back. The runner then runs every segment where its tensors are,
        and the step crosses GPUs once per range."""
        from . import dataflow
        state = set(self.binding["state"])
        programs = [("decode", self._decode_split.split)]
        for name, program in (("prefill", self._prefill), ("draft", self._draft)):
            if program is not None:
                programs.append((name, segments.split_at_host_ops(program)[0]))
        default = torch.device(self.device)
        if default.index is None:  # "cuda" is "cuda:0" here; the comparisons below need the index
            default = torch.device(default.type, torch.cuda.current_device())
        # where each dense segment runs, and which residents it reads and writes there
        plan = []  # (program, split, node, device)
        readers, writers = {}, {}  # resident -> [devices, first touch first] / {device: (program, node)}
        self._planned = {}  # (program, dense segment name) -> device, for the runners' placement
        for name, split in programs:
            dense = [n for n in self._graph(split).nodes if n.kind == "dense"]
            devices = [self._layer_device(max(int(x) for x in n.layers)) if n.layers else None for n in dense]
            # a segment with no layer goes with the segment before it (the head), or, before any
            # layered segment (the drafter's embedding), with the first that has one
            device = default
            for i, placed in enumerate(devices):  # forward: a layer-less segment follows the one before it
                device = placed or device
                devices[i] = device
            following = None
            for i in reversed(range(len(dense))):  # backward: leading layer-less segments join the first layered one
                if dense[i].layers:
                    following = devices[i]
                elif following is not None and not any(n.layers for n in dense[:i]):
                    devices[i] = following
            for node, placed in zip(dense, devices):
                plan.append((name, split, node, placed))
                self._planned[(name, node.name)] = placed
                written = {node.args[i].name for i in node.writes
                           if i < len(node.args) and isinstance(node.args[i], dataflow.Ref)
                           and node.args[i].kind == "resident"}
                for r in dataflow._flat(node.refs()):
                    if isinstance(r, dataflow.Ref) and r.kind == "resident":
                        readers.setdefault(r.name, []).append(placed)
                        if r.name in written:
                            writers.setdefault(r.name, {})[str(placed)] = (name, node.name, placed)
        owner, by_writer = {}, []
        for rname, devs in readers.items():
            ws = writers.get(rname, {})
            if len(ws) > 1:
                where = ", ".join(f"{prog}'s {node} on {dev}" for prog, node, dev in ws.values())
                raise RuntimeError(f"layer sharding: {rname} is written on two GPUs ({where}); a tensor must be "
                                   f"written on one GPU, since the replica rule copies the owner's writes to its "
                                   f"readers and never back")
            owner[rname] = next(iter(ws.values()))[2] if ws else devs[0]
            if ws and owner[rname] != devs[0]:
                by_writer.append(rname)
        # the segments' own tensors (the constants export lifted into them, which no argument names)
        # copied per segment; the residents moved in place to their owner, so every program holding
        # the tensor follows
        bytes_moved, shared, done = collections.Counter(), {}, set()
        for name, split, node, device in plan:
            bytes_moved[str(device)] += _move_module_tensors(getattr(split, node.name), device)
            for r in dataflow._flat(node.refs()):
                if not (isinstance(r, dataflow.Ref) and r.kind == "resident"):
                    continue
                if device != owner[r.name]:
                    shared[r.name] = (owner[r.name], device)
                if (name, r.name) in done:
                    continue
                done.add((name, r.name))
                t = dataflow._attr(split, r.name)
                if isinstance(t, torch.Tensor) and t.device != owner[r.name]:
                    t.data = t.data.to(owner[r.name])
                    bytes_moved[str(owner[r.name])] += t.numel() * t.element_size()
        counts = collections.Counter()
        for name, _, node, device in plan:
            counts[(name, str(device))] += 1
        by_program = {name: {d: n for (p, d), n in sorted(counts.items()) if p == name} for name, _ in programs}
        msg = ("layer sharding: " + (", ".join(f"{b / 1e9:.1f} GB to {d}" for d, b in sorted(bytes_moved.items()))
                                     or "nothing to move") + f"; dense segments by GPU {by_program}")
        if shared:
            names = sorted(shared)
            stale = [n for n in names if n in state]
            msg += f"; {len(names)} tensors read from a GPU other than their owner's (copied for the reader)"
            if stale:  # a cache written on one GPU and read on another: its replica is refreshed once a step
                msg += (f"; {len(stale)} of them state (a cache shared by layers on two GPUs, or the drafter's "
                        f"rings), replicated and refreshed once a step at the first read on the other GPU: "
                        f"{', '.join(stale)}")
        if by_writer:
            msg += (f"; {len(by_writer)} owned by the GPU that writes them rather than the first to read them: "
                    f"{', '.join(by_writer)}")
        print(f"[vxpu] {msg}", flush=True)
        self.plan.note(msg)

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
        runner = dataflow.Runner(split.split, *split.vxpu_segments, device=self.device,
                                 graph=self._graph(split.split) if self.plan["layer_devices"] else None)
        runner.state = set(self.binding["state"])
        if self.plan["layer_range"]:  # only this engine's range of layers, decode and prefill (#387)
            runner = dataflow.Runner(split.split, *split.vxpu_segments, device=self.device,
                                     graph=self._range_slice(runner.graph))
            prefill, _ = segments.split_at_host_ops(self._prefill)
            graph = self._range_slice(dataflow.Graph.from_split(prefill))
            self._prefill_run = dataflow.Runner(prefill, len(graph.nodes), 0, device=self.device, graph=graph)
            print(f"[vxpu] layers {self._range_text()}: decode {len(runner.graph.nodes)} nodes, "
                  f"prefill {len(graph.nodes)}", flush=True)
        elif self.plan["layer_devices"]:  # prefill segment by segment too, each where its tensors are
            # a host node (a layer's experts, an n-gram lookup) runs with its layer: its pools are
            # there, and its inputs and reply never cross
            runner.place(self._host_placement(runner.graph))
            prefill, _ = segments.split_at_host_ops(self._prefill)
            graph = self._graph(prefill)
            self._prefill_run = dataflow.Runner(prefill, len(graph.nodes), 0, device=self.device, graph=graph,
                                                placement=self._host_placement(graph, "prefill"))
            self._prefill_run.state = set(self.binding["state"])
            devices = collections.Counter(str(d) for d in runner._devices.values())
            pdevices = collections.Counter(str(d) for d in self._prefill_run._devices.values())
            print(f"[vxpu] layer sharding: decode nodes by GPU {dict(devices)}, prefill nodes {dict(pdevices)}",
                  flush=True)
        if self.plan["dataflow_placement"]:
            kinds = dict(item.split("=", 1) for item in self.plan["dataflow_placement"].split(","))
            runner.place({n.name: kinds[n.kind] for n in runner.graph.nodes if n.kind in kinds})
        self._decode_split = runner
        if self.plan["step_graph"]:
            from .stepgraph import StepGraph
            self._decode_split = StepGraph(runner, plan=self.plan)
            draft = self._draft_run
            if draft is not None and hasattr(draft, "split") and not self.plan["layer_range"]:
                # the drafter's step too, so the native loop runs a speculative step with no Python
                draft_runner = dataflow.Runner(draft.split, *draft.vxpu_segments, device=self.device,
                                               graph=self._graph(draft.split) if self.plan["layer_devices"] else None)
                draft_runner.state = set(self.binding["state"])
                if self.plan["layer_devices"]:
                    draft_runner.place(self._host_placement(draft_runner.graph, "draft"))
                    ddevices = collections.Counter(str(d) for d in draft_runner._devices.values())
                    print(f"[vxpu] layer sharding: draft nodes by GPU {dict(ddevices)}", flush=True)
                self._draft_run = StepGraph(draft_runner, plan=self.plan, role="draft")
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

    def _warm_up(self):
        """Run throwaway decode steps on zeros, then zero the state (#462).

        Compiled segments are built, or loaded from the cache, on their first
        call, and capture their CUDA graph on the second. Doing both here
        means the engine is ready when it says so. Before, a range engine on
        a fresh machine compiled during its first request for minutes, longer
        than the engine before it waits for a reply, so that request failed.
        A head engine warms only its own layers: the ranges after it may not
        be serving yet.

        One eager step comes first. The ops make some weight copies on their
        first call: the head widened to fp32, and fp8 weights as bf16, when a
        step has more rows than the GPU's GEMV kernels take. Made inside a
        compiled segment's first call, a copy would land in that CUDA graph's
        memory pool, which may hold only what the graph allocates each step,
        and the graph fails. The eager step makes them outside any pool."""
        run = self._decode_run.local if isinstance(self._decode_run, RangeChain) else self._decode_run
        values = self._zero_inputs(getattr(run, "runner", run))
        started = time.time()
        with self._lock, self.external_scope():
            if not self.plan["layer_range"]:  # eager, a range engine would call the other ranges' experts
                self._eager_step(self._decode, "decode")
            for _ in range(2):
                run(**values)
            if self._draft is not None:
                draft_values = self._zero_inputs(self._draft)
                self._eager_step(self._draft, "draft")
                for _ in range(2):
                    self._draft_run(**draft_values)
            torch.cuda.synchronize(self.device)
            for tensor in (self._resident or {}).values():
                tensor.zero_()
        msg = f"warm-up: decode steps in {time.time() - started:.1f}s (weight copies made, segments built or loaded, graphs captured)"
        print(f"[vxpu] {msg}", flush=True)
        self.plan.note(msg)
        counts = getattr(run, "node_counts", None)
        if counts:  # the captured step's shape: what fusion and the plan's choices change
            msg = "the decode step launches " + ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
            print(f"[vxpu] {msg}", flush=True)
            self.plan.note(msg)

    def _host_placement(self, graph, program="decode"):
        """Under layer sharding, where each node of graph runs: a host node (a layer's experts, an
        n-gram lookup) with its layer, so its expert pools are there and its inputs and reply never
        cross; a dense segment where _shard_layers planned it (else the runner puts it where its
        first resident tensor is, which is wrong for a segment whose first tensor is shared and
        owned elsewhere: the drafter's embedding, the segment after a range boundary)."""
        placement = {n.name: str(self._layer_device(max(int(x) for x in n.layers)) or self.device)
                     for n in graph.nodes if n.kind != "dense" and n.layers}
        planned = getattr(self, "_planned", {})
        for n in graph.nodes:
            if n.kind == "dense" and (program, n.name) in planned:
                placement[n.name] = str(planned[(program, n.name)])
        return placement

    def _eager_step(self, program, name="decode"):
        """One eager decode step of program (the decode or draft graph module) on zeros. With the
        layers sharded across GPUs the whole module cannot run as one (its weights are on several),
        so it runs segment by segment through the dataflow runner, each where its tensors are, as
        the prefill does; the host nodes go with their layer."""
        if not self.plan["layer_devices"]:
            program(**self._zero_inputs(program))
            return
        from . import dataflow
        split, _ = segments.split_at_host_ops(program)
        graph = self._graph(split)
        runner = dataflow.Runner(split, len(graph.nodes), 0, device=self.device, graph=graph,
                                 placement=self._host_placement(graph, name))
        runner.state = set(self.binding["state"])
        runner(**self._zero_inputs(runner))

    def _zero_inputs(self, runner):
        """Zeros shaped like each input of a decode step: from the dataflow
        graph's inputs and the placeholders of the segments that take them,
        or from the placeholders of runner (a graph module: the decode or
        draft graph) or else of the decode graph."""
        values = {}
        if hasattr(runner, "graph") and hasattr(runner, "split"):
            for node in runner.graph.nodes:
                sub = getattr(runner.split, node.name)
                phs = [n for n in getattr(sub, "_orig_mod", sub).graph.nodes if n.op == "placeholder"]
                for r, ph in zip(node.args, phs):
                    if getattr(r, "kind", None) == "input" and r.name not in values:
                        meta = ph.meta.get("val")
                        values[r.name] = (torch.zeros(meta.shape, dtype=meta.dtype, device=self.device)
                                          if isinstance(meta, torch.Tensor) else 0)
            return values
        module = runner if isinstance(runner, torch.fx.GraphModule) else self._decode
        for ph in (n for n in module.graph.nodes if n.op == "placeholder"):
            meta = ph.meta.get("val")
            if isinstance(meta, torch.Tensor):
                values[ph.name] = torch.zeros(meta.shape, dtype=meta.dtype, device=self.device)
        return values

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
        for module in (m for m in (self._prefill, self._decode, self._draft) if m is not None):
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
        if self._uses_rows():  # continuous batching: the turn takes a row
            with self._lock:
                session = self._session(session_id)
                ids = self._prompt_ids(session, text, raw_prompt)
            events = self._scheduled(session, ids, max_new_tokens, SamplingParams(do_sample=False))
            return self._chat_reply(session, text, raw_prompt, events)
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
        events = self._generate_locked(session, ids, max_new_tokens, SamplingParams(do_sample=False), telemetry=telemetry)
        return self._chat_reply(session, text, raw_prompt, events)

    def _chat_reply(self, session, text, raw_prompt, events):
        """A chat turn's reply from its generate events."""
        token_ids, stats = [], None
        for chunk in events:
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
        if self._uses_rows():  # continuous batching: the request takes a row
            with self._lock:
                session = self._session(session_id)
            yield from self._scheduled(session, list(input_ids), max_new_tokens, params, extra_eos_ids)
            return
        with self._lock:
            session = self._session(session_id)
            yield from self._generate_locked(
                session, list(input_ids), max_new_tokens, params,
                extra_eos_ids, logits_sink, telemetry)

    def _native_rows(self, sampler=None):
        """Whether rows decode on the native loop (_NativeRows): the decode is the captured step
        graph, with steps of several tokens the drafter's too, and the request needs no repetition
        penalty (the graph samples by temperature, top-k and top-p: stepgraph.choose)."""
        from .stepgraph import StepGraph
        sampled = sampler is None or sampler.repetition_penalty == 1.0  # the graph samples, without the penalty
        drafter = self.step_tokens == 1 or (isinstance(self._draft_run, StepGraph) and self._draft_run.graph is not None)
        return (sampled and drafter and isinstance(self._decode_split, StepGraph)
                and self._decode_split.graph is not None)

    def _uses_rows(self):
        """Whether requests decode in rows (_Rows, through the scheduler): an artifact with several
        rows, or with decode steps of several tokens (speculative decoding)."""
        return self.binding.get("batch", 1) > 1 or self.step_tokens > 1

    def _scheduled(self, session, ids, max_new_tokens, params, extra_eos_ids=()):
        """A batch engine's generate: the request decodes in a row of the batch, alongside the
        others, and its events stream back as generate's do. Each request prefills its whole
        prompt (this model's prefill starts at position 0), so no cache carries across turns."""
        self._check_prompt(ids)
        generated = []
        for event in self._scheduler().submit(ids, max_new_tokens, params, extra_eos_ids):
            generated.extend(event["token_ids"])
            yield event
        session["cached_ids"] = list(ids) + [t for t in generated if t not in self.eos_ids]

    def generate_rows(self, prompts, max_new_tokens, params, arrivals=None, telemetry=False, batches=1,
                      batch_of=None):
        """Decode several prompts together, each in its own row of the batch: continuous batching,
        run to completion, as the scheduler runs it for concurrent requests.

        An artifact exported with --batch B decodes B prompts at once. Each prompt is prefilled
        alone and then decodes in a row of its own. Prompts may have any lengths: every row sits
        at its own position.
        `arrivals` gives each prompt the decode step at which it joins (default: all at step 0);
        a prompt waits while every row is taken. Each prompt generates up to max_new_tokens
        tokens and stops at an end-of-turn id, which frees its row.

        Returns {"token_ids": a list per prompt, "prefill_ms": each prompt's prefill,
        "step_ms": each decode step's time, "rows_active": how many rows decoded at each step},
        and with telemetry=True also "telemetry": the decode's execution telemetry.
        """
        arrivals = list(arrivals) if arrivals is not None else [0] * len(prompts)
        requests = [_Request(prompt, max_new_tokens, params.resolve(self.generation_config), self.eos_ids)
                    for prompt in prompts]
        for request in requests:
            self._check_prompt(request.prompt)
        waiting = sorted(range(len(prompts)), key=lambda i: arrivals[i])
        # the duplicate-row check: a free row gets a copy of a decoding row (checks.Padding)
        pad = self.checks.padding(requests, self.binding.get("batch", 1), batches, batch_of, telemetry)
        if batches > 1:
            if telemetry or not all(self._native_rows(r.sampler) for r in requests):
                raise ValueError("batches in flight need the captured step graph's native loop, "
                                 "no telemetry and no repetition penalty")
            return self._generate_rows_pipeline(requests, waiting, arrivals, batches, batch_of, pad)
        if not telemetry and all(self._native_rows(r.sampler) for r in requests):
            return self._generate_rows_native(requests, waiting, arrivals, pad)
        step_ms, rows_active = [], []
        recorder = None
        with self._lock:
            rows = _Rows(self)
            if telemetry:
                recorder = Recorder(self.device, getattr(self, "_segment_info", None)
                                    if self._decode_split is not None else None)
                self.external.telemetry = recorder
                if self._decode_split is not None:
                    self._decode_split.set_recorder(recorder)
            try:
                with self.external_scope():
                    rows.bind()
                    step = 0
                    while waiting or rows.active():
                        row = rows.free_row()
                        while row is not None and waiting and arrivals[waiting[0]] <= step:
                            rows.admit(requests[waiting.pop(0)], row)
                            row = rows.free_row()
                        if pad is not None:  # a copy takes a row no real prompt wants now, and gives it back
                            due = bool(waiting) and arrivals[waiting[0]] <= step
                            shadow = pad.next([rows.requests[r] for r in rows.active()]) if row is not None and not due else None
                            if shadow is not None:
                                rows.admit(shadow, row)
                            elif row is None and due:
                                pad.make_room()
                        rows.emit()
                        if rows.active():
                            rows_active.append(len([r for r in rows.active() if rows.requests[r].shadow_of is None]))
                            step_ms.append(rows.step())
                            if recorder is not None:
                                recorder.step_ms.append(step_ms[-1])
                        step += 1
            finally:
                if recorder is not None:
                    self.external.telemetry = None
                    if self._decode_split is not None:
                        self._decode_split.set_recorder(None)
        out = {"token_ids": [r.token_ids for r in requests], "prefill_ms": [r.prefill_ms for r in requests],
               "step_ms": step_ms, "rows_active": rows_active,
               "drafted": rows.drafted, "accepted": rows.accepted, "padded_rows": len(pad.shadows) if pad else 0}
        if recorder is not None:
            recorder.phase = "decode"
            out["telemetry"] = recorder.result(plan_record(self.explain()))
        return out

    def _generate_rows_native(self, requests, waiting, arrivals, pad=None):
        """generate_rows on the native loop (_NativeRows): the loop steps by itself, and a prompt
        joins at the first boundary after the loop has run `arrivals[i]` steps (at once when no
        row decodes). `pad` (checks.Padding) fills a row no prompt wants with a copy of one."""
        with self._lock:
            rows = _NativeRows(self)
            try:
                with self.external_scope():
                    rows.bind()
                    while waiting or rows.requests:
                        row = rows.free_row()
                        while (row is not None and waiting
                               and (arrivals[waiting[0]] <= rows.loop.steps() or not rows.requests)):
                            rows.admit(requests[waiting.pop(0)], row)
                            row = rows.free_row()
                        if pad is not None:
                            due = bool(waiting) and (arrivals[waiting[0]] <= rows.loop.steps() or not rows.requests)
                            shadow = pad.next(list(rows.requests.values())) if row is not None and not due else None
                            if shadow is not None:
                                rows.admit(shadow, row)
                            elif row is None and due:
                                pad.make_room()
                        rows.pump(0.001 if waiting else 0.05)
                        if self.checks.reference_due(rows.loop.steps()):
                            self.checks.reference_native(rows.loop, self._decode_split, self._live_state(rows.state),
                                                         {q.row: q for q in rows.requests.values()})
                steps = rows.loop.take_steps()
                drafted, accepted = rows.loop.drafts()
            finally:
                rows.close()
        return {"token_ids": [r.token_ids for r in requests], "prefill_ms": [r.prefill_ms for r in requests],
                "step_ms": [ms for ms, _ in steps], "rows_active": self.checks.rows_active(steps, pad.shadows if pad else []),
                "drafted": drafted, "accepted": accepted, "native": True, "padded_rows": len(pad.shadows) if pad else 0}

    def expert_stats(self):
        """{layer: distinct experts selected, summed over every step graph's replays since its
        capture} (VXPU_EXPERT_STATS=1; None without it): the engine's own step graph, which
        single batches run on, plus every batch in flight's graph and drafter (_pipeline_graphs:
        since hero run 10 each batch runs on a graph of its own, so a reader of _decode_split
        alone sees nothing move in a pipeline run). Read before and after a run and divide the
        difference by the steps for the distinct experts a step (dev/hero/batches.py)."""
        graphs = [self._decode_split]
        for entry in getattr(self, "_pipeline_graphs", []):  # (state, decode graph, drafter graph or None)
            graphs.extend(entry[1:])
        return sum_expert_stats(graphs)

    def generate_batches(self, prompt_lists, max_new_tokens, params):
        """generate_rows with one batch per list in `prompt_lists`, all in flight at once (the
        pipeline across the layer ranges), returning one result dict per batch: that batch's
        token_ids and prefill_ms, its own step_ms, and its own drafted and accepted counts (each
        batch drafts and verifies on its own loop). The measurement harness
        (dev/hero/batches.py) calls this; a driver that wants every field at once calls
        generate_rows(batches=N, batch_of=...) itself."""
        prompts, batch_of = [], []
        for b, lst in enumerate(prompt_lists):
            prompts += list(lst)
            batch_of += [b] * len(lst)
        started = time.time()
        out = self.generate_rows(prompts, max_new_tokens, params, batches=len(prompt_lists), batch_of=batch_of)
        ended = time.time()
        results, at = [], 0
        for b, lst in enumerate(prompt_lists):
            k = len(lst)
            results.append({"token_ids": out["token_ids"][at:at + k], "prefill_ms": out["prefill_ms"][at:at + k],
                            "step_ms": list(out["batch_step_ms"][b]), "rows_active": [k] * len(out["batch_step_ms"][b]),
                            "drafted": out["batch_drafted"][b], "accepted": out["batch_accepted"][b],
                            "native": True, "start": started, "end": ended})
            at += k
        return results

    def _generate_rows_pipeline(self, requests, waiting, arrivals, batches, batch_of=None, pad=None):
        """generate_rows with `batches` batches in flight (_PipelineRows): batch b holds its own
        state and captured step graphs and decodes on a native loop of its own, on streams of its
        own, so the GPUs work on different batches' layer ranges at once. With a drafter, each
        batch drafts and verifies on its own. Prompt i, in arrival order, goes to batch i // rows.
        Returns, besides generate_rows' fields, "batch_step_ms": each batch's step times,
        "batch_rows": each prompt's batch, and "batch_drafted" and "batch_accepted": each
        batch's own counts (drafted and accepted are their sums)."""
        with self._lock:
            rows = _PipelineRows(self, batches)
            per_row = rows.rows
            try:
                with self.external_scope():
                    rows.bind()
                    assigned = {}
                    order = list(waiting)
                    for k, i in enumerate(order):  # prompt i's batch: given, else the next free chunk of rows
                        assigned[i] = min(k // per_row, batches - 1) if batch_of is None else int(batch_of[i])
                    while waiting or rows.active():
                        progressed = False
                        for b in range(batches):
                            queue = [i for i in waiting if assigned[i] == b]
                            row = rows.free_row(b)
                            while (row is not None and queue
                                   and (arrivals[queue[0]] <= rows.loops[b].steps() or not rows.requests[b])):
                                i = queue.pop(0)
                                waiting.remove(i)
                                rows.admit(requests[i], b, row)
                                progressed = True
                                row = rows.free_row(b)
                            if pad is not None:
                                due = bool(queue) and (arrivals[queue[0]] <= rows.loops[b].steps() or not rows.requests[b])
                                shadow = pad.next(list(rows.requests[b].values()), b) if row is not None and not due else None
                                if shadow is not None:
                                    rows.admit(shadow, b, row)
                                    progressed = True
                                elif row is None and due:
                                    pad.make_room(b)
                        rows.pump(0.001 if (waiting or progressed) else 0.02)
                        for b, loop in enumerate(rows.loops):
                            if self.checks.reference_due(loop.steps()):
                                self.checks.reference_native(loop, rows.graphs[b], rows.states[b],
                                                             {q.row: q for q in rows.requests[b].values()})
                                break
                steps = [loop.take_steps() for loop in rows.loops]
                drafts = [loop.drafts() for loop in rows.loops]
            finally:
                rows.close()
        flat = [ms for per in steps for ms, _ in per]
        shadows = pad.shadows if pad else []
        return {"token_ids": [r.token_ids for r in requests], "prefill_ms": [r.prefill_ms for r in requests],
                "step_ms": flat, "padded_rows": len(shadows),
                "rows_active": [n for b, per in enumerate(steps)
                                for n in self.checks.rows_active(per, [s for s in shadows if s.batch == b])],
                "batch_step_ms": [[ms for ms, _ in per] for per in steps],
                "batch_rows": [assigned[i] for i in range(len(requests))],
                "batch_drafted": [d for d, _ in drafts], "batch_accepted": [a for _, a in drafts],
                "drafted": sum(d for d, _ in drafts), "accepted": sum(a for _, a in drafts),
                "native": True, "batches": batches}

    def _rebind(self, state):
        """Point the split programs' state buffers (the decode split and the prefill split, which
        share them) at `state`'s tensors: what a step graph captured after this reads and writes,
        and what the next prefill writes. _bind copies a session into the one resident state the
        main step graph is bound to; batches in flight each need a state of their own bound to
        graphs of their own, which this does by re-pointing the attributes the segments read
        (dataflow._attr reads them by path each step)."""
        from . import dataflow
        state = {**state, **{alias: state[canonical] for alias, canonical in self.aliases.items()
                             if canonical in state}}
        modules = []
        for holder in (self._decode_split, self._prefill_run, self._draft_run):
            split = getattr(holder, "split", None)
            if isinstance(split, torch.nn.Module) and split not in modules:
                modules.append(split)
        for module in modules:
            for fqn, tensor in state.items():
                parent, name = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                try:
                    target = dataflow._attr(module, parent) if parent else module
                except AttributeError:
                    continue
                if name in target._buffers:
                    target._buffers[name] = tensor
                elif isinstance(target.__dict__.get(name), torch.Tensor):
                    setattr(target, name, tensor)

    def _pipeline_graph(self, state):
        """A captured step graph of the decode bound to `state`: a dataflow runner of its own
        over the shared split (its own inboxes and replicas), placed as the main one, run once
        eagerly and then captured, with the split re-pointed at `state` while it captures."""
        from . import dataflow
        from .stepgraph import StepGraph
        main = self._decode_split
        if not isinstance(main, StepGraph) or main.graph is None:
            raise RuntimeError("batches in flight: the decode is not a captured step graph")
        self._rebind(state)
        runner = dataflow.Runner(main.split, *main.vxpu_segments, device=self.device,
                                 graph=self._graph(main.split) if self.plan["layer_devices"] else None)
        runner.state = set(self.binding["state"])
        if self.plan["layer_devices"]:
            runner.place(self._host_placement(runner.graph))
        graph = StepGraph(runner, plan=self.plan)
        values = self._zero_inputs(runner)
        for _ in range(2):  # the first step eager (the inboxes allocated), the second captured
            graph(**values)
        return graph

    def _pipeline_draft(self, state):
        """The drafter's captured step graph bound to `state`, made as _pipeline_graph makes the
        decode's: a batch in flight drafts and verifies on its own, so its loop needs a drafter
        reading and writing its own caches (the drafter's split is one of the programs _rebind
        re-points). None when the engine has no captured drafter."""
        from . import dataflow
        from .stepgraph import StepGraph
        main = self._draft_run
        if not isinstance(main, StepGraph) or main.graph is None:
            return None
        self._rebind(state)
        runner = dataflow.Runner(main.split, *main.vxpu_segments, device=self.device,
                                 graph=self._graph(main.split) if self.plan["layer_devices"] else None)
        runner.state = set(self.binding["state"])
        if self.plan["layer_devices"]:
            runner.place(self._host_placement(runner.graph, "draft"))
        # captured with the drafter's GPU current: the step's inputs and its one graph live
        # there, with nothing copied in from the engine's GPU (the capture takes the current
        # device as the step's own, and the engine's drafter was captured on the engine's)
        devices = {self._layer_device(max(int(x) for x in n.layers)) for n in runner.graph.nodes if n.layers}
        devices.discard(None)
        current = devices.pop() if len(devices) == 1 else self.device
        graph = StepGraph(runner, plan=self.plan, role="draft")
        with torch.cuda.device(current):
            values = self._zero_inputs(runner)
            for _ in range(2):
                graph(**values)
        return graph

    def _check_prompt(self, prompt):
        if not self.min_prefill_len <= len(prompt) <= self.max_cache_len - 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens: the prefill graph takes "
                             f"{self.min_prefill_len} to {self.max_cache_len - 1}")

    def _scheduler(self):
        """The scheduler that decodes this batch engine's requests (started on first use)."""
        with self._lock:
            if getattr(self, "_rows_scheduler", None) is None:
                self._rows_scheduler = _Scheduler(self)
            return self._rows_scheduler

    def _live_state(self, state):
        """The state tensors the graphs read and write right now: the resident state when the
        compiled split keeps one bound (see _bind), else `state` itself."""
        return self._resident if self._resident is not None else state

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
            check = self.checks.reference_due(step + 1)
            if check:
                self.checks.snapshot(self._live_state(session["state"]))
            logits = self._decode_run(
                input_ids=torch.tensor([[next_id]], device=self.device),
                cache_position=torch.tensor([len(cached)],
                                            device=self.device))
            if check:
                self.checks.reference_step(torch.tensor([[next_id]]), torch.tensor([len(cached)]), [0], logits=logits,
                                           steps=step + 1)
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


class _Request:
    """One prompt decoding in a row of a batch engine (continuous batching)."""

    def __init__(self, prompt, max_new_tokens, sampler, eos_ids, sink=None):
        self.prompt, self.max_new_tokens = list(prompt), max_new_tokens
        self.sampler, self.eos_ids = sampler, eos_ids
        self.sink = sink  # receives each event: the scheduler's queue, or None in generate_rows
        self.token_ids, self.finish_reason, self.cancelled = [], None, False
        self.prefill_ms = 0.0
        self.first_token_at = self.done_at = None
        # the duplicate-row check (checks.Padding): a copy points at its original, an original lists
        # its copies, and both name the Checks that compares them as they emit
        self.shadow_of, self.shadows, self.checks = None, [], None
        self.row = self.batch = 0  # where it decodes, for the check's messages
        self.admitted_at = self.finished_at = None  # the loop's step count when it joined and left

    def emit(self, token_id):
        self.token_ids.append(token_id)
        if self.checks is not None:
            self.checks.row_token(self)
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        if self.sink is not None:
            self.sink({"token_ids": [token_id], "done": False})

    def finish(self, reason):
        self.finish_reason, self.done_at = reason, time.perf_counter()
        if self.sink is not None:
            n = len(self.token_ids)
            self.sink({"token_ids": [], "done": True, "finish_reason": reason,
                       "prompt_tokens": len(self.prompt), "prefilled_tokens": len(self.prompt),
                       "generated": n, "prefill_ms": round(self.prefill_ms),
                       "ms_per_token": round((self.done_at - self.first_token_at) / max(n - 1, 1) * 1e3, 1)
                       if self.first_token_at is not None else 0.0})

    def fail(self, error):
        self.finish_reason = "error"
        if self.sink is not None:
            self.sink({"token_ids": [], "done": True, "error": error})


def _move_module_tensors(module, device):
    """Give a segment module a copy on device of every tensor it holds itself (buffers,
    parameters, and plain tensor attributes such as export's lifted constants), following a
    compiled wrapper's `sub`. A copy, not a move: a constant can be shared by two segments on
    different GPUs, and moving it in place would pull it from under the other. These are the
    small tensors; the weights a segment reads come as arguments and move in place. Also points
    the segment graph's tensor factories (arange, zeros: the exported graphs name "cuda", which
    torch.export reads as cuda:0 whatever the current device) at device. Returns the bytes
    copied."""
    moved = 0
    seen = set()
    stack = [module]
    while stack:
        m = stack.pop()
        if id(m) in seen or not isinstance(m, torch.nn.Module):
            continue
        seen.add(id(m))
        graph = getattr(m, "graph", None)
        if isinstance(graph, torch.fx.Graph):
            changed = False
            for node in graph.nodes:
                d = node.kwargs.get("device")
                if isinstance(d, (str, torch.device)) and torch.device(d).type == "cuda" and torch.device(d) != device:
                    node.kwargs = {**node.kwargs, "device": device}
                    changed = True
            if changed:
                m.recompile()
        for name, t in list(m.named_buffers(recurse=False)):
            if t.is_cuda and t.device != device:
                m._buffers[name] = t.to(device)
                moved += t.numel() * t.element_size()
        for name, t in list(m.named_parameters(recurse=False)):
            if t.is_cuda and t.device != device:
                m._parameters[name] = torch.nn.Parameter(t.detach().to(device), requires_grad=False)
                moved += t.numel() * t.element_size()
        for name, v in list(vars(m).items()):
            if isinstance(v, torch.Tensor) and v.is_cuda and v.device != device:
                setattr(m, name, v.to(device))
                moved += v.numel() * v.element_size()
        stack.extend(m.children())
        sub = getattr(m, "sub", None)
        if isinstance(sub, torch.nn.Module):
            stack.append(sub)
    return moved



def _prefill_row(engine, state, request, row, row0_decodes, live=None):
    """Prefill request alone in row 0 (the prefill graph has one row), then move its state to
    `row`. When row 0 decodes another request (row0_decodes), its state is set aside for the
    prefill and put back. `live` names the state tensors the prefill writes (a batch in flight's
    own; default: the resident state, see _live_state). Returns the prefill's logits."""
    e = engine
    started = time.perf_counter()
    live = e._live_state(state) if live is None else live
    kept = {fqn: tensor[0].clone() for fqn, tensor in live.items()} if row != 0 and row0_decodes else None
    logits = e._prefill_run(input_ids=torch.tensor([request.prompt], dtype=torch.long, device=e.device),
                            cache_position=torch.arange(0, len(request.prompt), device=e.device))
    if row != 0:
        for fqn, tensor in live.items():
            tensor[row].copy_(tensor[0])
            if kept is not None:
                tensor[0].copy_(kept[fqn])
    # the shared caches' replicas on other GPUs take the rows the prefill wrote: on the step graph
    # of the batch that owns this state (a batch in flight has its own state and graphs; the
    # engine's own graph is batch 0's)
    graphs = [e._decode_split, e._draft_run]
    for entry in getattr(e, "_pipeline_graphs", []):  # (state, decode graph, drafter graph or None)
        if entry[0] is state:
            graphs = list(entry[1:])
            break
    for graph in graphs:
        refresh = getattr(graph, "refresh_rows", None)
        if refresh is not None:
            refresh({0, row})
    request.prefill_ms = (time.perf_counter() - started) * 1e3
    return logits


class _Rows:
    """A batch engine's rows (continuous batching): each row decodes one request at its own
    position. A prompt is prefilled alone, in row 0 (the prefill graph's one row), and then moved
    to its own row. A row with nothing to decode feeds its last token again; whatever that writes,
    the row's next prefill replaces, so a step needs no masking. Used with the engine's lock held.

    With more than one token per step (speculative decoding), each row feeds its next token and
    the drafter's tokens after it. The row keeps each draft that equals the token it would have
    picked there, up to the first that does not, and then the token it picked in that place. Its
    sampler picks every token, so the output is what one token per step would produce. A rejected
    draft leaves values in the caches past the row's position; the next step overwrites them (the
    exporter gives the window rings the slack for that)."""

    def __init__(self, engine):
        self.engine = engine
        self.batch = engine.binding.get("batch", 1)
        self.state = {fqn: torch.zeros(shape, dtype=dtype, device=engine.device)
                      for fqn, (shape, dtype) in engine.state_specs.items()}
        self.requests = [None] * self.batch  # the request each row decodes
        self.fed = [[] for _ in range(self.batch)]  # each row's tokens in its caches, in order
        # each row's tokens to emit; the last is its next token, not yet fed
        self.pending = [[] for _ in range(self.batch)]
        self.drafter = None
        if engine.step_tokens > 1:
            self.drafter = _ModelDrafter(engine) if engine._draft_run is not None else _LookupDrafter()
        self.drafted = self.accepted = 0
        self.steps_done = 0

    def bind(self):
        self.engine._bind(self.state)
        self.engine._current = {"id": "rows", "state": self.state}

    def free_row(self):
        return next((row for row in range(self.batch) if self.requests[row] is None), None)

    def active(self):
        return [row for row in range(self.batch) if self.requests[row] is not None]

    def admit(self, request, row):
        """Prefill request in row 0, then move its state to `row` (_prefill_row)."""
        logits = _prefill_row(self.engine, self.state, request, row, self.requests[0] is not None)
        self.requests[row], self.fed[row] = request, list(request.prompt)
        request.row = row
        self.pending[row] = [request.sampler.pick(logits[0, -1], self.fed[row])]

    def emit(self):
        """Each decoding row emits its pending tokens; a request that stops, or whose caller has
        gone, frees its row."""
        for row in self.active():
            request = self.requests[row]
            if request.cancelled:
                self.requests[row] = None
                continue
            for token_id in self.pending[row]:
                request.emit(token_id)
                if token_id in request.eos_ids:
                    request.finish("eos")
                elif (len(request.token_ids) >= request.max_new_tokens
                      or len(self.fed[row]) + self.engine.step_tokens > self.engine.max_cache_len):
                    request.finish("length")
                if request.finish_reason is not None:
                    self.requests[row] = None
                    break
            self.pending[row] = self.pending[row][-1:]

    def step(self):
        """One decode step for every row; returns its time in ms."""
        e = self.engine
        ids, pos = [], []
        for row in range(self.batch):
            if self.requests[row] is not None:  # feed its next token
                ids.append(self.pending[row][-1])
                pos.append(len(self.fed[row]))
            elif self.fed[row]:  # replay its last token, rewriting the same values
                ids.append(self.fed[row][-1])
                pos.append(len(self.fed[row]) - 1)
            else:  # never used
                ids.append(0)
                pos.append(0)
        started = time.perf_counter()
        position = torch.tensor(pos, device=e.device)
        tokens = [[t] for t in ids]
        if self.drafter is not None:
            drafts = self.drafter.propose(self, ids, position, e.step_tokens - 1)
            tokens = [[t] + d for t, d in zip(ids, drafts)]
        self.steps_done += 1
        check = e.checks.reference_due(self.steps_done)  # the reference check: the state before the step, then the step
        if check:
            e.checks.snapshot(e._live_state(self.state))
        logits = e._decode_run(input_ids=torch.tensor(tokens, device=e.device), cache_position=position)
        if check:
            e.checks.reference_step(torch.tensor(tokens, device=e.device), position, self.active(), logits=logits,
                                    steps=self.steps_done)
        for row in self.active():
            request, fed, kept = self.requests[row], self.fed[row], []
            for j, token_id in enumerate(tokens[row]):
                fed.append(token_id)
                picked = request.sampler.pick(logits[row, j], fed)
                kept.append(picked)
                if j + 1 == len(tokens[row]) or tokens[row][j + 1] != picked:
                    break
            self.drafted += len(tokens[row]) - 1
            self.accepted += len(kept) - 1
            self.pending[row] = kept
        return (time.perf_counter() - started) * 1e3


class _NativeRows:
    """_Rows' work with no Python between decode steps. The step graph's native loop
    (StepGraph.native_loop, node_server.cpp's DecodeLoop) runs every step and chooses each row's
    tokens in the graph, by its sampling settings; with steps of several tokens it also runs the drafter's step
    graph and keeps the drafts the choices agree with. Python admits requests between steps (the
    loop paused for the prefill) and hands the emitted tokens to their requests. For an engine
    whose decode is the captured step graph (Engine._native_rows)."""

    FINISH = {1: "eos", 2: "length"}

    def __init__(self, engine):
        self.engine = engine
        self.batch = engine.binding.get("batch", 1)
        self.state = {fqn: torch.zeros(shape, dtype=dtype, device=engine.device)
                      for fqn, (shape, dtype) in engine.state_specs.items()}
        self.loop = None
        self.requests = {}  # id(request) -> request, while its row decodes

    def bind(self):
        e = self.engine
        e._bind(self.state)
        e._current = {"id": "rows", "state": self.state}
        if self.loop is None:
            self.loop = e._decode_split.native_loop(draft=e._draft_run if e.step_tokens > 1 else None)

    def free_row(self):
        free = self.loop.free_rows()
        return free[0] if free else None

    def admit(self, request, row):
        """Prefill request into `row` with the loop held between steps, emit its first token,
        and hand the row to the loop (unless that token already ends it)."""
        e = self.engine
        self.loop.pause()
        try:
            logits = _prefill_row(e, self.state, request, row, 0 not in self.loop.free_rows())
            first = request.sampler.pick(logits[0, -1], list(request.prompt))
            request.emit(first)
            if first in request.eos_ids:
                request.finish("eos")
            elif request.max_new_tokens <= 1 or len(request.prompt) + e.step_tokens > e.max_cache_len:
                request.finish("length")
            else:
                self.requests[id(request)] = request
                request.row, request.admitted_at = row, self.loop.steps()
                sampler = request.sampler  # its settings, in the graph (stepgraph.choose)
                self.loop.admit(row, id(request), first, len(request.prompt), 1, request.max_new_tokens,
                                sorted(request.eos_ids), e.max_cache_len,
                                sampler.temperature if sampler.do_sample else 0.0, sampler.top_k, sampler.top_p)
        finally:
            self.loop.resume()

    def pump(self, timeout):
        """Hand the loop's emitted tokens to their requests, waiting up to `timeout` s for some,
        and stop the requests whose callers left. Returns how many requests still decode."""
        for rid, request in list(self.requests.items()):
            if request.cancelled:
                self.loop.cancel(rid)
        for rid, tokens, finish, error in self.loop.events(timeout):
            request = self.requests.get(rid)
            if request is None:
                continue
            for token_id in tokens:
                request.emit(token_id)
            if finish:
                del self.requests[rid]
                request.finished_at = self.loop.steps()
                if finish in self.FINISH:
                    request.finish(self.FINISH[finish])
                elif error:
                    request.fail(RuntimeError(error))
        return len(self.requests)

    def close(self):
        if self.loop is not None:
            self.loop.stop()


class _PipelineRows:
    """Several batches in flight (hero run 9). Every batch gets a state of its own (zeros shaped
    and placed like the resident tensors) and step graphs captured against it
    (Engine._pipeline_graph); the engine's own graph, bound to the resident state, stays out
    (see bind).
    Each batch decodes on a native loop of its own, launching on streams of its own per GPU: with
    the layers sharded, the GPUs then run different batches' ranges at once, a batch's graph on
    GPU k waiting (a stream memory wait) for its own range k-1 while another batch's range k runs.
    Python admits requests between a batch's steps (that batch's loop paused for the prefill,
    the others stepping) and hands the emitted tokens to their requests. With a drafter (steps of
    several tokens), each batch has a drafter step graph of its own too, captured against its
    state (Engine._pipeline_draft), and its loop drafts, verifies and counts acceptance on its
    own, each on its own drafter (the engine's own step and drafter graphs serve single batches)."""

    FINISH = {1: "eos", 2: "length"}

    def __init__(self, engine, batches):
        e = engine
        self.engine = e
        self.n = batches
        self.rows = e.binding.get("batch", 1)
        if e._resident is None:
            raise RuntimeError("batches in flight need the resident state of a split decode")
        self.states = []
        self.graphs = []
        self.drafts = []
        self.loops = []
        self.requests = [{} for _ in range(batches)]
        self._streams = []

    def bind(self):
        """Inside the engine's external scope (the captures resolve the experts and tables through
        it): every batch's state and step graphs, then a loop per batch. Batch 0 does not run on
        the engine's own graph (the one the warm-up captured on the resident state): beside
        another batch's graph that one produced other tokens at nine rows of different prompts
        (hero run 10; its own graph captured by _pipeline_graph never did), so each batch gets a
        state and graph made the same way."""
        e = self.engine
        started = time.perf_counter()
        cache = e.__dict__.setdefault("_pipeline_graphs", [])  # (state, graph, drafter) per batch, kept across calls
        built = 0
        for b in range(self.n):
            if len(cache) <= b:
                state = {fqn: torch.zeros_like(t) for fqn, t in e._resident.items()}
                graph = e._pipeline_graph(state)
                cache.append((state, graph, e._pipeline_draft(state) if e.step_tokens > 1 else None))
                built += 1
            state, graph, draft = cache[b]
            self.states.append(state)
            self.graphs.append(graph)
            self.drafts.append(draft)
        nbytes = sum(t.numel() * t.element_size() for t in e._resident.values())
        print(f"[vxpu] batches in flight: {self.n}, each {self.rows} rows; state {nbytes / 1e9:.2f} GB a batch; "
              f"{built} step graphs captured in {time.perf_counter() - started:.1f} s", flush=True)
        trace = os.environ.get("VXPU_STEP_TRACE")
        for b, (graph, draft) in enumerate(zip(self.graphs, self.drafts)):
            devices = set(graph.native_parts()[4])
            if draft is not None:  # the batch's drafter launches on the batch's streams too
                devices |= set(draft.native_parts()[4])
            streams = {d: torch.cuda.Stream(device=d) for d in sorted(devices)}
            self._streams.append(streams)
            if trace:  # the step trace per batch: the loop opens its file when it is made
                os.environ["VXPU_STEP_TRACE"] = f"{trace}.b{b}"
            self.loops.append(graph.native_loop(draft=draft, streams=streams))
        if trace:
            os.environ["VXPU_STEP_TRACE"] = trace
        e._current = {"id": "pipeline", "state": e._resident}

    def active(self):
        return any(self.requests)

    def free_row(self, b):
        free = self.loops[b].free_rows()
        return free[0] if free else None

    def admit(self, request, b, row):
        """Prefill request into row `row` of batch b, with that batch's loop held between steps
        (the others keep stepping), emit its first token, and hand the row to the loop."""
        e = self.engine
        loop = self.loops[b]
        loop.pause()
        try:
            e._rebind(self.states[b])  # the prefill writes this batch's caches
            logits = _prefill_row(e, self.states[b], request, row, 0 not in loop.free_rows(), live=self.states[b])
            first = request.sampler.pick(logits[0, -1], list(request.prompt))
            request.emit(first)
            if first in request.eos_ids:
                request.finish("eos")
            elif request.max_new_tokens <= 1 or len(request.prompt) + e.step_tokens > e.max_cache_len:
                request.finish("length")
            else:
                self.requests[b][id(request)] = request
                request.row, request.batch, request.admitted_at = row, b, loop.steps()
                sampler = request.sampler
                loop.admit(row, id(request), first, len(request.prompt), 1, request.max_new_tokens,
                           sorted(request.eos_ids), e.max_cache_len,
                           sampler.temperature if sampler.do_sample else 0.0, sampler.top_k, sampler.top_p)
        finally:
            loop.resume()

    def pump(self, timeout):
        """Hand every loop's emitted tokens to their requests. Returns how many requests still decode."""
        left = 0
        for b, loop in enumerate(self.loops):
            requests = self.requests[b]
            for rid, request in list(requests.items()):
                if request.cancelled:
                    loop.cancel(rid)
            for rid, tokens, finish, error in loop.events(timeout / max(1, len(self.loops))):
                request = requests.get(rid)
                if request is None:
                    continue
                for token_id in tokens:
                    request.emit(token_id)
                if finish:
                    del requests[rid]
                    request.finished_at = loop.steps()
                    if finish in self.FINISH:
                        request.finish(self.FINISH[finish])
                    elif error:
                        request.fail(RuntimeError(error))
            left += len(requests)
        return left

    def close(self):
        for loop in self.loops:
            loop.stop()
        self.engine._rebind(self.engine._resident)  # the split points at the resident state again


class _ModelDrafter:
    """Drafts from the model's own drafter (draft.pt2): each row's next token, at its position,
    in; the drafter's block of tokens after it out."""

    def __init__(self, engine):
        self.engine = engine

    def propose(self, rows, ids, position, count):
        e = rows.engine
        drafts = e._draft_run(input_ids=torch.tensor([[t] for t in ids], device=e.device), cache_position=position)
        return [d[:count] for d in drafts.tolist()]


class _LookupDrafter:
    """Prompt lookup, for any model: a row's drafts are the tokens that followed the last earlier
    occurrence of its latest tokens (the longest match of up to `ngram` tokens) in its own prompt
    and output. Without a match, it repeats the next token, which costs nothing extra to verify."""

    def __init__(self, ngram=3):
        self.ngram = ngram

    def propose(self, rows, ids, position, count):
        out = []
        for row, next_id in enumerate(ids):
            context = rows.fed[row] + [next_id] if rows.requests[row] is not None else []
            out.append(self._lookup(context, count) or [next_id] * count)
        return out

    def _lookup(self, context, count):
        for n in range(min(self.ngram, len(context) - 1), 0, -1):
            tail = context[-n:]
            for start in range(len(context) - n - 1, -1, -1):
                if context[start:start + n] == tail:
                    follow = context[start + n:start + n + count]
                    return follow + [follow[-1]] * (count - len(follow))
        return None


class _Scheduler:
    """Continuous batching for a batch engine's concurrent requests: a thread steps every decoding
    row together, and admits waiting requests into free rows between steps. With the step graph,
    the native loop steps the rows (_NativeRows) and this thread only admits requests and hands
    out their tokens."""

    def __init__(self, engine):
        self.engine = engine
        native = engine._native_rows()
        self.rows = _NativeRows(engine) if native else _Rows(engine)
        self.waiting = collections.deque()
        self.ready = threading.Condition()
        threading.Thread(target=self._native_loop if native else self._loop, name="vxpu-rows", daemon=True).start()

    def submit(self, prompt, max_new_tokens, params, extra_eos_ids=()):
        """Queue a request; yields its events as they come, as Engine.generate does."""
        events = queue.Queue()
        e = self.engine
        request = _Request(prompt, max_new_tokens, params.resolve(e.generation_config),
                           e.eos_ids | set(int(i) for i in extra_eos_ids), sink=events.put)
        with self.ready:
            self.waiting.append(request)
            self.ready.notify()
        try:
            while True:
                event = events.get()
                if "error" in event:
                    raise event["error"]
                yield event
                if event["done"]:
                    return
        finally:  # a caller that stops listening frees its row at the next step
            request.cancelled = True

    def _loop(self):
        while True:
            with self.ready:
                while not self.waiting and not self.rows.active():
                    self.ready.wait()
            try:
                with self.engine._lock, self.engine.external_scope():
                    self.rows.bind()
                    row = self.rows.free_row()
                    while row is not None:
                        with self.ready:
                            request = self.waiting.popleft() if self.waiting else None
                        if request is None:
                            break
                        if not request.cancelled:
                            self.rows.admit(request, row)
                        row = self.rows.free_row()
                    self.rows.emit()
                    if self.rows.active():
                        self.rows.step()
            except Exception as error:  # noqa: BLE001  (every decoding request fails with it)
                for row in self.rows.active():
                    self.rows.requests[row].fail(error)
                    self.rows.requests[row] = None

    def _native_loop(self):
        while True:
            with self.ready:
                while not self.waiting and not self.rows.requests:
                    self.ready.wait()
            try:
                with self.engine._lock, self.engine.external_scope():
                    self.rows.bind()
                    row = self.rows.free_row()
                    while row is not None:
                        with self.ready:
                            request = self.waiting.popleft() if self.waiting else None
                        if request is None:
                            break
                        if request.cancelled:
                            continue
                        if not self.engine._native_rows(request.sampler):
                            request.fail(NotImplementedError(
                                "the native decode loop samples in the graph, without a repetition penalty"))
                            continue
                        self.rows.admit(request, row)
                        row = self.rows.free_row()
                self.rows.pump(0.002 if self.waiting else 0.05)
            except Exception as error:  # noqa: BLE001  (every decoding request fails with it)
                for request in self.rows.requests.values():
                    request.fail(error)
                self.rows.requests.clear()


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

