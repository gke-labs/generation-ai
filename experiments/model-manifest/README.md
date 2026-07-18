# model-manifest

Experiment: a content-addressed interchange format for models — a thin
client that can describe, capture, and ship a model **without ever
touching its weights**, and an executor that rehydrates and runs it.

The artifact has three parts:

1. **Manifest** (stage A, `src/manifest.py`) — architecture config plus a
   table binding every tensor to the 5-tuple
   `(file_sha256, byte_offset, byte_length, dtype, shape)`.
   Built entirely from Hub metadata: per-file sha256 (computed at upload,
   served by the API) and range-fetched safetensors headers. The sha256
   of the manifest is the model's digest, like an OCI image digest.
2. **Weightless graph** (stage B, `src/export_graph.py`) — the model is
   instantiated on the `meta` device (shapes, no storage) and captured
   with `torch.export`. Lifted parameter FQNs are joined against the
   manifest: each graph input is `bound` to a content reference,
   `derived` from config (RoPE frequencies, tied embeddings resolve as
   aliases), or `unbound` (an error).
3. **Executor** (stage C, `src/executor.py`) — fetches each tensor with
   one HTTP range request into a local content-addressed cache (the
   stand-in for an OCI registry), computes derived buffers, injects
   everything into the exported program, retargets it off the meta
   device (`move_to_device_pass`), and runs it.

## Measured results

| step | cost |
|---|---|
| manifest of SmolLM2-135M (0.27 GB weights) | 31 KB fetched |
| manifest of DeepSeek-R1 (689 GB, 91,991 tensors) | 11.9 MB fetched, 0 weight bytes |
| weightless export of SmolLM2 (2,223 nodes) | 5.2 MB `.pt2` artifact |
| executor, cold cache | 269 MB ranged reads, 42 s |
| executor, warm cache | 0 bytes, 1.6 s to a running model |
| logits vs `from_pretrained` | max diff 0.00e+00 (bitwise identical) |

Free extras that fell out: the export's `nn_module_stack` metadata
recognizes the architecture (30 llama-style decoder layers); the tied
`lm_head` deduplicates in the cache because aliases share a content key;
the binding is statically checkable against the manifest before any
download.

## Run

```sh
uv sync
uv run python src/manifest.py HuggingFaceTB/SmolLM2-135M-Instruct
uv run python src/manifest.py deepseek-ai/DeepSeek-R1   # thin client scales
uv run python src/export_graph.py HuggingFaceTB/SmolLM2-135M-Instruct \
    --manifest HuggingFaceTB--SmolLM2-135M-Instruct.manifest.json
uv run python src/executor.py \
    --manifest HuggingFaceTB--SmolLM2-135M-Instruct.manifest.json --verify
```

## Running on Kubernetes

```sh
dev/tasks/run-in-kube            # build via Cloud Build, run on CPU + GPU pods
dev/tasks/run-in-kube --target gpu --image gcr.io/.../manifest-executor:v2
```

The thin client ships only manifest + graph + binding (~5.3 MB via
`kubectl cp` for now; OCI artifacts later) to an idle executor pod; the
pod range-fetches weights straight from the Hub into its own CAS.
Measured on GKE (same artifact exported on the arm64 Mac):

- CPU pod (c3d-standard): logits **bitwise identical** to
  from_pretrained (max diff 0.00e+00) — across OS and CPU architecture.
- GPU pod (L4, `--device cuda`): argmax match, max |diff| 5.6e-01 from
  cross-device bf16 kernels — the honest, expected numerics gap.

GKE notes: pip torch needs `LD_LIBRARY_PATH=/usr/local/nvidia/lib64`
(driver mount), and torch 2.13 JIT-compiles some eager CUDA ops via
Triton, so the image needs gcc.

## KV-cached generation (StaticCache + compile-on-arrival)

`src/export_cached.py` exports two weightless graphs via transformers'
export-friendly wrapper: `prefill.pt2` (dynamic sequence length) and
`decode.pt2` (constant shapes, StaticCache buffers as mutable state).
The binding gains a third category: `state` — cache tensors the
executor zero-initializes (60 KV buffers + 30 write counters).

`src/generator.py` rehydrates both graphs (sharing one cache between
them), `torch.compile`s the decode step on arrival, and runs the token
loop. Answering "Why is the sky blue?" (correctly — Rayleigh
scattering):

| where | mode | steady ms/token | prepare |
|---|---|---|---|
| Mac CPU | eager | 21.3 (46.9 tok/s) | — |
| Mac CPU | compiled | 16.9 (59.1 tok/s) | 17 s once |
| L4 GPU | eager | 45.2 (22.1 tok/s) | — |
| L4 GPU | compiled | **9.5 (105.5 tok/s)** | 41 s once |

Constant shapes mean exactly one compile; every later token reuses it.
(GPU-eager being slower than CPU-eager for a 135M model is real:
per-op launch overhead dominates tiny kernels — which is precisely
what compilation removes.)

Note: compiled modules need `_assert_tensor_metadata` guard nodes
stripped first (`strip_asserts`) — Dynamo cannot re-trace them on CUDA,
and `ep.module()` regenerates some inside submodules.

## Long-lived serving (sessions)

`src/engine.py` is a transport-agnostic serving core: weights fetched
once and shared read-only; each session gets fresh zero-init `state`
tensors (23.6 MB here) and its own module pair. `src/serve.py` wraps it
in a Monarch actor (a ~30-line shim — the engine has no Monarch in it,
so a gRPC servicer could replace the shim unchanged; Monarch is under
evaluation, not assumed).

Multi-turn conversation falls out of the graph design: the prefill
graph's dynamic sequence dimension + explicit cache positions mean a
follow-up turn prefills only the new suffix tokens (measured: turn two
prefilled 22 new tokens in 61 ms instead of re-processing 154). Session
isolation and memory verified: session A correctly summarizes its own
previous answer; session B never sees A's conversation.

```sh
PYTHONPATH=src uv run python src/serve.py            # two-session demo
```

## PagedAttention probe (`src/paged_probe.py`)

Can torch's experimental paged KV (FlexAttention + page table) survive
graph capture? Measured, with pages deliberately scattered out of
physical order:

| question | result |
|---|---|
| eager paged == dense SDPA | yes — 2.4e-07, paging does not change the math |
| torch.compile (CUDA, L4) | yes — needs page_size to satisfy Triton block-size constraints (128 works, 16 doesn't: `NoValidChoicesError`); macOS CPU has no flex lowering at all |
| torch.export, in memory | **yes** — 79 nodes, runs correctly via `ep.module()` |
| torch.export.save to .pt2 | **no** — `SerializeError`: the flex_attention HOP's BlockMask arguments aren't serializable yet |

Interpretation for the interchange format: paging *semantics* survive
capture (pool + page table + mask tensors are all just `state`-category
buffers), but the serialization schema hasn't caught up with the
FlexAttention higher-order op. Until it does, paged executors are
reachable via compile-on-arrival from a recipe, not via a shipped
`.pt2` — i.e. attention remains a *contract op* at the artifact
boundary, exactly the seam this experiment predicted.

## Known simplifications (v0)

- Artifact transport is `kubectl cp` into an idle pod; the intended
  replacement is OCI artifacts (ORAS push/pull of manifest + graphs,
  per-tensor layers).
- Range verification: a fetched range cannot be checked against the
  file-level sha256 alone. The enrichment path is per-tensor hashes (the
  4-tuple), computed once at publish time — turning each tensor into its
  own "docker layer" for a real OCI registry.
- Graph dtype is frozen at export (bf16 here); quantized variants are
  derivations with their own manifests.
- `.pt2` artifacts are pinned to the exporting torch version (the
  serialization schema promises newer-loads-older, untested by us);
  the manifest should carry `torch_version` explicitly.
- One model (SmolLM2-135M) exercised end-to-end; DeepSeek-R1 is
  manifested but not executed. Sliding-window architectures (Gemma)
  will need mask-aware handling in the cached export.
