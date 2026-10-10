# vXPU: run PyTorch models on remote accelerators

vXPU lets you take a PyTorch model, export it on a machine with **no
GPU, no CUDA, and no downloaded weights**, and run it on a Kubernetes
cluster that has the accelerators — shipping only a small, portable,
content-addressed artifact.

```sh
# 1. Export (thin client: meta device, no weights ever downloaded)
python -m vxpu.export google/gemma-4-E4B-it -o gemma-e4b/

# 2. Ask (creates the router + executor pods on demand, ships the
#    artifact, weights rehydrate on the executor from content-addressed refs)
vxpu ask --artifact gemma-e4b/ "Is the sky blue?"
```

From Python, transformers code runs unchanged except for one import:

```python
from transformers import AutoTokenizer
from vxpu import AutoModelForCausalLM   # instead of transformers

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
inputs = tokenizer("Is the sky blue?", return_tensors="pt")
out = model.generate(inputs.input_ids, max_new_tokens=30)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
```

`from_pretrained` exports (meta device, cached under `~/.cache/vxpu`),
ships, and waits; `generate` sends token ids and gets token ids back,
with the tokenizer, chat template, tools and `TextStreamer` being the
ordinary transformers objects. The lower-level `vxpu.client.Client`
offers the same over text (`chat`) or ids (`generate_ids`).

See [examples/gemma4](examples/gemma4/) for a notebook that runs the
Hugging Face Gemma 4 docs examples against `gemma-4-31B-it` from a
machine with no GPU.

The exported artifact for a 16 GB model is ~60 MB. For a 689 GB model
it is ~25 MB of manifest — the artifact size scales with the
*architecture*, not the weights.

## How it works

```
 thin client (laptop, CI)                 executor pod (GPU/CPU node)
┌──────────────────────────────┐  gRPC   ┌──────────────────────────────┐
│ meta-device instantiation    │ ──────▶ │ verify + cache weights by    │
│ torch.export (prefill+decode)│ artifact│   content hash (range reads) │
│ manifest: tensor →           │  ~60MB  │ recompute derived tensors    │
│   (sha256, offset, len,      │         │ zero-init session state      │
│    dtype, shape)             │ ◀────── │ torch.compile decode once    │
│ chat over sessions           │  tokens │ run the token loop           │
└──────────────────────────────┘         └──────────────────────────────┘
```

Three ideas carry the design:

1. **Weightless export.** torch's meta device instantiates a model's
   shapes without storage, and `torch.export` captures executable
   graphs from it. A laptop can export a model of any size. Two graphs
   are captured: a dynamic-length `prefill` and a constant-shape
   `decode` — constant shapes mean the executor compiles it exactly
   once and reuses it for every token.

2. **Content-addressed weights.** The manifest binds every tensor to
   `(file_sha256, offset, length, dtype, shape)`, built entirely from
   repository metadata (the Hub serves per-file sha256; safetensors
   headers are range-fetched). Executors pull each tensor with one
   HTTP range request into a local cache: cold loads stream, warm
   loads download nothing, and the sha256 of the manifest is the
   model's identity — `LoadModel` with a matching digest is a no-op.

3. **A three-way tensor classification.** Every tensor the graphs
   reference is `bound` (a weight reference), `derived` (computed from
   config at load time: RoPE tables via transformers' own formula
   registry, embedding scales — never stored in checkpoints), or
   `state` (per-session KV cache, zero-initialized, mutated in place
   by the graphs). The classification is also the serving
   architecture: bound/derived are shared read-only across sessions;
   a session *is* its state tensors. Multi-turn chat prefills only the
   new suffix tokens against the session's existing cache.

**Placement is the router's decision.** The artifact is hardware-
agnostic, so the router reads the manifest's weight sizes and places
the executor where the model fits: on the configured accelerator when
the weights are within 85% of its memory, otherwise on a CPU node with
memory and CPU requests sized from the weights (Gemma 4 31B, 62.6 GB
bf16, lands on a 240 GB CPU node when the cluster's GPUs are 24 GB
L4s; Gemma 4 E4B, 16 GB, goes to the L4). Override the accelerator's
memory with `VXPU_ACCELERATOR_MEMORY_GIB` and the CPU cap with
`VXPU_CPU_EXECUTOR_MAX_CPUS` on the router.

The executor applies device-specific compatibility passes to the
shipped graph at load time (e.g. rewriting `histc` — no CPU kernel for
integer inputs — to its exact `bincount` equivalent), so one artifact
serves heterogeneous executors: the graph carries reference semantics;
each engine adapts it to its hardware.

## Security & Trust Assumptions

The vXPU executor **runs the artifact it is given**. There is currently no verification or sandboxing in the executor, and verifying the graph's structure is not a viable security control (a check strong enough to guarantee safety would mean we could have regenerated the graph ourselves).

Consequently, the core operating assumption is: **only run artifacts you trust / produced.**

### The Pickle Risk in `.pt2` Archives

A `.pt2` file is actually a zip archive containing:
- `models/model.json` — the declarative, non-executable exported graph (ATen ops + shapes).
- `data/weights/*`, `data/constants/*` — raw tensor weights and constants.
- `data/sample_inputs/model.pt` — **a `torch.save` archive containing `archive/data.pkl`, which is a Python pickle.**

Unpickling runs Python `__reduce__`, which is a classic arbitrary code execution vector. A hostile `.pt2` file could carry a malicious payload inside this sample-inputs pickle that executes during load time before any graph inspection or execution occurs.

### Investigation & Findings

When the executor loads the program using `torch.export.load()`, we investigated its behavior regarding `data/sample_inputs/model.pt`:

1. **Eager Deserialization**: `torch.export.load` utilizes an internal `unpackage_pt2` call that **eagerly** unpacks and unpickles the entire archive, including `data/sample_inputs/model.pt` (loaded into the `example_inputs` property). There is no native lazy-loading or skipping support for these inputs at load time.
2. **Impact**: Because the unpickling happens eagerly upon calling `torch.export.load()`, code execution occurs immediately when loading an untrusted artifact.
3. **Hardening Mitigation**: Since the serving executor only needs the declarative graph, weights, and configuration metadata to perform inference, the sample/example inputs are completely unnecessary at serving/load time.
   - For future hardening, dropping the sample inputs during the export phase (e.g., by setting `exported_program.example_inputs = None` before calling `torch.export.save`) would completely prevent the creation of `/data/sample_inputs/model.pt` inside the zip archive.
   - Dropping this pickle file would eliminate the primary arbitrary code execution surface of the `.pt2` artifact, leaving it entirely declarative.

## Executor API

`Generate` is transformers' `generate()` over the wire: the client owns
the tokenizer and sends the complete prompt as ids; the executor runs
the same torch loop (prefill, decode, and the same logits processors
transformers applies for `do_sample`/`temperature`/`top_k`/`top_p`/
`repetition_penalty`, defaulting to the model's `generation_config`)
and streams back new ids. Greedy output is token-for-token identical to
transformers; sampled output is repeatable with `seed`. The session's
KV cache is reused for whatever prefix of the prompt matches, so
re-sending a growing conversation prefills only what is new.

`Chat` runs one text turn in a session. By default `text` is the user's
message and the executor applies the model's chat template and keeps
the transcript. With `raw_prompt=true` the client owns the transcript:
`text` is the complete rendered prompt (e.g. `apply_chat_template(...,
tools=[...], tokenize=False)`) and the executor tokenizes it verbatim.
In both modes the executor compares the new prompt's tokens with what
is already in the session's KV cache and prefills from the first token
that differs (the caches are position-addressed, so a rewritten tail
simply overwrites). Raw-mode replies keep the model's control tokens
(e.g. Gemma's `<|tool_call>` delimiters) so the client can parse them;
templated replies are plain text. Generation stops on any of
the model's end-of-turn ids (`config.eos_token_id`, e.g. Gemma's
`<end_of_turn>`), not just the tokenizer's `eos`.

## Layout

```
proto/        Executor gRPC API (LoadModel / NewSession / Chat)
python/vxpu/  export (thin client), client (thin client, no torch),
              modeling (transformers-shaped from_pretrained/generate),
              server (executor): manifest, export, rehydrate, engine
cmd/vxpu/     Go CLI: no Python/torch — ships artifacts, creates the
              router pod on demand, port-forwards, chats
              (`vxpu up` brings up just the router for other clients)
cmd/vxpu-router/  in-cluster router: caches weights, places and
              creates executor pods, proxies the Executor API
images/       executor and router container images
examples/     notebooks (Gemma 4 on vXPU)
```

## Building

```sh
# CLI
go build ./cmd/vxpu/

# Executor and router images (from vxpu/):
gcloud builds submit --config cloudbuild.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-executor:v1 .
gcloud builds submit --config cloudbuild-router.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-router:v1 .
export VXPU_EXECUTOR_IMAGE=gcr.io/$PROJECT/vxpu-executor:v1
export VXPU_ROUTER_IMAGE=gcr.io/$PROJECT/vxpu-router:v1
```

The CLI applies the router pod together with a ServiceAccount/Role that
lets it create executor pods, and a `vxpu-router` Service for in-cluster
clients.

## Verified behavior

- Executor logits are bitwise-identical to `from_pretrained` for the
  same device (verified cross-OS and cross-architecture), and cached
  generation is token-for-token identical to Hugging Face `generate`
  with a static cache.
- Gemma-4-E4B (hybrid local/global attention, p-RoPE, 16 GB bf16)
  exports with every tensor classified and generates coherently
  through the executor on an L4: ~22 tok/s steady-state, plus a ~33 s
  one-time `torch.compile` on the first decode step. That ~22 tok/s is
  close to the L4's bf16 memory-bandwidth roofline (~300 GB/s ÷ ~16 GB
  read per token ≈ 19 tok/s) — LLM decode is bandwidth-bound, so the
  lever for more speed is quantization or a higher-bandwidth card, not
  the pipeline. (The same L4 served a 4-bit 26B-A4B at ~60 tok/s in the
  experiment, reading ~4x fewer bytes per token.)
- A 26B MoE (53 GB) exports to a 20 MB graph and executes on a
  240 GB-RAM CPU node — the artifact is hardware-agnostic; placement
  is the router's decision.
- Gemma-4-31B-it (dense, 62.6 GB bf16) exports to a 75 MB artifact on a
  laptop in about a minute and, on an L4-only cluster, is placed by the
  router on a c3d-standard-60 CPU node (30 CPUs, 74 GiB requested,
  67 GB resident). Cold path measured: ~9 min for the router to cache
  the two safetensors shards from the Hub, ~195 s for the executor to
  rehydrate 65 GB from the router (model ready 750 s after shipping
  the artifact), then 1.2–1.4 s/token decode and a 1.5 s prefill for a
  19-token prompt — bandwidth-bound CPU decode, fine for a notebook,
  not for serving. The Hugging Face docs examples (causal LM, function
  calling, streaming multi-turn) run unchanged from a GPU-less notebook
  against it through `vxpu.AutoModelForCausalLM` (see `examples/gemma4`).

## Status and roadmap

vXPU is an early extraction of the `experiments/model-manifest`
work — the experiment retains the full research trail.

Known simplifications / tasks for the immediate roadmap are:
- Tokenizer/processor files ride alongside the manifest rather than
  in it; executors currently fetch them from the source repo.
- Artifacts travel inside the LoadModel RPC; OCI registries
  (per-tensor layers, signed manifests) are the intended distribution.
- One model per executor; engine-backed executors (vLLM behind the
  same proto) for architectures whose graphs are not natively
  runnable (quantized fused-MoE) are planned.
- `.pt2` graphs pin the exporting torch version (the serialization
  format promises newer-loads-older; same-version is what we test).
- A shared paged KV pool is designed (see the experiment's
  PLAN-paged-rewrite.md) to replace fixed-shape per-session static caches.
