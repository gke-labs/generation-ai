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

3. **A tensor classification.** Every tensor the graphs
   reference is `bound` (a weight reference), `stacked` (a bank of
   weight references assembled on the executor, e.g. one MoE expert
   parameter from 128 per-expert checkpoint tensors, following
   transformers' own conversion recipes), `constant` (a buffer the
   model computes from its config, RoPE tables and embedding scales,
   captured with its real values at export and carried in the
   artifact), `state` (what the graphs write in place: the per-session
   KV cache, zero-initialized on the executor), or `unused` (lifted by
   export but read by no op, e.g. another tower's tables). Older
   artifacts carry `derived` tensors the executor recomputes from the
   config; the harness showed that recomputation wrong for Gemma 4's
   p-RoPE table, which is why constants are captured now. The classification is also the serving
   architecture: bound/derived are shared read-only across sessions;
   a session *is* its state tensors. Multi-turn chat prefills only the
   new suffix tokens against the session's existing cache.

**Pre-quantized checkpoints run as published.** A model-maker fp8
checkpoint (`quant_method: fp8`, e.g. Qwen's `-FP8` releases: fp8 weights
plus one scale per 128x128 block, bf16 activations) exports with
transformers' own `FP8Linear`/`FP8Experts` modules, installed on the meta
model exactly as `from_pretrained` would. The graph consumes the fp8
tensors directly: block-dequantize-then-matmul in plain aten, with a
mixture-of-experts bank bound as a `stacked` parameter the executor
assembles from the checkpoint's per-expert tensors (prefill is
transformers' `grouped_mm` routing, decode its `batched_mm` routing; both
static-shape, both the sparse computation it runs eagerly). Weights stay
fp8 in memory and are dequantized during compute, so the executor needs
the checkpoint's size, not the bf16 size, and the same artifact runs on
a CPU node or a GPU. (One exception trades memory for speed only where
memory allows: on a GPU, the DeepSeek fork's dense fp8 linears keep a
bf16 copy of each weight while at least `VXPU_BF16_CACHE_HEADROOM_GB`,
default 4, stays free; `VXPU_BF16_WEIGHT_CACHE=0` turns that off.) Nothing is re-quantized: the user's
`from_pretrained(repo)` is the model-maker's checkpoint, bit for bit.
Upstream transformers has no fp8 path off the GPU (its forward is CUDA
kernels only), which is why this needs the fork in `third_party/`.

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

## The physical plan

The artifact is the logical plan: the model's computation. How to run it
on a given machine is the executor's choice, recorded in a physical plan
(`python/vxpu/plan.py`), as a database chooses a plan for a query. Each
engine builds one at load, from the hardware and the configuration:

- **Hardware:** device, GPU and its memory, CPU, SIMD level, RAM.
- **Choices:** split decode, compile mode, where the experts live, the
  GPU's bf16 weight copies, the CPU kernels, and more. Each records its
  value and where the value came from (default, environment, argument).
- **Implementations:** for each op the graphs call (`vxpu::fp8_linear`,
  `vxpu::moe_experts`, ...), the registered implementations in
  preference order, each with the devices it runs on, when it applies, and
  its numerics class (bit-identical to the reference computation, or equal
  to rounding). The ops dispatch through the plan, and the plan records
  which implementation each input shape got.

The engine logs the plan at load, and `Engine.explain()` returns it.
`python -m vxpu.plan --artifact DIR --device cuda --compile` explains the
plan for an artifact without loading weights. `vectors_test record` and
`dev/vm/attribute_decode.py` store the plan with their results, so a
measurement says which plan produced it.

The plan can also be lowered into the decode graph (`python/vxpu/lower.py`,
`VXPU_LOWER_DECODE=1`). Each dense op call becomes a call of the
implementation selected for its shapes, with that implementation's Triton
kernels registered as torch ops. The compiler then sees the kernels
themselves, not an opaque Python call. That is the first step toward
compiling the segments ahead of time (AOTInductor, #277). A lowered call
runs the same kernels on the same inputs, so the output is bit-identical.

With `VXPU_SEGMENT_RUNTIME=aoti` (`python/vxpu/aoti.py`, CUDA only, which
implies lowering), each dense segment is compiled by AOTInductor into a
package, and the engine runs the packages in its own CUDA graphs instead
of torch.compile. Packages are built on first use and cached under
`VXPU_AOTI_CACHE_DIR` (default `~/.cache/vxpu/aoti`), keyed by the
segment's code with its values named by position, its argument shapes,
the kernels' source, the torch version and the device it runs on (the
GPU, or the CPU's features: inductor compiles CPU code for the machine
it runs on). Layers that compute the same thing on different weights
share a package: DeepSeek-V4.1-Flash's 85 decode segments are 14
packages. A later load of the same artifact on the same kind of machine
compiles nothing: the cache holds compiled plans.

The environment switches are read by one parser, which refuses values it
doesn't understand: `VXPU_KERNELS`, `VXPU_FUSED_EXPERTS`,
`VXPU_KERNEL_THREADS`, `VXPU_SPLIT_DECODE`, `VXPU_COMPILE_MODE`,
`VXPU_SPLIT_COMPILE`, `VXPU_DIRECT_HOST_OPS`, `VXPU_LOWER_DECODE`,
`VXPU_SEGMENT_RUNTIME`, `VXPU_BF16_WEIGHT_CACHE`,
`VXPU_BF16_CACHE_HEADROOM_GB`, `VXPU_GPU_EXPERTS_MOVES`,
`VXPU_GPU_EXPERTS_REQUEST_WEIGHT`, `VXPU_EXPERT_MAILBOXES`, `VXPU_DATAFLOW`,
`VXPU_DATAFLOW_PLACE`, `VXPU_STEP_GRAPH`; and `VXPU_EXTERNAL_DIR`,
`VXPU_AOTI_CACHE_DIR`, `VXPU_GPU_EXPERTS` and `VXPU_REMOTE_NODES`, taken as
given. `vxpu machines [NAME]` shows the machine database (`pkg/machines`).

To see how a model would run on machines you could get, without running
it, rank the planner's candidate layouts by the cost model (#409):

```sh
python -m vxpu.cuts ARTIFACT_DIR > ARTIFACT_DIR/cuts.json  # where the step can be cut into ranges
vxpu explain --artifact ARTIFACT_DIR --machines g2-standard-48:2,c3d-standard-60:2 \
  [--records records.jsonl]   # recorded traffic (vxpu stats --records), for GPU experts
```

It lists every layout (one machine; one machine running the step with
other machines serving experts; ranges of whole layers on several
machines) with its predicted ms/token, the parts it adds up from and
what each number rests on. `--json` gives them in full.

`--serve :8080` shows the planner searching instead: every layer's dense
half and experts each on some machine, random layouts and changes to the
best ones scored by the cost model at a few hundred a second, the layout
being scanned and the best found so far drawn coloured by machine, with
network crossings in red. `dev/tasks/record-plan-search` records it as a
video.

The router runs the layer ranges layout when the cost model ranks it
first, on its own. By hand, start each later range on its executor and
point the head at them:

```sh
VXPU_LAYER_RANGE=20: ...                                # an executor running layers 20 to the last;
                                                        # ServeRange gives its port
VXPU_LAYER_RANGE=0:20 VXPU_REMOTE_RANGES=20:=tcp://HOST:PORT ...   # the head
```

## The dataflow executor: segments, experts and mailboxes

A decode step split at its host ops is a dataflow graph (#372,
`python/vxpu/dataflow.py`):
- the dense segments, the routed experts and the n-gram lookups are its
  nodes;
- a node's inputs arrive in its inboxes, written by the nodes that
  produce them;
- each node runs where the plan places it.

On DeepSeek-V4.1-Flash the graph is 85 nodes (43 dense, 40 experts,
2 engram) and 418 edges.

- **`VXPU_DATAFLOW=1`** runs decode through the reference runner: one
  inbox per edge at the consumer's device, allocated once, and a node
  fires when its inboxes are full. `VXPU_DATAFLOW_PLACE=experts=cpu,engram=cpu`
  places nodes by kind. The runner computes exactly what the split
  module computes.
- **`VXPU_EXPERT_MAILBOXES=1`** answers decode's routed experts through
  mailboxes (`python/vxpu/mailboxes.py`):
  - each (token, k) pair goes to its expert's owner: the CPU kernels, a
    GPU pool of hot experts (`VXPU_GPU_EXPERTS`);
  - every owner replies with the pair's bf16 down projection;
  - the replies are summed in k order whoever answered.

  With every expert on the CPU this is bit-identical to the fused kernel.
- **`VXPU_REMOTE_NODES=name=tcp://host:port/index[,...]`** names host
  nodes (an experts layer, an n-gram table) a node server serves: another
  executor, on this machine or another, running the nodes as native
  packages (`python/vxpu/nodeserver.py`; the coordinator starts them with
  `ServeNodes` and sets this). The engine does not load those nodes'
  banks or tables, and the results are bit-identical.
- **`VXPU_STEP_GRAPH=1`** (CUDA, with `VXPU_DATAFLOW=1`,
  `VXPU_SEGMENT_RUNTIME=aoti` and the CPU kernels) captures the whole
  decode step in one CUDA graph (`python/vxpu/stepgraph.py`):
  - the dense nodes are their AOTInductor packages;
  - each host op is a mailbox in the graph: the GPU copies the inputs to
    pinned memory and raises a flag (`cuStreamWriteValue32`); the step
    host, a native thread (`csrc/node_server.cpp`), runs the node (its
    native package, or a node server's through the native client) and
    raises a flag back; the GPU waits on that flag
    (`cuStreamWaitValue32`) and copies the reply in;
  - hot experts in a GPU pool run inside the graph while the step host
    answers the rest.

  Python launches a step and waits for its end; no Python runs between a
  host op's flag and its reply.

Measured on DeepSeek-V4.1-Flash (a g4-standard-96, two RTX PRO 6000s; ten
prompts none of which were in the traffic the placements came from):

| decode | ms/token |
|---|---|
| node by node, every expert on the CPU | 46.9–47.7 |
| node by node, hot experts on both GPUs (163 GB) | 31.7–38.0 |
| one graph a step, every expert on the CPU | 35.7–36.1 (3 prompts) |
| one graph a step, hot experts (68 GB) inside it | 24.1–30.9 |

## Models by name

With modelstore deployed, the engine takes a model *name*: it imports
the weights, compiles the model for its executors, and serves it to any
gRPC client (Go here; the `vxpu` CLI is one).

```sh
vxpu register deepseek-ai/DeepSeek-V4.1-Flash           # import + compile, once per commit and executor image
vxpu ask --model deepseek-ai/DeepSeek-V4.1-Flash "Is the sky blue?"
```

`RegisterModel` imports the repository into modelstore, then runs a
compile Job: the executor image's `python -m vxpu.compile`, which
exports the artifact reading the model through modelstore's Hub API
(config and safetensors headers only, never the weights) and pushes it
to modelstore as `vxpu-artifacts/<model>`, tagged with the commit and
the executor image. The artifact is exported by exactly the torch and
forks the executors run, on a CPU node. The tag is the catalog entry: a
model is registered for this engine when the tag exists, so the catalog
lives in modelstore, not in the router. `NewSession` with a model name
fetches the artifact from modelstore and loads it the first time
(`FAILED_PRECONDITION` while it loads, so clients retry); later
sessions reuse it. Models are any whose architecture the exporter and
the forks support.

The engine keeps statistics about models served by name: every
request's tokens, prefill and ms/token, which commit, artifact and
executor served it, and, for every Nth request (`--telemetry-every` on
the router, default every one), the executor's execution telemetry
(segment times, expert selections, the plan). They are stored in
modelstore (`vxpu-stats/<model>`, a tag per day) and outlive the
router; telemetry stays internal unless a client asks for it.

```sh
vxpu stats deepseek-ai/DeepSeek-V4.1-Flash                    # summary
vxpu stats --records reqs.jsonl deepseek-ai/DeepSeek-V4.1-Flash
python -m vxpu.expert_stats reqs.jsonl                        # expert usage from what was served
```

Executors for models served by name can run outside Kubernetes, on
machines launched for the purpose (a GPU VM from `gcloud`, a Mac)
running `python -m vxpu.server`: start the router with their addresses,
`vxpu up --external-executor 10.128.0.5:50051[,...]`. The router dials
them (on GKE's VPC-native networking a pod reaches a VM's internal IP);
it places a model on the executor already holding it, else an idle one,
and reloads a model an executor evicted, or whose load failed. Kubernetes
keeps the bursty, CPU-only work: compile jobs, modelstore, the router.

A machine outside the cluster cannot resolve modelstore's Service. Either
let it copy weights from the bucket itself (the executor's
`VXPU_BLOB_BUCKET`, its local weight tier), or tell the router where it
can reach modelstore: `vxpu up --modelstore-external-url
http://10.128.0.9:8080` (an internal load balancer, say). The executor
image keys compiled artifacts (`vxpu register`): pin it by digest, since
a rebuilt tag such as `:latest` reuses what the previous build compiled.

Or give it a weights disk (#407). A weights disk holds the checkpoint's
files, and the VM attaches it instead of copying the files:

```sh
# Once per checkpoint, a factory VM copies the files onto a disk, and we snapshot it.
vxpu disk build --project P --zone Z --image ds-flash --bucket gs://BUCKET --manifest ARTIFACT/manifest.json
# For each executor VM, make a disk from the snapshot in its zone, attach it read-only,
# and have the VM's agent mount it.
vxpu disk attach --project P --zone Z --vm VM --image ds-flash --disk ds-flash-z --agent VM_IP:50052
```

`attach` prints the blob directory to give the executor as
`VXPU_BLOB_DIR`. The VM runs `vxpu-agent` as root (`cmd/vxpu-agent`), the
per-machine agent that mounts the disk and checks it against its index.

## Weights: the store

Weights are pulled from the Hub once and kept in the cluster's weight
store, [modelstore](../modelstore/): a content-addressed bucket
(`gs://…/_blobs/<sha256>`) with a `Model` object per repository
revision and a Hugging Face compatible read API. When modelstore is
deployed (`modelstore.modelstore` resolves; or pass `vxpu up
--modelstore URL`), the router no longer caches weights itself: on
`LoadModel` it asks modelstore to make sure every file in the manifest
is present (modelstore streams what it lacks from the manifest's Hub
sources, verifying each sha256) and rewrites the sources to
modelstore's blob URLs. Executors range-read those, one tensor per
request as before, so a second load of the same checkpoint on any node
touches the Hub not at all. modelstore is the only endpoint executors
know: it carries the bucket credentials, and whether it proxies the
bucket or redirects a read to a signed object URL is its decision, not
the executor's. Without a modelstore (kind, the e2e) the router's own
cache is used as before.

```sh
vxpu mirror Qwen/Qwen3-8B-FP8        # import once, ahead of time (optional)
kubectl get models                   # phase, files, bytes
```

Reference and vLLM pods load from the same store with one variable,
`HF_ENDPOINT=http://modelstore.modelstore`, which transformers, vLLM and
`huggingface_hub` honour unchanged. The one-time bucket and
workload-identity setup is in modelstore's README.

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

## Forked dependencies (`third_party/`)

When a library stands between a model and a faithful export, we carry a
small patch in a fork rather than build around it in vXPU. The forks are
git submodules, branched from the pinned release:

| submodule | branch | patch |
|---|---|---|
| `third_party/transformers` | `vxpu/v5.18.0` (from `v5.18.0`; `vxpu/v5.14.1` carries the fp8 patches on the older release) | plain-torch path for `FP8Linear`/`FP8Experts` off the GPU (CPU, MPS, meta), so fp8 checkpoints load on a CPU without `dequantize=True` and export weightlessly; `FP8Embedding` for tables stored as fp8 with a per-tensor scale; float `histc` off the GPU in the grouped experts; the export wrapper sizes and registers linear-attention/conv states and passes `cache_position`; `qwen4_exp` exports with a symbolic prefill length (static indexer selection, `while_loop` chunk recurrence, no Python branches on the length); `sdpa` evaluates `is_causal` without guarding the length |

```sh
git submodule update --init
uv pip install -e third_party/transformers -e python/
```

The executor image installs the same fork, so exporter and executor
run one transformers. Each patch is written to be upstreamable.

## Verifying a model: test vectors

`vxpu.vectors_test` records what a model says and compares two recordings.
Eager transformers writes the reference; the vXPU engine (in-process,
or through a router) and vLLM write candidates. Where the candidate's
greedy tokens leave the reference's, the reference's own logits at
that position decide whether the flip was a tie (within the tolerance
of its top choice: a coin toss two correct engines can call
differently) or a divergence; and the top-k logits of every generated
position, up to the first flip, must agree within the tolerance. The file format (`vxpu-vectors/1`) is the
interchange point between engines.

```sh
# reference, from the same transformers the exporter uses
python -m vxpu.vectors_test record --producer transformers --model Qwen/Qwen3-8B-FP8 -o ref.json
# candidate: the exported artifact on this machine's CPU ...
python -m vxpu.export Qwen/Qwen3-8B-FP8 -o qwen8b/
python -m vxpu.vectors_test record --producer vxpu-engine --model Qwen/Qwen3-8B-FP8 --artifact qwen8b/ -o got.json
# ... or through the router in the cluster (tokens only; no logits on the wire)
python -m vxpu.vectors_test record --producer vxpu-router --model Qwen/Qwen3-8B-FP8 --artifact qwen8b/ -o got.json
# ... or vLLM; in a vLLM image with no vxpu installed, copy vectors_test.py in and
# run it as a file: python vectors_test.py record --producer vllm ...
python -m vxpu.vectors_test record --producer vllm --model Qwen/Qwen3-8B-FP8 -o vllm.json
python -m vxpu.vectors_test compare ref.json got.json
```

Two tiers. The CI tier, `python -m vxpu.vectors_test selftest`, runs in the
e2e presubmit on three tiny models built in-process or fetched from the
Hub: a random fp8 Qwen3-MoE (`vxpu.fixtures.tiny_fp8_moe`: block
scales, per-expert tensors, an unquantized lm_head), the tiny random
Llama, and a random Qwen4 hybrid (`vxpu.fixtures.tiny_qwen4`: Gated
DeltaNet linear attention, indexed attention, n-gram embedding, MoE),
each exported, run on the in-process engine, and required to be
token-exact against transformers (a flip is excused only when the logit
error measured at that position covers its gap: an exact tie that two
rounding paths call differently). The cluster tier is the same commands
against the real models on the c3d CPU node (the 30B through the router
on the RTX PRO 6000, tokens only). Latest run:

| model | reference | prompts | greedy tokens | top-k logits, max diff | ms/token, reference → vXPU |
|---|---|---|---|---|---|
| google/gemma-4-E4B-it | transformers bf16 eager | 3 | all equal | 0.44 (1.7%) | 160–290 → 220 |
| google/gemma-4-31B-it | transformers bf16 eager | 3 | all equal | 0.59 (2.4%) | 790–1330 → 1070–1110 |
| Qwen/Qwen3-8B-FP8 | transformers, fp8 dequantized to bf16 | 3 | all equal | 0.50 (1.2%) | 270–350 → 5700–6300 |
| Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | transformers, fp8 dequantized to bf16 | 3 | all equal | n/a (router) | 190–330 → 11.7 (RTX PRO 6000) |
| Qwen/Qwen3.8-Flash-Next-FP8 | none yet (186 GB; a reference needs a larger node) | | export + run only | | |

Two things this table found on its first run, both fixed in the same
change: Gemma 4's long prompts diverged because the executor
recomputed the p-RoPE table from the config and got it wrong past the
first entries (the artifact now carries the model's own buffers), and
sessions after the first ran linear-attention models on stale state
because the engine only re-bound registered buffers, not the cache
tensors export had lifted as constants. The fp8 numbers on the CPU also
say plainly that dequantizing per step costs 20x against bf16 weights
there; the GPU is where fp8 pays.

### A second engine: vLLM

`--producer vllm` records what vLLM serves for the same repo and
revision: the prompt as the same token ids (the chat template is
applied by the harness, not by vLLM), greedy, and the top-k *raw
logits* of each generated position (`logprobs_mode="raw_logits"`, the
pre-softmax values before any processor), so a vLLM recording compares
with the others at the logit level. vLLM places the model itself, and
the recording is labelled with the card it used. The producer needs
only the `vllm` package: copy `vectors_test.py` into a vLLM image and run
it as a file.

Qwen3-8B-FP8 through vLLM 0.31 on two cards, against eager transformers
on the same card (the checkpoint dequantized to bf16) and against the
vXPU engine on the same card (the artifact, compiled):

| card | vLLM's fp8 path | engine | greedy tokens vs transformers | top-k logits, max diff | ms/token |
|---|---|---|---|---|---|
| L4 | Marlin, weight-only (fp8 weights, bf16 activations) | vLLM | 3 of 3 prompts equal | 0.75 (2.0%) | 34–39 |
| L4 | (the graph: fp8 weights, bf16 activations) | vXPU | 3 of 3 equal; also 3 of 3 equal to vLLM | 0.75 (1.8%) | 40 |
| RTX PRO 6000 | DeepGEMM, fp8 weights *and* activations | vLLM | 1 of 3 equal; two tie flips (gaps 0.0 and 0.75) | 2.75 (6.5%), fails | 7.6–8.0 |
| RTX PRO 6000 | (the graph: fp8 weights, bf16 activations) | vXPU | 3 of 3 equal; vs vLLM the same two tie flips | 0.625 (1.7%) | 10.3 |

Reading the table: on the L4, where Ada has no fp8 compute and vLLM
dequantizes the weights and multiplies in bf16, vLLM computes the same
function as transformers and the vXPU graph, and its logits sit within
bf16 rounding of them. On the RTX, vLLM takes its DeepGEMM path, which
quantizes the activations to fp8 per 128-group as well, and its logits
move by up to 6.5% of the top logit; the two token flips are an exact
tie (the transformers reference has both candidates at 40.5 at that
position) and a 0.75 gap at a top logit of 31.5, which is 2.4% and so
inside the tie tolerance too. The rows fail on the logits, not on the
tokens: the tokens were coin tosses, the distribution is what moved.
Transformers' own fp8 kernel path on the same card (`FP8Linear` with
the Hub kernel, also fp8 activations) lands 3% from its bf16 path and
6–7% from vLLM, so vLLM's path differs in more than the GEMM.

Three rules the vLLM column taught the harness. A token flip is judged
by the reference's logits at that position (its *tie set*: every token
within the tolerance of its top choice), not by equality: eager
transformers on the CPU and on an L4 already disagree on one token of
the kubernetes prompt, a near-tie 0.25 apart in logits of magnitude 40,
one bf16 ulp, and that is a tie, not a bug. For the same reason the
reference must be recorded on the same device class as the candidate,
and logits are recorded for every generated position, since a flip at
a position without them cannot be judged. And logits are compared only
up to the first flip; after that each side's logits answer a different
question, and the first version of `compare` reported 130% diffs that
meant nothing (teacher forcing, #243, will lift that limit).

The equivalence bar for the vXPU graph is unchanged: it must match the
computation transformers runs for the checkpoint, and it does on both
cards. vLLM's path with fp8 activations (W8A8, in the usual shorthand;
the graph is W8A16) is a different computation of the same weights,
and the question of which one the model maker's fp8 recipe intends is
the question #217 and #218 ask of DeepGEMM substitution: a kernel that
changes the logits by 6% is a placement choice the user should see in
this table, not an invisible rewrite.

## Layout

```
proto/        Engine (client) and Executor (internal) gRPC APIs
python/vxpu/  export (thin client), client (thin client, no torch),
              modeling (transformers-shaped from_pretrained/generate),
              vectors_test (test vectors: record with any producer, compare),
              server (executor): manifest, export, rehydrate, engine
cmd/vxpu/     Go CLI: no Python/torch — ships artifacts, creates the
              router pod on demand, port-forwards, chats
              (`vxpu up` brings up just the router for other clients)
cmd/vxpu-router/  in-cluster router: keeps weights in modelstore (or
              caches them itself), places and creates executor pods,
              proxies the Engine API to executors
images/       executor and router container images
examples/     notebooks (Gemma 4 on vXPU)
third_party/  forked dependencies as submodules (transformers)
docs/         design notes (design.md: vXPU as a database engine for LLMs, today and where it is heading; representation.md: why the artifact is pt2)
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

- Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, 128 experts top-8, model-maker
  fp8 checkpoint, 31.2 GB) exports weightless in about a minute to a
  77 MB artifact using transformers' own fp8 modules (from the fork in
  `third_party/`); the router places it as a 40 GB / 15-CPU pod on the
  c3d node and the executor holds ~32 GB resident. Greedy output through
  `vxpu.AutoModelForCausalLM` is token-for-token identical to
  transformers loading the same checkpoint on the same node class, both
  with `FineGrainedFP8Config(dequantize=True)` (bf16, ~61 GB) and, with
  the fork, as fp8 on the CPU. Speed on the CPU is bound by the fp8→bf16
  cast: prefill dequantizes every expert once (22 s at 13 tokens, 26 s
  at 474; `grouped_mm` routing keeps the growth to ~8 ms/token) and
  decode is ~2.3–2.9 s/token, against ~170 ms/token for the bf16
  reference at twice the memory. Eager transformers on the CPU with fp8
  weights is far slower per token (~20 s) because its `grouped_mm`
  experts dequantize the whole bank per step; the decode graph uses the
  `batched_mm` routing, which touches only the routed experts. See
  [docs/representation.md](docs/representation.md).
- The same fp8 graph on a GPU runs at full speed without a device-specific
  artifact. Qwen3-8B-FP8 (dense, 9.4 GB) on an L4: the bf16 checkpoint
  decodes at 67 ms/token in eager transformers and transformers' own fp8
  kernel path (Hub Triton kernel) at 136 ms/token; the exported graph,
  compiled plainly, at 132 ms/token; with inductor's prologue fusion
  enabled by the executor on CUDA (`engine.configure_inductor`), at
  **46 ms/token**, the fp8 bandwidth floor. The dequantization is folded
  into the GEMM's operand load, so fp8 weights are read once at fp8
  width. The cost is a one-time autotune of about two minutes on first
  load (`VXPU_COMPILE_MODE=default` skips it).

- Qwen3.8-Flash-Next-FP8 (hybrid: 36 Gated DeltaNet linear-attention
  layers, 12 indexed-attention layers, 512 experts top-10, a per-layer
  n-gram embedding stored as 128 fp8 shards; 186 GB) exports weightless
  on a laptop with transformers 5.18 from the fork: a 38 MB prefill
  graph (36 `while_loop` chunk recurrences with a symbolic trip count,
  96 `grouped_mm`, 72 triangular solves, 12 `scaled_dot_product_attention`
  with a static indexer selection) and a 29 MB decode graph; 1071 bound,
  2 derived, 134 state and 193 merged tensors (expert banks and the
  sharded table, assembled on the executor from transformers' own
  conversion recipes). In the cluster: the router cached the 131 shards
  in ~9 min, the executor (a 217 GiB / 30-CPU pod on the c3d node)
  rehydrated 182 GB in 12 min and holds 177 GB resident with the fp8
  tensors kept as fp8; the first prompt answered at ~2.3 s/token after
  a ~75 s prefill (every expert bank is dequantized once per prefill).
  (That run predates the session-state fix above: only its first
  session had correct linear-attention state.)
  Correctness of the static-cache path and of the pt2 round trip is
  verified against eager transformers on a small random Qwen4 model
  (float32 rounding); a same-node reference run of the 186 GB model is
  still to do.

- On an RTX PRO 6000 (Blackwell Server Edition, 96 GB, Spot node) the
  same two fp8 artifacts, unchanged, with the executor's default CUDA
  compile settings: Qwen3-30B-A3B-Instruct-2507-FP8 decodes at
  11.7–12.2 ms/token (~85 tok/s) with the same tokens as before, against
  25.4–26.9 ms/token for eager transformers running the checkpoint
  dequantized to bf16 (61 GB) and 46–62 ms/token for eager transformers
  running it as fp8 with its own kernels (`FP8Linear`/`FP8Experts`, the
  Hub Triton kernel; DeepGEMM needs a datacenter part) on the same card;
  Qwen3-8B-FP8 at 12.2 ms/token. Prefill of a 474-token prompt takes 0.4 s (24 s on the CPU
  node). Gemma-4-31B-it (dense bf16, 62.6 GB) rehydrates onto the card
  in 104 s and decodes at 50–53 ms/token (1.2–1.4 s/token on the CPU
  node), the same text. The first load pays ~1–2 min of autotuning. That the
  8B and the 30B take the same time per token says the step is bound by
  kernel launches, not bandwidth, on this card (CUDA graphs are the next
  lever; see the issue tracker).

## Status and roadmap

vXPU started as the `experiments/model-manifest` experiment, which is in
the git history (removed in #392).

Known simplifications / tasks for the immediate roadmap are:
- Tokenizer/processor files ride alongside the manifest rather than
  in it; executors currently fetch them from the source repo (with a
  modelstore deployed, `HF_ENDPOINT` can point them at the store).
- Artifacts travel inside the LoadModel RPC; OCI registries
  (per-tensor layers, signed manifests) are the intended distribution.
- One model per executor.
- fp8 checkpoints are dequantized per block on every use; an executor
  that dequantizes at load when memory allows, or uses the device's fp8
  kernels, would be a placement choice, not an artifact change.
- `.pt2` graphs pin the exporting torch version (the serialization
  format promises newer-loads-older; same-version is what we test).
- A shared paged KV pool is designed (see PLAN-paged-rewrite.md in the
  experiment's git history) to replace fixed-shape per-session static caches.
