# vXPU design: a database engine for LLMs

A database separates two levels of abstraction. The logical level is
what a query asks for: SQL says which rows and columns to return, not how
to find them. The physical level is how the database computes that:
which indexes to read, which join algorithm to use, in what order. The
optimizer chooses the physical plan, picking between candidates by
predicting their cost from statistics the database keeps about its
tables, and the executor runs it. When the data grows or the hardware
changes, the query stays the same. Only the physical plan changes.

vXPU does the same for LLMs. An LLM compiles once into its logical plan:
its computation as a graph (see [representation.md](representation.md)).
vXPU then chooses the physical plan for each machine: which device runs
each part of the graph, where each weight lives (GPU memory, RAM, local
SSD, another machine), and which kernel computes each op. Like a
database, it measures what it runs and feeds those statistics back into
the next choice.

The abstraction that makes this work is the mailbox. vXPU breaks an
LLM's execution graph into subgraphs. Each subgraph receives its inputs
in inboxes and writes its output to the mailbox of the next subgraph. A
subgraph might run on a GPU, the CPU or another machine; the mailbox
abstraction lets them intercommunicate and makes them interchangeable.
In databases, the Volcano model gave every operator the same interface,
and its exchange operator turned parallelism into a plan choice instead
of operator code. Mailboxes do the same for placement, so the planner
can put each subgraph wherever it costs least (§6).

This note describes the choices vXPU makes today and, section by section,
where the design is heading. A PR that changes the design updates this
note. The backlog of ideas lives in GitHub issues, linked from here.

## 1. The physical plan

An engine builds a physical plan when it loads an artifact
(`python/vxpu/plan.py`):

- **Hardware:** the device, the GPU and its memory, the CPU, SIMD level, RAM.
- **Choices:** compile and split the decode graph, where the experts live,
  whether to use the CPU kernels, and so on. Each records its value and
  where the value came from: a default, the environment, an argument, or
  the coordinator.
- **Implementations:** every op the graphs call (`vxpu::moe_experts`,
  `vxpu::fp8_linear`, ...) has registered implementations in preference
  order, each with the devices it runs on, when it applies, and its
  numerics class: *bit-identical* to the model's own computation, or
  equal up to *rounding*. Ops dispatch through the plan, and the plan
  records what each input shape got.

The coordinator can set the plan's switches. A model registered with
`vxpu register --plan VXPU_NAME=VALUE` keeps that plan in modelstore, and
the router sends it with every load of the model (`LoadModelRequest.plan`,
#394). The executor reads those switches ahead of its environment and
refuses any it may not take: paths and endpoints on its own machine stay
its own. The plan is part of what an executor loads, so a new plan
loads the model again.

The planner (`pkg/planner`) writes the rest of the plan for each load by
name. Before loading, the router asks the executor what it runs on, and
the planner chooses from that, from what the artifact holds (its dense
weights, and each layer's experts and their bytes) and from what the
model's traffic selected (§5). v0 chooses where the experts live (§4):
in which machine's RAM, then which of the engine's the GPUs hold. The registered plan's switches win over the planner's,
as a person's hints. `vxpu explain MODEL` shows the switches sent and
the reason for each choice.

The plan can be lowered into the decode graph (`lower.py`): each op call
becomes the selected implementation, its Triton kernels visible to the
compiler. Dense segments then compile ahead of time with AOTInductor
(`aoti.py`), cached per artifact, plan and machine.

The lowering also chooses what the compiler fuses. Some ops have a
second lowerable implementation: a decomposed body of plain tensor
operations, which inductor fuses with its neighbours. Small aten
matmuls (linear, einsum) can likewise become reductions. Fusing saves a
kernel launch and a round trip through memory, which is most of what an
op on small tensors costs. But inductor turns a matrix-vector product
into one reduction per output element, which reads the weight again for
each row. At DeepSeek's shapes on an L4, the fused fp8 linear took
290 µs where our GEMV took 32 µs (5120×8192, one row, measured). So the
lowering measures both candidates at each call's shapes on the engine's
GPU and keeps the cheaper (`costs.py`), as a database optimizer costs
its alternatives. The decisions are cached per GPU and shape. On the
tiny DeepSeek on an L4 at 3 rows, it fuses rotary, the small linears
and einsums, and most fp8 linears, and keeps the Sinkhorn kernel. With
the sampling epilogue compiled too, a decode step went from 364 kernels
to 234, and from 1.13 ms to 0.93 ms (measured; the two numbers come
from different sessions).

The graph has natural regions, and each has a floor. The step's state
between blocks is one tensor (DeepSeek's hyper-connection stream); inside
a block, that tensor stays live beside the block's own intermediates. So
the lowering cuts the graph where live state falls to the stream alone
(`regions.py`), and the cuts land after each block's `hc_post`, around
the engram layer, and nowhere inside an attention block: a five-layer
model has 22 dense regions. Regions can be costed on their own, and a
kernel from a database could replace one whole. Decode is memory-bound,
so each region's floor is the bytes it reads over the GPU's bandwidth,
and `python -m vxpu.regions ARTIFACT` measures every region in step
order against it. On a fixture with the released model's dimensions
(`tiny_deepseek_v41 --shaped`: five layers, 16 experts, 6 GB) on an L4,
measured:

| rows | dense regions | the head | attention blocks |
|---|---|---|---|
| 1 | 9.8 ms | 5.2 ms (114% of its floor) | 89–98% of their floors |
| 8 | 50.1 ms | 40.6 ms (15%) | 89–96%, of twice the fp8 bytes |
| 32 | 25.1 ms | 12.1 ms (50%) | 74–87%, of twice the fp8 bytes |

At one row every region that reads weights is at its floor. At 8 rows
the head's kernel reads its 1.3 GB weight once per row, and every fp8
linear reads a bf16 copy because the fp8 GEMV applies up to 4 rows. The
gap to the floor, in every regime, is two kernel shapes: a product over
several rows that reads its weight once, for bf16 weights and for
block-scaled fp8 weights. The small regions between blocks (the
hyper-connection mixes and the gate: 2 MB of weights in 13–19 kernels)
are launch-bound at 11–23% of their floors, about 80 µs a layer, which
on a 40-layer model at one row is a seventh of the step: the third
kernel shape. So a handful of kernels covers the dense step, and the
region table says which to write first.

The first of them, one pass over a weight for up to 64 rows
(`triton_ops.gemv_rows`), ran as hero run 5 on the real model (Oct 30,
g4-standard-192, four RTX PRO 6000, every expert on the GPUs; the old
code is hero 4's, in the same session; measured):

| rows | step, new | step, old | decode tok/s, new | old |
|---|---|---|---|---|
| 1 | 22.0 ms | 22.1 ms | 46 | 46 |
| 9 | 36.9 ms | 42.7 ms | 252 | 216 |
| 33 | 62.6 ms | 67.6 ms | 525 | 488 |

With the drafter, one row decodes at 68 tok/s (hero 4: 53). At one row
nothing changed, and the region profile of the real model says why: the
dense regions take 12.5 ms of the 22 ms step while their floor is about
5 ms. They launch 2,089 kernels, 164 copies and 126 mailbox operations
per step, so at one row the step is bound by launches, not bytes; the
rest is the experts and their mailboxes. That is the next target. The
MoE gate as its own op with a two-kernel body was tried and reverted:
alone its kernels were three times faster than aten's, but in the step
the op boundary cost the fusion inductor had already done around the
gate, and the region got slower. A region's kernels are judged in the
region.

**Region ops from two kernel templates.** The dense half's gap is launches, not
bytes: a plain layer ran about 40 kernels, 134 µs over its 118 µs floor
on the hero machine, most of them small kernels around the GEMVs (norms,
rotary, quantization, index arithmetic). Five regions of the attention
block and the MoE are now ops, each the fork's computation as a
bit-identical reference, a fused tensor body the lowering may choose, and
Triton kernels built from two templates:
- a GEMV over a block-scaled fp8 weight with an optional prologue (RMS
  norm, rotary, quantization to the fp8 grid per 32) and epilogue (rotary,
  or a SwiGLU over two weights' products with the quantization of the
  result), `_gemv_pe_kernel` and `_dual_gemv_swiglu_kernel`;
- a per-row finishing kernel (norm, rotary, quantization to fp8 or fp4), `_row_post_kernel`.

| op | replaces | kernels before | after | on an L4, one row |
|---|---|---|---|---|
| `vxpu::shared_ffn` | the shared expert: three fp8 linears and the SwiGLU | 9 | 3 | 152 µs against 154 |
| `vxpu::attn_q` | q down-projection, q_norm, up-projection, rotary | about 9 | 4 | 208 µs against 206 |
| `vxpu::window_kv` | kv projection, kv_norm, rotary, act_quant | about 6 | 3 | 21 µs against 25, bit-identical |
| `vxpu::compress_post` | compressed KV and index key: rotary, fp4 quantize-dequantize | about 6 | 1 | bit-identical |
| `vxpu::attn_out` | inverse rotary, act_quant, grouped and output projections | 5 | 4 | 310 µs against 312 |

(Measured against the fork's reference at the released model's shapes,
1 to 64 rows; "against" is the compiled fused body on the same card. The
L4 is bandwidth-bound at these sizes, so the kernel count shows in time
only on a faster card.) Two lessons from writing them. A prologue that
needs the whole row, a norm or a rotary with quantization, must not be
folded into a GEMV whose programs tile a wide output: every program
recomputes it, 2,048 times for the query up-projection, and the kernel
lost to the compiler's version by 25% at one row and 2.7 times at 33.
The per-row work runs once in the per-row kernel; the GEMV keeps only
the epilogue, which runs once per output. And the fork rounds to bf16
between its steps (the linear's output before the rotary, the rotated
tail before quantization); a kernel that skips a rounding differs at a
few elements per row, which the random-weight fixture turns into
different tokens. The selection among them is the planner's: the
lowering times the fused body against the kernels per call shape and
GPU. The ops are instances of templates on purpose: the next step is
for the planner to instantiate the templates (which norm, rotary or
quantization to fold where) rather than for a person to write each op.
On the shaped fixture (five layers) at one row on an L4, with the ops'
kernels forced, the decode step launches 265 kernels where the plain
graph launched 310, nine fewer a layer; the step time is unchanged
there (14.7 against 14.3 ms), since the L4 is bandwidth-bound at that
size. With the lowering's measured choices on the L4 it keeps the fused
bodies for `attn_q` and `shared_ffn`, and the count stays at 310; the
hero machine measures for itself. [TODO hero 7: the step with the ops
on the real model, kernels per layer and ms.]

**Direction: the planner chooses the kernels, in Go (hero run 8).** Today
two planners exist and the split is the wrong one. The coordinator's
planner (Go) decides placement, which machines and GPUs run which layers
and experts, and hands each executor its plan as named switches. The
choice of implementation per op and call shape is made in Python at
engine load: the registry's preference order and `applies` rules, then
the lowering timing the fused body against the kernel per call site on
whatever GPU it finds, cached on disk per GPU and shape. Nothing above
Python sees that decision, it costs the hero machine minutes of warm-up,
and it is untestable without a GPU. The change:
- **The substitution table is data.** Per call site (op, shape key,
  layer): the implementation name and its block parameters (the GEMV
  handover row count, the FFN front's width, the K step). The measured
  cost record behind it is data too: per (op, shape, GPU model,
  implementation), the microseconds and the kernel count, recorded by a
  measuring step that runs on a GPU once per card type and is checked in
  as fixtures.
- **Python applies a given table** (the "plan given" path, extended from
  switches to the table) and measures only when asked. Loading a model
  then runs no timing.
- **Go owns the search and the record.** The planner reads the cost
  tables and emits the substitution table, with the placement, in
  `LoadModelRequest.plan`. Its decisions are unit-tested against the
  fixtures with no GPU; a CPU-only Python test checks that a given table
  is applied exactly.
- The templates are what it chooses among: with the GEMV template's
  prologue and epilogue flags and the per-row kernel as its catalogue,
  "fold the norm and the quantization into this GEMV, or not" becomes an
  enumerable choice the planner makes from measurements, not an op a
  person writes.

**The roofline, and the production stacks.** A decode step at one row
reads every dense weight once and the selected experts once, so its floor
on one GPU is those bytes over the GPU's copy bandwidth. On an RTX PRO
6000 (copy measured at 1,460 GB/s) DeepSeek-V4.1-Flash reads 10.5 GB of
dense weights and about 4.5 GB of selected experts a step: a floor of
10.3 ms, 97 tokens/s. We measure against that floor and against the
production stacks on the same cards, not against our previous run. Hero
run 6 (Oct 30, four RTX PRO 6000, every expert on the GPUs) took 20.6 ms
a step at one row:
- 11.8 ms in the dense segments, against a floor of 7.2 ms. They launch
  2,089 kernels a step, and a kernel boundary inside a CUDA graph costs
  1.9 µs (measured on an A100), so about 4 ms of the step is boundaries.
- 8.8 ms in the experts, against a floor of about 3 ms. Each layer's
  experts sat on all four GPUs, so every layer went out to three GPUs
  and back, about twelve PCIe transactions a layer.

The production stacks on these cards (reported by others in Oct 2026,
on 250 W cards; decode is memory-bound, so comparable): SGLang's plain
step decodes at 36 tokens/s on eight GPUs; with DSpark speculation
SGLang reaches 112 tokens/s on four GPUs and a vLLM fork 247 to 252
(494 on code). Our plain step is faster than theirs. Our speculative
decode is three times slower than the vLLM fork's: 84 tokens/s at one
row, and at eight rows it accepted 52 of 2,270 drafts where one row
accepts a third of them.

**Layer sharding** (`VXPU_LAYER_DEVICES=0:9=cuda:0,9:20=cuda:1,...`).
Each GPU holds a contiguous range of layers whole: those layers' dense
weights and caches, and all of their experts. The token ids enter on the
first GPU, the chosen tokens leave the last, and in between the step
crosses GPUs once per range, carrying the hyper-connection stream. The
exchange between GPUs for the experts disappears. One GPU streams at a
time, which is the roofline we aim at first: one GPU's bandwidth at one
row. The four GPUs hold the model; they do not add bandwidth to one row.
Several batches in flight, one per range, use all four later. In the
code:
- the engine moves each dense segment's resident tensors to its layers'
  GPU in place, so the prefill and draft programs that share them follow
  (`_shard_layers`); a segment spanning two layers goes with the later
  one, and the head stays with the last;
- the host nodes (a layer's experts, an n-gram lookup) run with their
  layer, so their inputs and replies never cross;
- the dataflow runner runs each node on its GPU, with that GPU current,
  and copies step inputs, shared residents (norm weights, tables) and
  crossing edges there, once a step;
- the step graph captures one CUDA graph per GPU, the crossings as
  copies and flags between them, and the native loop reads the chosen
  tokens from the last GPU.

Measured on two L4s with the shaped fixture (five layers, 16 experts,
the released model's dimensions) at one row, every run token-identical
to the others:

| placement | step | kernels | copies | mailbox operations |
|---|---|---|---|---|
| everything on one GPU | 14.31 ms | 310 | 2 | 3 |
| dense on one GPU, layers 3 and 4's experts in a pool on the other | 14.06 ms | 294 | 8 | 9 |
| sharded: layers 0 to 2 on one GPU, 3 and 4 on the other, one graph per GPU | 13.92 ms | 310 | 12 | 30 |
| sharded, the dataflow runner without the step graph | 14.77 ms | | | |

The crossing between the GPUs costs nothing measurable at this size:
the sharded step is no slower than the single GPU's. The prefill program
turned out to hold its own copies of the dense weights (1.7 GB here,
10.5 GB on the real model), so the sharding moves every program's
tensors, not the decode's alone, and reports any state tensor read from
two GPUs, since its copy on the second would go stale. The real model
runs as `0:9=cuda:0,9:20=cuda:1,20:31=cuda:2,31:43=cuda:3` with each
range's experts on its GPU (67, 82, 82 and 78 GB; hero run 7).

The planner also generates the layouts a model could run in on a set of
machines and ranks them by the cost model (`planner.Candidates`, #409),
without running anything:
- **one machine:** it runs the step and holds every expert;
- **experts spread:** one machine runs the step, every layer's dense half
  on its GPU, and other machines' node servers serve the experts it
  can't hold, a network round trip per remote layer;
- **layer ranges:** each of a few machines runs a contiguous range of
  whole layers, dense half and experts, on its own GPU and RAM, so a
  token crosses the network once per range.

`vxpu explain --artifact DIR --machines g2-standard-48:2,c3d-standard-60:2`
lists them with their predicted parts. The cost model is calibrated from
measurement: the network wire per call, the GPU-CPU crossings, each
machine's dense and expert rates. A CPU's dense half (attention, the
compressor, the hyper-connections) runs at 6.65 GB/s on a c3d-standard-60
against 203 GB/s on an L4. So for DeepSeek-V4.1-Flash on a g2-standard-24
and two c3d-standard-60s, the model rejects moving whole layers to the
c3ds (1050 against 115 ms/token), and on two g2-standard-48s it picks a
layer range on each (66 against 97). The planner arrives at the layout
from the measurements, not from a rule written for it. The same holds
beyond the three shapes: a search over every layer's two placements
(`planner.Search`: random layouts and changes to the best, kept as in
simulated annealing) finds the same layouts, and occasionally better
ones the shapes miss. `vxpu explain --serve` shows it running.

vXPU can split one LLM across machines by layer (#387). Each machine runs
a range of whole layers. An engine started with
`VXPU_LAYER_RANGE=START:END` loads and runs only layers START to END-1:
their slice of the decode and prefill graphs (`dataflow.Graph.slice`),
their experts and n-gram tables, and each session's state for those layers.

The engine that runs the first range is the head. For each step, the head
runs its own layers, then sends the step's inputs and its layers' outputs
to the engine that runs the next range. `VXPU_REMOTE_RANGES` tells the head
where the other ranges are. Each of the other executors serves its range
on a port (`ServeRange`, `rangeserver.py`). Its server receives each step
as one message in the node wire format, tagged with the head's session id.
Once the range has captured its step graph, the server runs decode steps
in C++, with no Python.

The router chooses ranges when the cost model ranks them cheapest. It asks
the engine's executor where the step can be cut (`DescribeNodes`), loads
each range on its executor, and tells the head where the ranges are.
Ranges produce output bit-identical to one engine's, on the CPU and on the
GPU with the step graph.

On two g2-standard-48s, with layers 0-19 on one and 20-39 on the other,
DeepSeek-V4.1-Flash decodes at 78 ms/token (measured over 10 prompts),
against the 66 ms the cost model predicted. The network round trip takes
1.5 ms of that, less than the 2.2 ms predicted. Most of the gap comes from
the last range, which also runs the logits head; the cost model doesn't
charge that to it yet (#424).

Not every layer boundary can start a range (`cuts.py`): a cut must not
separate state a layer writes every step from the layers that read it.
DeepSeek-V4.1-Flash's layers 2, 8, 14 and 20 write compressed KV caches
read by the layers after them (layer 20's by every layer to 39), so its
step can only be cut before layers 1, 2, 8, 14 and 20, with about 72 KB
crossing a cut each decode step. The planner and the search only start
ranges there.

**Direction.** The planner chooses between plans by cost, the cost model
grows to time to start, cost to start and cost per minute (#349), and in
the end the planner owns all placement (§4, §6; #316, #387).

The planner is part of the coordinator, and the coordinator is Go, next
to the router: it manages cloud machines and runs across the cluster,
and Go is the usual language for that. So the machine database the
planner reasons over is Go too (`pkg/machines`, #394): the machines vXPU
knows, with their CPUs, GPUs, memory, bandwidths, local storage, prices
and what we measured on them (`vxpu machines` lists them). Each executor
describes the machine it runs on (the `Describe` RPC: its own probe), and
the router matches that to the database. The executors stay Python,
where torch and the kernels are, and report what they measure.

## 2. Models by name, and executors

A model is registered by name (`vxpu register REPO`). The router imports
its weights into modelstore, resolves the revision to a commit, and
compiles it in a Kubernetes Job running the executor image (`python -m
vxpu.compile`). The artifact is stored in modelstore under a tag naming
the commit and the executor image, so it was exported by exactly the
torch and forks the executors run. Clients then ask for the model by name.

Executors are of two kinds today:
- pods the router creates;
- external executors (a GPU VM, a Mac) running `python -m vxpu.server`,
  which the router dials.

Kubernetes keeps the bursty, CPU-only work: compile Jobs, modelstore, the
router. A machine that runs an executor is dedicated to it.

**Direction.**
- Executors and models become separate things: a pool of executors (VMs,
  pods), each holding several models, with placement in between, and each
  executor reporting per-model state (#385).
- An agent on each worker VM lets the engine manage machines directly:
  stage weights ahead of a load, serve them peer to peer, run the executor
  (#384).

**Languages: the agent is Go, the runtime stays Python (later C++), and
a process boundary separates them.** A worker machine runs two programs
with different jobs:
- **The agent** is the machine's control plane. It downloads weights and
  stages them on local SSD, serves them to peers, starts and supervises
  the runtime, and reports the machine and its statistics to the
  coordinator. That is files, HTTP, gRPC and processes, which is what Go
  and its standard library do well, and it shares code with the
  coordinator (the weightstore client, `pkg/machines`, the protos). It
  never loads compiled model code.
- **The runtime** runs the steps. Everything it calls is C or C++:
  libtorch and its AOTInductor package loader, the CUDA driver (graphs,
  stream memory operations), our CPU kernels (`csrc/`), RDMA verbs. Today
  it is the Python executor. With the step graph (§6), Python only
  launches a step and waits, so it can stay Python until measurements say
  otherwise. When it moves off Python, it moves to C++, which calls those
  libraries with no foreign-function layer. Go would be the weakest
  choice there: every call into C goes through cgo, C++ APIs need a C
  shim, Go's rules for sharing memory with C get in the way of pinned
  buffers and mailboxes, and its garbage collector can pause the step.
  Rust's C interface costs nothing, but the hard parts are calls into C++
  libraries, so it would add a toolchain for little gain.
- **The boundary between them is a process boundary**, not a foreign
  function interface: they talk over gRPC or a local socket, and the agent
  stages weights as files that the runtime maps into memory. So neither
  needs the other's language, the agent works the same with today's Python
  runtime and a later C++ one, and the C++ stays small: no HTTP client,
  downloader or process manager written in it.

## 3. Where weights live

Weights are content-addressed by sha256 at every tier:

| tier | what | code |
|---|---|---|
| bucket | modelstore's blobs (`gs://…/_blobs/<sha256>`) | modelstore |
| weights disk | a checkpoint's blobs on a GCE disk, attached read-only | `pkg/weightdisk`, `pkg/agent` |
| executor's disk | the local weight tier, filled from the bucket | `localblobs.py` |
| RAM, memory-mapped from local SSD, or read from checkpoint files | expert banks and n-gram tables, outside the graphs | `engine.py` (`external_dir`) |
| GPU | dense weights; hot experts (§4) | `engine.py`, `experts_gpu.py` |

The tables a token reads a few rows of (the n-gram tables, ~200 GB on
DeepSeek-V4.1-Flash) stay on disk: when local (on a weights disk or in
the local weight tier) and the engine runs without the step graph, the
engine reads their rows straight from the checkpoint files at their byte
offsets using io_uring, with no copy into `external_dir` and no
memory-mapping (#452). A pread fallback runs when io_uring is
unavailable. When the step graph is on, native packages require resident
tensors, so the tables are copied and memory-mapped (#470). Tables from a
remote source are copied into `external_dir` and memory-mapped.

**Weights disks (#407).** Today each executor VM copies its checkpoint
from the bucket before it can load a model. The copy is slow for two
reasons. Downloading uses the VM's CPU for TLS and checksums. And many
machines write to local SSD slower than they can download: a
g2-standard-24 copies at 0.83 GB/s, the speed of its local SSD. A weights
disk removes the copy. The checkpoint goes onto a disk once, and executor
VMs attach that disk:
- **Layout:** the disk uses the same layout as the executor's local
  weight tier, one file per checkpoint file (`blobs/<sha256>`), plus an
  index (`weights.json`). The executor reads the files in place
  (`VXPU_BLOB_DIR`).
- **Making one:** once per checkpoint, a disk factory, which is a
  short-lived VM, copies the files from the bucket onto a fresh disk. We
  then make a snapshot of that disk. A GCE snapshot is a global resource:
  a zone in any region can make disks from it. Its data, though, is
  stored in one location, by default a multi-region such as `us`.
- **Using one:** the coordinator makes a disk from the snapshot in the
  executor's zone and attaches it read-only. Many VMs can attach the same
  Hyperdisk ML disk at once. The machine's agent (`vxpu-agent`, the start
  of #384) mounts the disk and checks it against the index.

For now, `vxpu disk build`, `attach` and `detach` run their GCE operations
through the gcloud CLI; #443 replaces it with the Compute API.

**What a disk saves.** We measured both ways of getting weights onto a new
VM on Oct 23 2026. The VM was a g2-standard-48 in us-central1-b, and it
loaded layers 20 to 39 of DeepSeek-V4.1-Flash (154 GB) into a cold
engine. Time to first token here means the time until the engine is
loaded, not counting the VM's boot or the prompt's prefill:

| step | copy from the bucket | weights disk |
|---|---|---|
| put the checkpoint on the VM | 338 s: 511 GB to local SSD at 1.5 GB/s | 6 s: attach 5.7 s, mount 0.3 s |
| load the engine | 221 s | 230 s |
| time to first token (sum) | about 9 minutes | about 4 minutes |

Making the disk from the snapshot took 476 s, once for the zone. Raw
reads on this VM reach 2.94 GB/s from local SSD and 2.52 GB/s from the
disk. The disk stops at 2,400 MiB/s although it is provisioned for
13,080 MB/s, so the limit is probably the VM's (inferred). The engine
reads at 1.35 to 1.5 GB/s from either source, so today the engine, not
the storage, limits how fast a model loads.

**Disks are caches.** The bucket holds the canonical copy of every
checkpoint, and we don't copy it per region or per zone. It should be one
multi-region bucket. Today it is a regional bucket in us-central1
(`modelstore/README.md`), which is enough while everything runs there.
Snapshots and disks are caches of the bucket:
- A snapshot costs its stored bytes. It saves rerunning the factory,
  which copies the whole checkpoint from the bucket.
- A disk serves one zone. It costs its capacity and provisioned
  throughput by the minute, and every VM that attaches it shares them.

The planner decides when each cache is worth its cost (#349). It makes a
disk in a zone when the VMs it expects there, each ready about 5 minutes
sooner, are worth the disk's cost. It can make the disk ahead of demand,
so that a scale-up doesn't wait 476 s for it. It deletes a disk that sits
idle, because the snapshot can make it again in about 8 minutes. A VM
without a disk copies its checkpoint from the bucket, which suits a VM
that starts once.

**Open questions:**
- **Where the bucket and snapshots are stored.** A VM that copies from
  a bucket in another region, or a disk made far from its snapshot's
  storage location, probably runs more slowly and pays for inter-region
  transfer; we haven't measured either. When we run outside us-central1,
  the bucket moves to a multi-region, and we would keep a copy of the
  snapshot near each geography. On
  GCP, that is a snapshot or image with its own `--storage-location`.
  AWS works the same way at a different scope. An EBS snapshot belongs
  to one region, so we would copy it to each region we use. On both
  clouds, disks belong to one zone.
- **Volumes that fill as they're read.** On AWS, a volume made from a
  snapshot loads each block from S3 the first time it is read, unless
  Fast Snapshot Restore is on. A new volume is not at full speed
  straight away.
- **Mapping straight from the disk.** The engine copies n-gram tables to
  local SSD before it memory-maps them. With a weights disk, it could map
  them from the disk instead (#452).

**Direction.** Hierarchical placement: RAM as a cache over SSD when the
expert banks don't fit (#348, #259), and weight blocks shared across
processes on a machine (#322).

## 4. The experts

A mixture-of-experts layer's banks stay on the host. They are the
authoritative copy, and the CPU kernels compute any expert from them.
Expert use is skewed, so the hottest experts are also held on the GPUs,
one slot pool per GPU (`experts_gpu.py`). An expert op computes a token's
held pairs on the GPUs and the rest on the CPU at the same time. Because
weights are immutable, a GPU slot is a copy: dropping it costs nothing,
and no copy goes back.

Which experts are hot is a placement: the hottest experts across all
layers, by the engine's own statistics (§5), that fit each GPU's budget.
The planner chooses it at every load by name (§1), with each GPU's budget
derived from the machine: its memory, less headroom, and on the engine's
GPU less the dense weights and caches (2.5x the dense bytes, measured on
DeepSeek). `expert_stats --placement-gb` does the same offline with
budgets a person gives.

Before the GPUs, the planner decides whose RAM holds each layer's
experts. The engine keeps the layers that fit its budget (85% of its
RAM, less 16 GiB, less the dense weights when it has no GPU). The next
layers go to other free executors, each up to its own budget. Each of
those executors serves the experts nodes of its layers from a native
node server (§6, the `ServeNodes` RPC), and the router wires the engine's
mailboxes for those nodes to it (`VXPU_REMOTE_NODES`, which only the
router sets). Layers that fit no machine stay on the
engine, paged from its disk. The GPUs then hold the hottest experts of
the layers the engine keeps. So a model that fits no single machine runs
on several smaller ones: GPUs with 24 GB, or none, and CPU machines for
the experts that do not fit. `expert_sim` replays
recorded traffic against placement policies before any is built: static,
decayed, informed by the prompt, and the oracle as a bound.

With `gpu_experts_moves`, the placement then follows the traffic
(`experts_gpu.Mover`). Each layer scores its experts by all traffic so far
mixed with the current request's own selections. At each request's
prefill it plans swaps of cold experts that outscore held ones, and a
worker copies them while the prefill computes. This is off by default:
on DeepSeek, the static placement already serves most pairs, and moves
during decode cost more than they saved.

**Direction.**
- A node can be in several places at once because its weights are
  immutable: experts on the CPU and a GPU, dense layers on both GPUs
  (#387).
- Experts on another machine, in another process that holds their banks
  (§6).
- Open question: who decides placement. A central planner can coordinate
  across a cluster of executor nodes, while an executor deciding locally
  responds far faster. A likely shape is both: the planner sets policy
  and budgets (what may be held where, how many moves may be in flight),
  and each executor reacts within them.

## 5. Statistics

The router records every request served by name: the generation's sizes
and timings, and the engine's execution telemetry (on every request by
default; every Nth with `--telemetry-every`). That
includes which experts each layer selected, and each segment's and host
op's time. Records are batched into modelstore (`vxpu-stats/<model>`, one
tag per day). `vxpu stats` summarizes them, and `expert_stats` and
`expert_sim` read them. The `expert_sim requests` subcommand replays
recorded traffic against placement policies: static, adaptive, prefill,
and the oracle bound. The `expert_sim steps` subcommand replays ordered
selection traces per decode step through cache policies, storage tiers,
and prefetch predictors (#348, #447).

The planner predicts each plan's decode time (`pkg/planner/cost.go`,
#349): decode is memory-bound, so each part costs its bytes over the
bandwidth that reads them. The dense weights a step reads go over the
GPU's GEMV bandwidth. Each layer's top-k experts go partly over GPU
bandwidth and partly over the CPU expert kernel's, in parallel, or over
another executor's CPU and a network round trip. Each host op adds a
fixed crossing cost, and without the step graph each dense segment adds
Python's launch. The bandwidths come from the machine catalog: measured
where we measured them (an L4's GEMVs reach 252 of 300 GB/s, bf16 and
fp8 alike), otherwise a measured share of the memory bandwidth. The
constants are fitted to the hero run, which the model reproduces within
9% (35.9 ms/token with every expert on the CPU, 26.4 with 68 GB on a
GPU). `vxpu explain` shows the prediction by part, and telemetry
measures what ran.

**Why.** A database keeps statistics about its tables (row counts, value
distributions) because its optimizer can only choose between plans by
predicting what each will cost. vXPU is building the same kind of
prediction system: given a model, a machine and a candidate plan, predict
the step time, the memory and the numerics. Statistics have two jobs.
They are its inputs: which experts real traffic selects, and how long
each segment and host op takes on each machine. They also check it: a
prediction for a plan that then ran can be compared with what it
measured.

**Direction.** The statistics feed the planner (§1): placement chosen
from measured traffic, and plans compared by predicted cost on a model of
the hardware (#349). Every plan that runs reports predicted next to
measured, so we can see how far the predictions are off.

## 6. Executing a step

A decode step's graph is split at its host ops, the calls that leave the
GPU: the routed experts, the n-gram lookups (`segments.py`). The dense
segments between them compile and replay in CUDA graphs. The host ops run
eagerly, called directly rather than through torch's dispatcher. Each
host op is where the GPU waits for the CPU, and where Python launches the
next segment.

The split step is also an explicit dataflow graph (`dataflow.py`,
`VXPU_DATAFLOW=1`):
- **nodes:** the segments (dense, experts, engram);
- **edges:** the tensors one segment passes to another;
- **resident tensors:** weights and caches, which never travel.

The reference runner gives every edge an inbox at its consumer's device,
allocated once at a fixed address, and fires a node when its inboxes are
full. A placement maps node names to devices; the host-op nodes can run
away from the engine's device. It computes exactly what the split module
computes.

**Direction: a dataflow executor with mailboxes (#372).**
- Every node is placed by the planner on a GPU, the CPU or another
  machine, with its inputs arriving through inboxes its producers write
  into.
- The routed experts are nodes too: a token's (token, k) pairs go to the
  experts' owners, and the replies are summed in a fixed order, so the
  result does not depend on who answered.
- On one machine, the whole step is one CUDA graph (`stepgraph.py`):
  host ops become mailboxes the GPU waits on itself (a flag in pinned
  memory). The step host, a native thread (#409), answers them: it sees
  a node's flag, runs the node on the pinned inputs (its native package,
  or a node server through the native client), writes the reply and
  raises the flag back. No Python runs between the GPU's flag and the
  reply, and the step host times each crossing.
- Code on the data path, per message or per step, is native from the
  start, with its timers built in: we don't build a Python version to
  measure first and replace later. Python sets things up (graphs,
  lowerings, packages, residents) and gets out of the way.
- Across machines, a node runs on a node server (`nodeserver.py`,
  `csrc/node_server.cpp`, #394): the node compiled to an AOTInductor
  package, a host op lowered to the native kernels (`nodes.py`), its
  residents loaded once and passed by address (so every layer's experts
  node is the same package, compiled once), answering its mailbox over TCP (`tcp://`, a
  small binary framing) with no Python per message. The engine's side is
  a native client: the step host calls it directly, and node by node,
  the dataflow runner and prefill reach it from the host op. The
  coordinator places nodes by name: an executor describes the decode
  step's nodes (`DescribeNodes`) and serves those it is given
  (`ServeNodes`). Remote nodes are bit-identical to local ones. RDMA
  verbs, with Soft-iWARP where there is no RDMA NIC (#386), and shared
  memory are further mailbox addresses for the same messages.
- Chains of mailboxes that traffic always takes compile as one unit
  (#387).
- The coordinator places subgraph nodes by name, on CPUs, GPUs, SSDs and
  nodes across a cluster, and wires up the mailboxes between them,
  across machines too.
- A node fires when all of its inputs have arrived. For a dense node that
  is a fixed set of inboxes. For an expert that is its own node, the
  messages per step vary (zero to n tokens, from one router or, with
  sessions batched, several), so "all" needs an end marker: every router
  feeding the layer has finished posting for the step, or a batch is full
  or a deadline passes. Until experts batch across sessions, one experts
  node per layer, with the owners behind it, avoids the question.
- The coordinator (the engine layer) shows the graphs as placed (which
  nodes are on which CPU, GPU, SSD or machine) with the traffic and
  latency through each node and mailbox. It builds this from the
  statistics the executors report. Only the coordinator sees across
  nodes and mailboxes; executors grow simpler about placement, carrying
  out what they are given and reporting what they measure. The telemetry
  already records most of it: each segment's and host op's time, and the
  experts each layer selected. The site is Go, served by the coordinator
  (#331).

**Direction: mailboxes as message rings.** Today a mailbox holds one
message: the step graph's flag in pinned memory, raised once per node per
step, and a TCP frame to a node server. So the producer of a step waits
for every reply before it can go on, and each transport has its own
code. A mailbox becomes a ring instead:
- A ring is a single-producer, single-consumer queue of fixed-size slots
  in one region of memory. The producer writes a slot's payload, then
  publishes it by writing the slot's sequence number (a release store);
  the consumer polls the sequence number (an acquire load), reads the
  payload, and hands the slot back by advancing its own sequence. When
  the ring is full, the producer waits. A GPU producer publishes with a
  system-scope release store from a kernel, as the flags do today.
- The kind of memory is the address: pinned host memory between a GPU
  and the CPU in one process, shared memory between processes on one
  machine (a node server without TCP), RDMA-registered memory between
  machines (an RDMA write of the payload, then of the sequence number;
  Soft-iWARP where there is no RDMA NIC), GPU memory between GPUs. The
  producer and consumer code is the same for all of them.
- Depth lets work overlap. With several slots, a GPU can go on to other
  rows' dense work while the experts answer a layer, a node can answer
  one message while the next arrives, and experts streamed through a GPU
  (cycling, §6) arrive in slots while the previous ones compute. Replies
  still sum in a fixed order: a reply carries its pair's index.
- Each slot's header carries its timestamps (written, read), so every
  crossing is timed without extra code.
- The rings are native code (C++ for the CPU side, a kernel for the GPU's
  publish), with Python only creating them. Today's flags are rings one
  slot deep, so the step graph moves over first, unchanged in behaviour;
  then node servers over shared memory, then RDMA. When the step graph
  on rings is the default, the host-op segments that run in Python
  between compiled segments (42 a decode step for DeepSeek-V4.1-Flash)
  go away.

**Three scenarios.** Where the experts live decides what a step costs,
and three settings want different plans:
1. **A service:** enough GPUs, maybe across machines, to hold every
   weight in GPU memory (DeepSeek-V4.1-Flash: 10.5 GB dense, 288.8 GB of
   routed experts); high throughput at low latency.
2. **A personal device:** a GPU that holds only the dense layers, the
   experts in host memory; high throughput for one session.
3. **Many background agents:** many sessions, too little GPU memory;
   high aggregate throughput, at the expense of latency. One plan cycles
   the experts through the GPUs, layer by layer, for the tokens waiting
   on them. A session's decode state is small (18.4 MB at 4096 tokens),
   so thousands of sessions fit; the cycle's length is the cost. At the
   host-to-GPU bandwidth measured below, streaming every expert takes
   about 2.6 s (estimated), shorter when a hot set stays resident.

They are policies over one mechanism: which experts are on which GPU
during a step, and which tokens wait for them. The hot-expert placement
and its mover are the static and slowly moving ends; cycling is the
fast end.

Measured on Oct 29 on one g4-standard-192 (4 RTX PRO 6000, an AMD EPYC
9B45 with 96 cores), 37 to 47-token chat prompts, 64 new tokens. One
session (tokens/s, decode only):

| plan | ms per token | tokens/s | with drafts (S = 6) |
|---|---|---|---|
| experts on the CPU (scenario 2), default path | 39.7 | 25.6 | 27.0 |
| every expert on the GPUs, each layer split across 4, default path | 41.9 | 24.2 | 49.7 |
| every expert on the GPUs, each layer on one GPU, default path | 24.2 | 42.0 | 49.4 |
| every expert on the GPUs, step graph | 22.1 | 46.0 | 53.2 |

Several sessions at once (decode tokens/s): with every expert on the
GPUs split across them, 195 at 9 rows (46.4 ms a step), 236 with drafts
at 8 rows, and 473 at 32 rows (69.7 ms); each layer on one GPU, 154 at
9 rows; experts on the CPU, 77 at 9 rows.

What the numbers say:
- On the default path, fixed costs dominate a small step: each layer's
  experts are a host op, with a Python segment and a synchronization
  around it, and splitting a layer across four GPUs adds copies to and
  from three of them. The step graph (one CUDA graph a step, the host
  ops as native mailboxes) removes most of it: 22.1 ms a token with
  every expert on the GPUs, and with half of them on the CPU too.
- Placement trades latency for throughput: a layer whole on one GPU is
  faster for one session (24.2 against 41.9 ms), a layer split across
  the GPUs faster for many (195 against 154 tokens/s at 9 rows), since
  its expert work runs on four GPUs at once. Keeping every GPU busy with
  whole layers needs several batches of rows in flight, which is what
  rings with depth allow.
- Drafts pay where an extra token costs little: 2.05 times with every
  expert on the GPUs and the default path's expensive step, 1.16 times
  on the step graph, 1.05 times with the experts on the CPU, whose
  verification grows with the tokens.
- The machine: memory reads at 474 to 523 GB/s; one token's experts on
  the CPU take 0.39 ms a layer on 48 or more threads (290 GB/s, 56% of
  the read rate; one core reaches 14 GB/s); host-to-GPU copies at 56.9
  GB/s a GPU, 113 GB/s for all four at once; no steal time and 2.7 to
  4.1 GHz throughout, and no run as slow as the outliers of Oct 27 (a
  g4-standard-96).

**Batched decode.** Decode is bound by memory bandwidth, so stepping
several sequences together reads each dense weight once for all of them.
`export --batch B` exports the graphs with B rows. Every state tensor
gains the leading dimension, and the decode graph takes one position per
row (`cache_position [B]`), so each row decodes its own sequence at its
own position. The forked model writes its caches (the window ring, the
compressor, the indexer, the n-gram hash) per row.

We measured it on DeepSeek-V4.1-Flash on one g4-standard-96 (2 RTX PRO
6000) on Oct 26 and 27, with hot experts on the GPUs (7,712 experts, 145
GB, placed from other prompts' telemetry), 32 distinct prompts of 37
tokens, and 64 new tokens, every row at the same position:

| batch | ms per step | tokens/s | dense (ms) | experts (ms) |
|---|---|---|---|---|
| 1 | 45.7 | 21.9 | 24.9 | 17.0 |
| 8 | 90.7 | 88.2 | 42.6 | 44.5 |
| 16 | 126.8 | 126.2 | 41.3 | 75.9 |
| 32 | 197.9 | 161.7 | 49.4 | 130.3 |

(Batch 1 and 8 used 40 to 42-token prompts.) Throughput at batch 32 is
7.4 times batch 1's, for a step 4.3 times as long. The dense layers
barely grow with the batch. The experts the CPU computes, the ones not
held on a GPU, are the next limit: 111 of the 130 ms at batch 32.

**Continuous batching.** Requests come and go, and each takes a row
(`_Rows`):
- A prompt is prefilled alone, in row 0 (the prefill graph has one row),
  and then copied to a free row. A request decoding in row 0 has its
  state set aside for the prefill and put back.
- A scheduler thread steps every active row together, and admits waiting
  requests between steps (`_Scheduler`). `Engine.generate` and `chat` go
  through it for an artifact with rows; `generate_rows` runs a list of
  prompts to completion, for benchmarks.
- A row with nothing to decode feeds its last token again. Its next
  prefill replaces whatever that writes, so a step needs no mask.

Prefill matters here, because it stalls every row. The CPU's expert
kernels group each expert's tokens and work in tiles, so one path serves
a decode step and a prefill alike. Until Oct 28 a separate path took
anything over 32 token rows; it dequantized every expert it touched, and
a 45-token prefill took 7.5 s, which dominated continuous batching end to
end (8 rows: 7.9 tokens/s). On the decode kernels it takes 0.37 s. The
separate path lost at every length measured (on the hero VM, every
expert on the CPU: 626 tokens 4.8 s against 16.7 s; 2414 tokens 18.3 s
against 29.7 s), so it is gone.

Measured on Oct 27 on the same VM, with prefills on the decode kernels, 37 to 47-token
chat prompts arriving at once, and 64 new tokens:

| rows | requests | prefill per prompt (s) | ms per step | decode tokens/s | end to end tokens/s |
|---|---|---|---|---|---|
| 1 | 1 | 0.4 | 35.6 | 28.4 | 24.3 |
| 9 | 8 | 0.37 | 84.0 | 96.1 | 61.3 |
| 33 | 32 | 0.37 | 177.9 | 183.8 | 89.4 |

With the old prefill path, the 8-request run spent 59.5 s in prefill. At 32
requests, the prefills (11.8 s, one after another) are the larger part
of the wall time. Every row computes in every step, so one request on a
9-row artifact pays for nine (95.5 ms a step). About half the processes
on that VM ran the decode step three times slower (300 ms at 8 rows),
steady within a process and whatever the code; we have not found why.

**Speculative decoding.** A decode step can take S tokens per row
(`export --step-tokens S`): the row's next token and S - 1 drafts. The
row keeps each draft that equals the token its own sampler picks at that
position, up to the first that does not, and then the token it picked
there. Every token is the sampler's pick given the tokens before it, so
the drafter changes the speed and not the output, for any sampler.
- A rejected draft leaves values in the caches past the row's position.
  Caches indexed by position or by group are rewritten when the next
  step feeds those positions. The window rings and the compressor's
  partial group are indexed by position modulo their size, so they get
  S - 1 slots of slack (`window_slack`): a rejected write never lands on
  a position the row still reads.
- Two drafters share one interface: the model's own, and prompt lookup
  for any model. DeepSeek-V4.1's own is DSpark: three stages (`mtp.*`,
  7.9 GB) that draft a block of 5 tokens from the target's hidden states
  at layers 37 to 39. `export --draft` makes it `draft.pt2`, a third
  program next to prefill and decode, and sets S to 6.
- The drafter reads the target's hidden states from state. The prefill
  and decode graphs write them into the drafter's window rings; the
  draft graph reads the rings and writes nothing. The engine sees three
  programs and their shared state, and knows nothing of DSpark. Its
  routed experts are external like the backbone's (layers 40 to 42).

Measured on Oct 27 on the same VM, with S = 6, the drafter's experts on
the CPU, and the same prompts:

| rows | ms per step | tokens per row and step | decode tokens/s | without drafts |
|---|---|---|---|---|
| 1 | 79.3 | 2.56 | 32.8 | 28.4 |
| 8 | 225.4 | 2.94 | 90.4 | 96.1 (9 rows) |

The draft graph takes 9.8 ms of a step. Verifying 6 tokens a row takes
the rest, and it grows with the token rows the way a batch does, because
the experts dominate. So drafts pay only where a step has rows to spare:
1.15 times at one row, nothing at eight. Per prompt, 33 to 65% of the
drafts were accepted (2.6 to 4.3 tokens a step). Next: verify fewer
drafts when the drafter's confidence head is low, and place the
drafter's experts on the GPU.

**Numerics across batches.** On the GPU, which kernel runs and how it
sums depend on the number of token rows (the head's Triton kernel takes
at most 8; the fp8 GEMV, 4). So a sequence's greedy tokens depend on
what it was batched with. Over 8 prompts, batched runs left the batch-1
run at a near-tie after 3 to 49 tokens, and both continuations read
well ("main differences TCP vs UDP" against "TCP vs UDP main
differences"). A row's tokens do not depend on when the other requests
arrive: the same prompts, all at once or one every two steps, gave the
same tokens. On the CPU the tiny models are exact, and the selftest
checks batched and speculative decoding token for token against
decoding each prompt alone.

Still to build:
- **Prefill alongside decode:** a long prompt stalls every row for its
  prefill. Steps of several tokens could carry a prompt's chunks
  alongside the other rows' decoding.
- **Ranges:** a range server message that carries several rows.
- **Idle rows:** every row computes in every step, so an artifact sized
  for the peak load is slow for one request.

## 7. Numerics

Every implementation declares whether it is bit-identical to the model's
own computation or equal up to rounding, and the selftest holds it to
that: token-exact against a reference, and with logits compared, for
every tiny model. Where a choice changes the result (an expert computed
on the GPU instead of the CPU rounds differently), that is a property of
the plan, recorded with every result the engine returns.

**Direction.** Sums whose order does not depend on placement (§6), so
that moving work between devices changes the speed and not the answer.
