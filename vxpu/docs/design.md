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
loads the model again. `LoadModel` accepts the artifact and loads in the
background. The executor reports its load state through `GetLoad` (`NONE`,
`LOADING`, `READY`, `FAILED`), along with any failure message and bytes
rehydrated so far (#446). Callers check that state rather than matching
error messages.

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
on a port (`ServeRange`, `rangeserver.py`). While an engine loads its weights,
`ServeRange` reports its load state (`LOADING`, `READY`) in its response. Its server
receives each step as one message in the node wire format, tagged with the head's
session id. Once the range has captured its step graph, the server runs decode
steps in C++, with no Python.

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

## 7. Numerics

Every implementation declares whether it is bit-identical to the model's
own computation or equal up to rounding, and the selftest holds it to
that: token-exact against a reference, and with logits compared, for
every tiny model. Where a choice changes the result (an expert computed
on the GPU instead of the CPU rounds differently), that is a property of
the plan, recorded with every result the engine returns.

**Direction.** Sums whose order does not depend on placement (§6), so
that moving work between devices changes the speed and not the answer.
