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

**The kernel library.** Physical kernel variants live under
`kernels/<op>/<variant>.py`. Each variant self-registers using the `@variant`
decorator, naming itself by its module path. The kernel table names a variant by
its file stem. Discovered file variants register after the op's reference
implementation, so adding a file changes no engine's default selection. Every
variant declares its numerics class: *bit-identical* (shape, dtype, and bits
match the reference via `torch.equal`) or *rounding* (relative deviation
at most 0.05 for floating outputs, and integer outputs equal).

**Kernel tooling.** Developer tools for building and measuring kernels live
apart from the serving path in `kernel_tools/` (`check.py` and `score.py`).
The serving engine imports `kernels/` (the C++ kernel loader, `@variant`, and
variants) and never imports `kernel_tools/`. A reader can tell from the directory
alone whether Python code serves requests or supports kernel development.
The per-op input builders and score shapes stay in `kernels/<op>/__init__.py`
and are imported by discovery when the engine starts, keeping an op's definitions
and measurement inputs together where an agent adds a variant.
The check `python -m vxpu.kernel_tools.check` runs the variant against the reference on
specified shapes, computes tensor bytes using the ledger's formula via
`lower.site_bytes` on a one-node graph (the floor counts weights plus
activations, not state), and measures warm-cache latency (`timing=warm`).
The caller supplies the card's memory bandwidth from the machine database;
Python holds no card table. The check computes the floor latency and floor
fraction from this supplied bandwidth, printing a summary line for each shape.

Variants come from people, codebot issues, and auto-optimizer goals under
`.goals/create-cpu-kernels.md` and `.goals/create-gpu-kernels.md`. The scoring
tool `python -m vxpu.kernel_tools.score` measures the library on CPU or GPU.
On CPU (`./vxpu/dev/goals/cpu-kernel-score`), it measures ops by summing the base-2
logarithm of each op's best speedup over its reference. A variant that is no faster
than the reference contributes zero. Summing the logarithm makes a percentage gain,
such as 10%, worth the same across every op regardless of its baseline speedup.
This formula prevents an agent from climbing the metric by adding trivial variants
to new ops without making anything faster, and pairs with an absolute improvement
threshold of 0.1 in `.goals/create-cpu-kernels.md`. A real 10% speedup on any op
adds log2(1.10) ≈ 0.1375 bits, clearing the threshold.

To make the improvement threshold hold against noise on unchanged code, the timing
helper calibrates repeats to target 100 ms per measurement (up to 50 repetitions)
on CPU and evaluates each implementation's 20th percentile latency across 20 rounds.
The 20th percentile discards transient positive latency outliers within a run while
avoiding the single-sample noise of a minimum. Under steady conditions, run-to-run
noise stays safely below the 0.1 threshold, while competing work on the machine can
shift the score by several times the threshold because compute-bound parallel kernels
and memory-bound references scale differently under load. Therefore, a score compares
only with another measured on the same machine under the same load; the runner must
evaluate baseline and candidate back to back in one invocation (or re-measured with
`./vxpu/dev/goals/cpu-kernel-score`).

On GPU (`./vxpu/dev/goals/gpu-kernel-score`), variants run on a card and are
measured against the card's memory-bandwidth floor. The machine database records
each card's nominal memory bandwidth and, where we have measured it, the rate a
plain read achieves. Kernel checks compute the minimum time to read an op's
bytes from the measured rate when there is one and say which they used. The read
op's shape is large enough that the card's cache does not matter. A fraction
above 1 means the op's bytes fit in the card's cache when timed warm, or the
recorded rate is out of date. The planner's cost model uses the nominal figure
today, scaling it by an op efficiency factor in `pkg/planner/cost.go`. For each
op, active variants that run faster than the reference contribute the fraction
of the card's memory-bandwidth floor gained over the reference
(`max(0.0, best_frac - reference_frac)`), while variants matching or slower
than the reference contribute zero. This formula is continuous at the reference
speedup and prevents timing noise from jumping an op's score. The op's score is
its best variant's gain over the reference, and the goal metric sums these gains
across ops, paired with an improvement threshold of 0.05 in
`.goals/create-gpu-kernels.md`. A consequence of measuring against the physical
floor is that a 10% speedup on a kernel already near the floor (for example, at
0.82 adds 0.082) clears the threshold, whereas a 10% speedup on one far from the
floor (for example, at 0.09 adds 0.009) does not, favouring closing the gap on
kernels already near hardware limits. For ops whose tensors span only a few
kilobytes (such as `vxpu::hc_sinkhorn` and `vxpu::rotary`), kernel launch
overhead rather than memory traffic sets latency, making the memory-bandwidth
floor fraction negligible. Conversely, warm measurement of a tensor that fits
within the card's on-chip cache (such as `vxpu::fp8_grouped_linear`) is not a
memory-bandwidth floor measurement because repeated warm iterations read from
on-chip cache rather than DRAM. On CPU,
the `cpu_kernel` variant of `vxpu::fp8_grouped_linear` reads the packed fp8
weights and block scales directly without dequantizing them into memory,
eliminating tens of megabytes of intermediate tensor allocations and cache
churn on each decode call. On a machine
with no card, the GPU score fails with
a clear message and never falls back to CPU. A variant that cannot run on the
machine is reported as skipped on its own line and excluded from the op's score.
The planner, not the score, decides what runs in an engine. The score guides kernel
authoring, while the planner selects variants from measurements on the target machine.

**Cross-card sum (`vxpu::allreduce`).** `vxpu::allreduce` sums the partial
results of several ranks into a single result. Each rank produces a partial
tensor, and the op computes their elementwise sum.

The reference sums the partials in rank order using 32-bit floating-point
addition. It converts the sum back to the input data type at the end. Summing in
rank order guarantees that every rank produces identical bits, regardless of
when partials arrive from peers.

Variants may differ in how they transfer and accumulate partials across cards.
Today, variants run on buffers that sit on one card for measurement. The forms
that transfer bytes between cards arrive in subsequent moves. Variants differ in:

- **Communication pattern:** A variant can exchange full buffers directly
  between cards. Alternatively, it can split the reduction into reduce-scatter
  and all-gather phases to transfer fewer bytes across the interconnect.
- **Order of summation:** A variant that uses a parallel GPU reduction kernel
  reassociates floating-point additions across threads. It declares the
  `rounding` numerics class because reassociated additions round differently
  from strict rank order.
- **Precision:** A variant can narrow data to a smaller format on the wire
  (such as bfloat16) to save interconnect bandwidth. Because narrowing drops
  precision, the variant declares the `rounding` numerics class and requires an
  explicit plan choice so that the planner never selects it by default.

The coordinator can set the plan's switches. A model registered with
`vxpu register --plan VXPU_NAME=VALUE` keeps that plan in modelstore, and
the router sends it with every load of the model (`LoadModelRequest.plan`,
#394). Load and registration requests carry the physical plan document
itself (`PlanFile plan_file`). The router forwards it to the executor unchanged.
Client requests cannot set switches that belong to the router
(`VXPU_REMOTE_NODES`, `VXPU_REMOTE_RANGES`, `VXPU_LAYER_RANGE`). The router
refuses any client request naming them with `InvalidArgument` naming the switch.
The router's own use of those switches when wiring executors continues to
work. A client request on `LoadModel` cannot supply a switch map; the typed plan
message (`plan_file`) carries the physical execution plan instead. A load for a
model already serving under a different plan updates the desired state; the
reconcile loop converges to it by reloading the executor with the new plan and
repeating verification. The executor reads those switches ahead of its environment
and refuses any it may not take: paths and endpoints on its own machine stay
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
compiler. Dense segments compile ahead of time with AOTInductor (`aoti.py`),
cached per artifact, plan and machine. The engine builds or loads every
segment's package during load, before it reports ready, using example arguments
from the split graph. An engine that has reported ready guarantees that every
dense segment package it will run is loaded into memory, and every local
exchange node's package is built, extracted, and shown to load; requests
build no packages (the step host loads its own instance when the step graph is
captured). When packages are missing from the cache, the cold load
builds them concurrently across worker processes (`spawn`), up to the number of
builds given by `VXPU_AOTI_PARALLEL_BUILDS` (defaulting to the machine's CPU
threads, or 1 to build sequentially). Segments that share an identical cache
key (such as repeated transformer layers) are grouped so each package is built
only once. Builds stay deterministic: Inductor deterministic mode is enabled in
every builder, ensuring packages have the same cache keys and produce identical
execution results whether compiled in parallel or alone. Builders write to
staging files with process identifiers and move them into the cache with an
atomic replace, so concurrent loads on the same cache directory both come up
without reading half-written packages. If a build fails, all active worker
processes are terminated, temporary files are removed, and the load fails
naming the segment.

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

### The plan documents

Documents cross the boundary between the Go planner and the Python executor. In database terms, a plan is what `EXPLAIN` prints: what the planner decided to do. The same plan with the measurements of a real run filled in is what `EXPLAIN ANALYZE` prints: the same tree, with the actuals beside each node. Both the plan file and the step plan work this way: one set of messages, with an execution-stats message (`stats`) that is absent until a run fills it in:
- **The kernel table** (`KernelTable`) is what the planner hands an engine: the GPU it targets, the kernels' knobs, and the implementation body to run for each call site. Every entry in the table must match at least one call site of the model being planned, or the planner refuses the table as an error naming the unused entry.
- **The plan file** (`PlanFile`) is one physical plan (the `EXPLAIN` output): the optimizer's choice of operator and knobs for every call site of one model and query shape, derived from the table. It answers "which kernel runs at this site for this plan". This is what `vxpu plan emit --artifact DIR --out plan.pb` emits. It records `artifact_md5` (a checksum of the manifest and binding bytes), `tree_commit`, and the model's call sites across five kinds (`SiteKind`): dense GEMV, expert op, sum site, attention, and drafter. Each site records its tensor shapes, supported row counts, and current default strategy.
- **The kernel record** (`KernelRecord`) is the document that captures compiler lowering rather than run-time execution measurements. After lowering (`VXPU_PLAN_RECORD`), the record captures the chosen body and inlined Triton kernels. The Go planner reads call sites directly from the exported graph in the artifact, planning every call site without a description step from Python.
- **The run record** (a run record is a plan file with its measurements filled in) is the `EXPLAIN ANALYZE` output: what one execution of a plan measured, read back by the optimizer to correct the table. Just as a database `EXPLAIN ANALYZE` fills actual numbers beside estimates in the same plan tree as `EXPLAIN`, the run record shares the plan's message structure (`PlanFile`). The executor preserves the plan's planned choices and fills in execution measurements under `stats` sub-messages. On `PlanFile`, the planner writes `artifact_md5`, `tree_commit`, `sites`, and `plan_hash` (the plan's content identity, computed once in Go via `kernelplan.PlanHash` and traveling with the plan so the executor reports it back and the router compares it without re-hashing non-canonical protobuf serialization across languages); the executor fills `stats` (`PlanExecutionStats`) with `plan_hash` (the hash of the plan file this record was filled from), effective knobs, median step latency in milliseconds, tokens a stream a step, accepted draft tokens a step, verification check results, per-card peak memory on CUDA, and per-class device times when in-graph phase stamps are active. On `PlanSite`, the planner writes `id`, `kind`, `shape`, `row_counts`, and `strategy`; the executor fills `stats` (`SiteExecutionStats`) with `measured_us` and reports executed operator implementation bodies under `executed_bodies` (`ExecutedBody`). For host ops whose choice may vary across calls, the run record lists all bodies that ran alongside the row counts they observed without inventing a single answer. A call site in the record is in one of three states: measured and executed with its bodies listed, measured and did not run, or not measured. Silence in the record reads as not measured. Call sites with no execution observation are reported as not measured. The value `measured_us` records the execution latency of one call to the site in microseconds, as a median across the recorded steps of the run. Nothing measures a site yet: no caller passes site measurements, so the engine leaves `stats` unset on each site during normal runs. Unmeasured fields stay unset, so readers distinguish unmeasured fields from zero. When an engine receives a plan file that already contains measurements, it starts the new record by clearing all run measurements so earlier actuals do not leak into the new record. When enabled through the `VXPU_RUN_RECORD` switch or the engine's `run_record_file` parameter, the record describes the last `generate` call and replaces the file atomically.

We unified the plan file and the run record under the `PlanFile` message structure. We rejected keeping parallel schemas because the plan and the record share the same call site hierarchy. We considered wrapping `PlanFile` in an outer envelope message, but dropped that idea. An envelope adds nesting without changing the site structure. A database plan tree does not wrap its nodes in a container just to report measurements. We also considered sidecar files keyed by site name, but dropped that idea too. Joining documents across files by site name is fragile. It requires verifying that both files describe the exact same run. Grouping execution measurements under `stats` sub-messages on `PlanFile` and `PlanSite` keeps what was chosen and what was measured in one place with no join.

The documents stay separate because their lifecycles differ: the table belongs to the card and outlives any plan; a plan belongs to a model and a query shape; a kernel record belongs to one lowering of one plan; a run record belongs to one run. All four are ours, versioned, and inputs to or outputs of a run; none holds user data.

The proto package `vxpu.plan.v1alpha1` (`vxpu/proto/plan/v1alpha1/plan.proto`) is the single source of truth for these documents. Go and Python share field names and types by construction. Documents that only programs exchange—the step graph description, the step plan, the physical plan file, the run record, the kernel record, and the kernel catalog—travel as binary protobuf because no person reads them. Documents and test fixtures intended for people default to JSON. The kernel table (`KernelTable`) is authored and tuned by engineers by hand to configure knobs and implementation choices, so it is always JSON. Test fixtures checked into the repository remain formatted JSON so reviewers can read changes in pull request diffs. File readers and writers decide between JSON and binary by file extension: files ending in `.json` are formatted as JSON; all other files are binary protobuf. Commands that take `--json` use it for standard output, and refuse a `--json` flag when writing to a file whose extension is not `.json`. Python readers and writers refuse paths ending in `.json`, so a plan written to a `.json` path cannot be given to the engine. Any plan document can be shown as stable JSON using `vxpu plan show --type TYPE --json FILE`.

### Verification checks

Before a speed measurement or serving model is accepted, the workload must demonstrate that it computes the correct tokens. A verification check has three parts:
- **Decided by whom:** The router decides whether to verify a model before serving it based on the model's registered facts, such as a golden reference document registered with the model or provided in an artifact. The golden file's shape is defined once as the protobuf message `GoldenFile` in `vxpu.plan.v1alpha1`. The file on disk stays JSON, read with protojson with unknown fields refused. The executor has no golden-specific logic and does not read reference files or compute their digests. If no golden document is known for a model, verification is skipped and the model transitions directly to passed.
- **Executed how:** The router drives verification through the production serving path: it sends each prompt from the reference document to the executor as an ordinary generation request using greedy sampling, collecting the streamed token IDs. Because it uses the exact same serving path as user requests, the check exercises the true end-to-end execution pipeline.
- **Judged where:** The router evaluates the verdict in Go by comparing the generated token IDs against the reference document. Exact token equality is required. If any prompt produces an unexpected token, the router fails the load, logging an error that names the prompt and the first differing position.

**Model verification standing.** The router serves a model only after its verification check has passed. A model has one standing: `unchecked`, `passed`, or `refused`. The router routes sessions only to models in `passed` standing, and rejects requests for an unchecked or refused model. A successful load moves the model to `passed`, whereas a token mismatch moves it to `refused`. Transport errors and context cancellations leave the standing unchanged so another caller can load the model. The router serializes concurrent loads of the same model so they do not overlap. Reloading a serving model whose executor remains alive is a no-op. If its executor is gone, the model returns to `unchecked` and can be loaded again. When an executor unloads or evicts a model, the router drops the model's placement and standing back to `unchecked`. When the router restarts, its in-memory model table starts empty, so all models start in `unchecked` standing until loaded and verified again. A refused model that is reloaded and passes its check moves to `passed`.

Future verification checks follow this same principle: standard serving requests drive the executor, and Go judges the verdict.

### Model loader and reconciliation

The router is moving model loading out of request paths into Kubernetes-style reconciliation. A model may run under several physical plans at once, one per kind of machine it is loaded on. The loop therefore reconciles an **instance** (a model on an executor, with the plan for that machine) rather than a single model. The queue key, the desired state (`InstanceDesired`), and the status (`InstanceStatus`) are per instance.

The reconciler manages all pod executors, whether loaded by artifact (`LoadModel`) or by name (`RegisterModel` followed by `NewSession`, `TokenizeChat`, or `GetVocab`). External executors keep their direct path (`loadExternalArtifact` and `modelStanding`) until their own move, where contention for a shared executor is designed as multiple instances contending for one resource.

**One status per pod instance.** All pod-executor loads read and write only the reconciler's `InstanceStatus`, which records the lifecycle phase (`not loaded`, `loading`, `checking`, `passed`, `refused`, or `failed`), last error, next attempt due time, and observed generation. Legacy `modelStanding` remains only for external executors during transition.

**One planning function.** The router's planner emits the plan for an instance when none is given. Until the planner emits a `PlanFile`, a load by name carries the registered switch hints (`RegisterModel --plan`) in `InstanceDesired.Request.Plan`, and the reconciler invokes the router's single planning function (`planLoad`) against the executor's hardware probe (`Describe`) before calling `LoadModel` on the executor. Once the planner emits a `PlanFile`, a load by name carries a `PlanFile` in desired state as a load by artifact does.

**One reconciler with a work queue.** The router uses a single reconciler backed by a Kubernetes work queue with rate-limited backoff (`workqueue.TypedDelayingInterface[string]`) rather than a dedicated goroutine and timers per instance. The queue keys are instance identifiers. Four sources trigger runs of the reconciliation loop:
1. Desired state changes: when desired state is registered, updated via `SetDesired`, or removed via `DeleteDesired`.
2. Failed step backoff: when a reconcile step fails, the queue requeues the key with exponential backoff.
3. In-progress polling: when background manifest weight caching or executor weight loading (`LOAD_STATE_LOADING`) is underway, the step returns without blocking and the queue requeues the key after a fixed short polling interval without counting against failure backoff.
4. Periodic resync: the reconciler runs a periodic background resync loop that checks idle keep-alive expiry for desired instances and re-enqueues active ones. If an executor unloads a model or its pod is deleted while desired state remains, the resync pass observes the actual state fresh and restores it.

The work queue guarantees at most one active reconciliation per instance at a time without per-instance locks or goroutine lifecycle management. Rate-limited requeuing handles backoff directly within the queue, and carrying the idle check on the periodic resync avoids per-instance timers.

**Instance status and states.** An instance has one status, defined once across the router. The status records the instance's current phase (`not loaded`, `loading`, `checking`, `passed`, `refused`, or `failed`), the last error encountered (if any), when the next attempt is due if the instance failed, and the observed generation:
- `not loaded`: Desired state is recorded, but loading has not started or the instance was removed.
- `loading`: The router is caching manifest weights, starting the executor runtime process through the agent, or loading model weights into the executor.
- `checking`: Model weights are loaded and the router is running golden verification prompts. If no golden reference exists for the instance, this step is skipped.
- `passed`: The instance passed verification (or skipped it) and is ready to serve.
- `refused`: The request itself is wrong or cannot be satisfied as it stands (`InvalidArgument`, `FailedPrecondition`, or golden token mismatch). A refused instance is terminal for its current desired state; the reconciler forgets the key and halts until the desired state changes.
- `failed`: An infrastructure, RPC, or process failure occurred (for example, the executor runtime process exited, the executor reported `LOAD_STATE_FAILED`, or an RPC returned `Internal`).

**Failed is retried; refused is not.** Serving an instance is stateless. A failure (such as a runtime process crash or executor `Internal` error) represents an infrastructure fault. The reconciler records `failed` with the error and next attempt timestamp so waiting callers receive the error immediately, and continues retrying with exponential backoff behind them until the environment converges. Transient `Unavailable` transport errors on a live runtime leave status unchanged while retrying. In contrast, a refusal (`InvalidArgument`, `FailedPrecondition`, or golden token mismatch) is deterministic; retrying without changing desired state cannot succeed. The reconciler halts retrying for refused instances until a caller registers a modified desired state.

**Ending an instance.** An instance owned by the reconciler ends when its desired state is removed, either through `DeleteDesired` or when its idle keep-alive expires.
- **Explicit removal:** `DeleteDesired` removes the desired state and enqueues the key for teardown. The reconcile loop closes pooled connections and deletes the executor pod (`vxpu-executor-<key>`). For keys the reconciler never held (such as models on external executors), `DeleteDesired` does nothing and never touches their pods.
- **Keep-alive expiry:** The reconciler starts an instance's idle clock when its current generation first reaches `passed`, not while it is waiting to schedule, loading, or checking. Serving requests (`NewSession`, `Chat`, `ExecuteChat`, `Generate`, `Execute`, `TokenizeChat`, and `GetVocab`) and matching `SetDesired` calls advance a per-instance atomic timestamp without acquiring the reconciler's mutex. On each periodic resync, the reconciler checks whether any instance that has passed and is not currently `not loaded`, `loading`, or `checking` has been idle for at least the keep-alive duration (`VXPU_EXECUTOR_KEEP_ALIVE`, default 3600 seconds). When an instance expires, the reconciler removes its desired state and enqueues it for teardown, deleting the pod once and leaving it deleted across later resyncs.
- **Observations never edit desired state:** Observations made by serving requests (a missing pod in `executorEndpoint`, or `LOAD_STATE_NONE`/`LOAD_STATE_FAILED` in `NewSession`) or by resync never remove desired state for reconciler-owned instances. They return an error and enqueue the instance so the loop restores it.

**Router shutdown and restart.** On shutdown (`Server.Close`), the router stops the reconcile loop, cancels in-flight reconciliation steps and background manifest rewrites, wakes any blocked `LoadModel` waiters with `Unavailable`, shuts down the work queue, waits for all loop goroutines to exit, and closes pooled connections. The router intentionally leaves executor pods and their running runtime processes up on shutdown, logging that it did so. Pods outlive the router so that a router restart or redeployment does not force a cold-start storm of pod rescheduling and multi-GB weight downloads.

When a restarted router comes up, its in-memory tables start empty and it leaves pre-existing pods untouched until desired state asks for them.
- **Artifact instances:** Callers re-issue `LoadModel` to establish desired state on the new router. If a pod from before the restart already exists and matches the planned placement and executor image, the reconciler adopts that pod and its running executor process without recreating either, sends an idempotent `LoadModel` to confirm the artifact and plan, runs golden verification if configured, and serves from it. If the pod's image or placement no longer matches (for example, after an executor image upgrade), the reconciler replaces the pod. Once adopted, the new router owns the instance and takes it down on keep-alive expiry or `DeleteDesired`. The new router leaves any unrequested pre-existing pod untouched, and that pod's executor evicts idle weights after its own `--keep-alive` window.
- **Named models:** Desired state for instances registered by name resides in persistent storage (modelstore). The restarted router resolves them from modelstore, observes existing executor pods fresh from Kubernetes, adopts running executors that are already serving, and reconciles any missing instances.

**The next moves in order.** Subsequent moves follow in order:
1. External executors: external executor discovery and multi-instance contention for shared external resources, retiring the remaining legacy `modelStanding` and `loadExternalArtifact` path.
2. Ranges: chunked pipeline parallelism and layer range loading adopt the reconciliation pattern.

**The planner's contract.**
- *The kernel table* is what the planner hands an
  engine: the GPU it targets, the kernels' knobs, and the implementation
  body to run for each call site. Without a table, the engine picks
  implementations using the registered preference order. A call site is
  the module path that calls the op in the exported graph, such as
  `model.layers.14.engram.wkv`. An entry may target a specific call site,
  an op at specific input shapes, or an op alone. The engine matches
  entries in that precedence order: site first, then shapes, then op
  alone. Every entry in the table must match at least one custom operator call site
  of the model being planned. If an entry matches no call site in the model's
  exported graph (by op, site module path, or shapes), `vxpu plan emit` refuses
  the table as an error, naming the unused entry. A table tailored for one model
  architecture or tensor shape fails rather than silently misapplying overrides
  to another model. An eager call carries no site, so it matches shapes and op
  entries only. A named body that cannot run on an eager call falls back
  to the registry's choice with a plan note. During graph lowering, if a
  named body cannot run, the engine raises an error instead of silently
  falling back. A site entry whose body is not lowerable is kept in the
  graph and then dispatched at run time without the site. The table also
  specifies typed kernel knobs (`knobs` and `knobs_by_rows`). When the
  engine loads a table, it applies the table's knobs to their home modules
  unless an environment variable or coordinator switch `VXPU_<NAME>` overrides
  them. The environment provides the operator's A/B override over table
  choices.
  Per-row knob choices belong to the plan (`knobs_by_rows`) rather than the
  environment, and an engine applies them per batch size at run time.
  The AOTInductor package cache key (`kernel_digest`) includes every
  in-segment knob that a dense package can see. Routed expert kernels and
  collectives run outside compiled packages, so the engine excludes their knob
  families from the package cache digest and applies them at run time. The
  package key includes `kernel_digest` so that changes to in-segment knobs
  produce a different key (#490).
- *The kernel catalog* (`KernelCatalog`) reports the kernel bodies available in
  the build once, with their device targets, execution kinds, and applicability
  conditions stated as data. Most conditions are bounds on row counts or data
  types. The Triton sinkhorn kernel's power-of-two width condition is reported
  as a boolean fact `sinkhorn_supported` on the call site rather than embedding
  Triton kernel math into the condition schema, and the body's condition specifies
  `require_sinkhorn_supported`.
- *The kernel record* (`KernelRecord`) records every call site the graph lowering
  saw, the knobs in effect (`effective_knobs()`), execution device, GPU compute
  capability when on CUDA, and per-site entries (`SiteRecord`). For each site, it
  records the op, module path, fx node name, input shapes, layer index, chosen
  body, origin source (`table`, `registry`, or `kept`), inlined kernels, and
  tensor byte footprints (`weights`, `activations`, and `state`). The byte counts
  determine the site's floor in the roofline ledger, counting weights plus
  activations over device bandwidth.
  Implementation rules:
  - A kernel's selection predicate reads its knob from the module at call time,
    not at import.
  - The engine applies `knobs_for_rows` per batch size in the per-rank decode
    path.

**How call sites relate to plan sites.**
A plan file's sites (`PlanSite`) and a kernel record's call sites (`SiteRecord`) describe the model's operations:
- For the dense kind, the two are nearly the same thing. A plan site's id is the call site's module path plus `.weight` (such as `model.layers.0.attn.wq_a.weight` for the call site at `model.layers.0.attn.wq_a`). Of the 12 dense plan sites in `golden_plan.json`, 8 match an `fp8_linear` call-site path. The other 4 (`wo_a` and `ffn.gate` across two layers) have no `fp8_linear` call of their own: `wo_a` is invoked as a grouped linear (`fp8_grouped_linear`) rather than `fp8_linear`, and `ffn.gate` is the MoE router gating weight invoked via standard PyTorch linear rather than a custom op.
- In the tiny graph, each `fp8_linear` path is called once, but operations without dedicated weights repeat at the same module path. For example, at `model.layers.0`, `hc_sinkhorn` is called twice, and inside `model.layers.0.attn`, `rotary` is called 3 times.
- Because operations repeat at the same module path, `(op, site)` is not unique. The graph node identifier `node` (such as `fp8_linear_7` or `rotary_1`) makes each call site uniquely identifiable within an exported graph. Node names are unique within one export only and change when the model is exported again, so a plan keyed by node name is tied to that specific export. A reader checks that the plan belongs to the artifact via `artifact_md5` on `PlanFile`.
- When both `strategy` and `body` are present, they govern different decisions in the plan. `strategy` selects the macroscopic architectural policy for an operator family (for example, whether `moe_experts` runs `fused` as one C++ call per layer or `per-expert` where each expert decodes individually). `body` selects the exact physical kernel implementation body to execute at a single call site (for example, `gpu-fp8-gemv` or `reference` for an `fp8_linear` call). The plan names the chosen body in one place per site on `PlanSite.body`. For call sites where no strategy was chosen, `strategy` is left unset so that a reader does not take it for a decision.
- The plan file's `sites` list carries an entry for every call site of custom operations in the graph, keyed by graph node name, with each site's chosen kernel body. When `vxpu plan emit` reads an artifact's exported graph, it plans every call site read from the graph directly using the kernel catalog and table, matching graph call sites one for one by node name. Unmatched sites on either side are refused by name. Ops recognized by the reader that have bodies registered in the catalog receive a chosen body matching the target device and conditions. For custom ops where the planner does not choose a body (no candidates registered in the catalog), the planner emits a plan site with body unset, indicating the op executes its default runtime implementation without substitution. For a model with none of our custom operations, `vxpu plan emit` produces a pass-through plan where no call sites have substitutions (all bodies unset); every call site executes the engine's default runtime implementation (plain PyTorch).
- Each condition currently exists in two places: the `applies` lambda in Python (which checks runtime environment flags) and the `condition` data in the catalog. Until the lowering executes plan choices and decides nothing, both copies exist so that the lowering continues to make its own choices and the golden test proves agreement with the planner.
- Some conditions are not visible from the graph (whether an n-gram table is file-backed, and whether a layer has mailboxes or hot experts). Where candidate bodies depend on facts not yet available to the planner, the catalog records `require_external_fact` and the planner never chooses such a body on its own, only when a table names it.
- Note on test fixtures: `golden_plan.json` (2 layers) comes from the hand-crafted illustrative test fixture in `pkg/kernelplan/testdata/manifest.json`, while `golden_call_sites_plan.json` (4 layers, 67 call sites) is the golden plan for `tiny_deepseek_v41`.

### Compiled package planning and cache keys

The planner names and tracks compiled packages centrally before engines load them. For an artifact, a plan, and a machine kind, the planner computes which packages an engine needs, determines the content key for each package, and queries the store to report which packages already exist, which are missing, and which are known to fail (`kernelplan.CheckPackages`). When a compile task reports a failure, the package store records it under the model's failures repository (`vxpu-failures/<model>`). When the planner checks packages for the same plan, machine kind, and software version, packages recorded as failed appear under `failed` with their recorded diagnostic reason, and the planner omits them from `missing` so it does not ask for them again. A different software version or a different machine kind continues to ask for the package. A failure record can be cleared by name for a deliberate retry (`vxpu plan packages --clear NAME`), and a later successful build of the package replaces the failure record.

**What determines a package.**
The planner names a package from its own inputs:
1. The artifact: the checksum over `manifest.json` and `binding.json` bytes (`artifact_md5`).
2. Which segment: the segment identifier within the model (such as `submod_0`).
3. The chosen kernel implementation bodies and compiled-in tuning knobs from the plan. The planner reads the model's segments and call sites directly from the artifact's exported graph (`decode.pt2`). Host-op segments require no package compilation. For each compiled segment, its package name is determined by the kernel implementation bodies chosen for the call sites located in that specific segment, taken in graph execution order (joined by the graph node name). A call site whose plan site specifies an empty body is allowed and is named with an empty body string at that call site position. Changing a chosen body renames only the package of the segment containing that call site; all other segment packages remain unchanged. Each knob's definition in `plan.proto` states whether the planner compiles it into a package (`compiled_in = true`) or whether only code outside every compiled package uses the knob (`compiled_in = false`). Knobs used only outside compiled packages leave package names unchanged. The planner reads `compiled_in` from the knob definition rather than keeping a separate list in code, and refuses any knob that lacks the declaration.
4. The machine kind: identified by hardware facts from the machine database or executor probe.
5. The executor software version: the image digest or git tree commit it will run. A new executor version renames every package, so each release rebuilds once.

**What a machine kind is for this purpose.**
A machine kind identifies the compilation target using only facts present in the machine database or executor probe: the GPU model name for accelerators, and the host processor model and SIMD feature flags for CPU hosts. Probes do not yet report GPU compute capability or CUDA version, so the planner does not guess them.

The machine kind reads measured and cataloged facts. It does not assert unprobed hardware properties in code or assume cards share architectures when their names differ.

**How planner package keys relate to engine cache keys.**
The planner computes the package key in one place in Go to name packages stored in modelstore. When an engine receives planned packages, it uses the planner's key for each named segment; if a named package is absent from the cache, the engine builds it under that key, and unnamed segments fall back to the engine's internal cache key. This fallback ends once the engine is told all its packages.

### Segment boundaries and graph partitioning

The planner defines which operations cut a segment in Go in `vxpu/pkg/kernelplan` (`HostOps`). The planner owns segment boundaries because dividing a model into executable segments is a structural planning decision, rather than an executor implementation detail. PyTorch is an execution backend that the system may replace. The planner reads the model's exported graph, determines the segment partitions and call sites, and instructs the executor which packages to compile and run. During the transition until the Python graph split is removed (#748, move 4), Python keeps `HOST_OPS` in `segments.py`, verified by tests to stay identical with Go's `HostOps`.

### Planning goals

A caller states a goal (`PlanGoal`) on a plan request (`vxpu plan emit --goal`, `LoadModelRequest.goal`, or `RegisterModelRequest.goal`). The single planning function (`kernelplan.PlanArtifact`), used by both the `vxpu` command and the router, decides how to meet that goal for the target machine. The goal states what the caller wants, never which runtime or kernel body to pick. The emitted `PlanFile` records the goal (`PlanFile.goal`), and the tree page displays it. Registering a model with a non-default goal on `RegisterModel` is refused as not supported yet until the router plans loads by name; pass the goal on `LoadModel` instead.

- **`pass-through` (`PLAN_GOAL_PASS_THROUGH`, default):** The caller asks the planner to run the model faithfully with minimal changes. For a model with no custom vxpu operations (such as tiny Llama), every segment runs as plain PyTorch (`SEGMENT_RUN_PLAIN_TORCH`) and requires no compiled packages. For a model the planner has kernels for (such as tiny DeepSeek), the default goal gives the planner's standard plan: every non-host segment runs from an ahead-of-time compiled package (`SEGMENT_RUN_CAPTURED` on a GPU machine, `SEGMENT_RUN_PACKAGE` on a CPU machine) with the catalog or table's chosen bodies.
- **`speed` (`PLAN_GOAL_SPEED`):** The caller asks the planner to make the model run as fast as possible on the target machine. Every non-host segment of any model runs from an ahead-of-time compiled package, captured into a CUDA graph (`SEGMENT_RUN_CAPTURED`) when the target machine has a GPU and called directly (`SEGMENT_RUN_PACKAGE`) on a CPU machine. Host segments always run as plain PyTorch. For a model the planner has kernels for, `speed` makes the same segment runtime choices as the default goal today and records `PLAN_GOAL_SPEED` on the plan; measurement-driven selection builds on this goal later.

### How each segment runs

The plan specifies how each segment runs on the target machine through `SegmentRun` (`Segment.run`):
- **`plain torch` (`SEGMENT_RUN_PLAIN_TORCH`):** The segment executes directly in PyTorch without ahead-of-time compilation. The executor invokes the segment callable directly as plain torch. The planner sets this for every segment of a model with no custom operations under the `pass-through` goal (such as tiny Llama) and for eager host-op segments.
- **`jit` (`SEGMENT_RUN_JIT`):** The segment compiles just-in-time at load using `torch.compile`. The engine on main defaults to this runtime when no plan is provided or when JIT compilation is opted into. Plan emit does not choose this value today.
- **`package` (`SEGMENT_RUN_PACKAGE`):** The segment executes from an ahead-of-time compiled package called directly without CUDA graph capture. Dense segments run with this value when the plan targets a CPU machine under `speed`, or under the default goal for a model with custom vxpu operations.
- **`captured` (`SEGMENT_RUN_CAPTURED`):** The segment executes from an ahead-of-time compiled package captured into a per-segment CUDA graph. Dense segments run with this value when the plan targets a GPU machine (such as NVIDIA L4) under `speed`, or under the default goal for a model with custom vxpu operations.

The planner decides each segment's run mode for the machine and goal the plan targets; the executor builds each segment's callable from that value and nothing else. Today the executor reads switches in Python instead (`VXPU_SEGMENT_RUNTIME`, `VXPU_SEGMENT_GRAPH`, and `VXPU_SPLIT_COMPILE` in `vxpu/python/vxpu/plan.py`), which a later move replaces once the plan travels on load.

From this one fact follow:
1. **Required packages:** `RequiredPackages` includes exactly the segments whose plan value says package (either `SEGMENT_RUN_PACKAGE` or `SEGMENT_RUN_CAPTURED`). Plain PyTorch and JIT compilation require no compiled packages. A segment with an unspecified run mode is refused. Host segments cannot run from a package; a host segment marked as a package is refused.
2. **CUDA graph capture:** `HasCapture` reports whether any segment runs captured into a CUDA graph (`SEGMENT_RUN_CAPTURED`).
3. **Pass-through plans:** A plan is pass-through when every segment runs as plain torch and no call site names a body (`kernelplan.IsPassThrough`). The tree page shows each segment's value as a badge, uses the pass-through definition to display its note ("Pass-through plan: Every call site runs the engine's own implementation"), and omits the unmatched plan sites section when pass-through. A plan without segments is not interpreted.

### Segment values: reads and writes

Each segment in the plan declares the values it reads and writes by the names the exported graph gives them. An executor uses these lists to lay out value slots and execute segments in order without analysing the graph.

Values are distinguished by lifecycle:
- **Step inputs** (such as `input_ids` and `cache_position`) are supplied anew for each decode step and do not outlive the step.
- **Resident values** (model weights, KV caches, scratch buffers, and constants) persist across decode steps. Every resident value, whether read or written, names the persistent slot in `resident` and the segment-local tensor in `name`. A later segment reading the output of a buffer mutation carries the same `resident` slot and observes the new state; a later segment reading the pre-step placeholder observes the start-of-step state. At the end of the step, the executor commits the mutated output into `resident`.
- **Intermediate values** pass between segments within a single step. Produced by an earlier segment and consumed by a later segment, they do not outlive the step.
- **Step outputs** (such as logits) are produced by the final segment and returned to the caller.

Every value a segment reads is a step input, a resident value, or an intermediate value written by an earlier segment. The final segment writes the step's outputs. Both `vxpu plan segments` and `vxpu plan emit` extract these reads and writes from the exported graph into the plan proto, and the tree page and its JSON representation display each segment's reads and writes.

### Model structure tree visualization

The management interface draws a model's structure offline as a tree (`vxpu plan tree --artifact DIR [--plan FILE] [--record FILE]`). The generated self-contained HTML page displays what the Go planner read from the exported graph in the artifact: the model at the root, its segments in execution order beneath it, and each segment's custom operator call sites showing their op and site identifier (graph node name). Host-op segments are visibly differentiated from runs of ordinary compiled nodes, and segments can be collapsed to keep large models navigable.

When provided with an execution plan (`--plan FILE`), the page draws the plan on the tree (`EXPLAIN`). Each call site displays the implementation body chosen by the planner. Any call site not covered by the plan is explicitly identified rather than left silently blank. Plan sites that match no call site in the graph appear in a collapsible unmatched plan sites section with a count in the header, rather than being silently dropped. The plan's configuration knobs are displayed once for the plan, noting whether each is compiled into packages or executes outside compiled packages.

When provided with a run record (`--record FILE`), the page displays execution actuals beside the planned choices (`EXPLAIN ANALYZE`). Each call site displays its measured time with its unit (`µs`), each segment displays its total measured time, and a slowest call sites section highlights the top time consumers with relative percentages. An interactive sort control allows reordering call sites within segments by measured time. Unmatched measurements appear with their count rather than being dropped. Overall execution figures match the recorded run statistics. Every number on the page states its unit and that it was measured.

The command validates that the plan's artifact checksum matches the artifact directory's manifest and binding checksum, refusing mismatched plans before rendering. The same joined tree and plan data is available as JSON (`--json`), reusing the existing plan documents via protojson.

### Step plan graph visualization

The management interface draws a step plan offline as a graph (`vxpu plan graph --plan FILE [--graph FILE]`). The generated self-contained HTML page displays the decode step nodes and edges placed at the depth their waits give them in a layered grid layout, aligning streams horizontally and flowing downward in execution order. Nodes that can run concurrently sit side by side in the same row. Every wait dependency declared on a node appears as a directed edge. Edges that skip rows and edges between streams route through dedicated gutters outside and between stream columns so that no edge segment passes behind a node. Go computes the layout and edge routing into `StepPlanGraph`, providing bounding boxes for nodes sized to fit their contents without clipping, waypoints and SVG path strings for edges, canvas dimensions, and an edge crossing count. The layout guarantees that no edge passes behind a node and distinct lanes do not share a gutter track; edges entering or exiting the same node share stem endpoints. The edge crossing count tallies perpendicular crossings between different routed edges, without counting collinear overlap along shared stems. Hovering or selecting a node highlights its incoming and outgoing dependency edges; when script is disabled, every routed edge remains fully drawn and visible on the page. When provided with an optional step graph description (`--graph FILE`), the page displays each node's layer kind, execution mode, and answering handler, and marks wait dependencies connecting to peer GPU or remote handlers as card crossings. Without an attached graph description, the step plan records stream assignments and wait dependencies, but does not identify exchange handlers or cross-card answers, and card crossings are reported as not known rather than zero.

When provided with a step plan carrying run measurements (`EXPLAIN ANALYZE`), each node displays its measured execution time with its unit (`ms`) and states that it was measured; without measurements (`EXPLAIN`), no node displays zero. Wait dependencies that reference nodes not present in the step plan appear in an unmatched waits section with their count in the header, rather than being dropped. The page opens without network access and references no external resources. The identical graph structure, layers, streams, edges, card crossings, and measurements are available as JSON (`--json`), rendered from the `StepPlanGraph` protobuf message via protojson.

### Standalone package compilation

Compiling packages is a dedicated task apart from serving (`python -m vxpu.compile_package`). It is not an engine: it loads no weights it does not need to compile, allocates no serving state, and takes no requests.

**What the task is told.**
The task is given an artifact, a required plan, a target output directory, and the packages the planner names. Each named package provides its segment name and the key the planner gave it. The task is told what to build; it does not work out which packages are missing or compute keys of its own.

**What the task does and reports.**
The task compiles exactly the named packages and leaves each under the key the planner gave it (`<key>.pt2`). It refuses any package it was not given enough to build, such as an unknown segment, a host op, or a missing key. When a package cannot be compiled, the task reports the failure as a structured result naming the package, machine kind, software version, and compiler diagnostic reason; it does not rewrite anything to get past the failure, and it still builds the packages that can be built. The driving Go command (`vxpu plan packages`) records reported failures in the store, prints them to stderr, and exits non-zero when a required package could not be built. When asked to build (`--build --agent HOST:PORT`), `vxpu plan packages` compiles missing packages through the compile service on the named agent. The compile process writes built packages directly to the package store from where it built them. The command re-checks the store and reports them present. Subsequent planning requests read the failure from the store and stop asking to compile it again on that machine kind. An engine pointed at a directory containing the pre-compiled packages loads all of them and builds none.

When compiling packages, the system records the observed outcomes of building operator bodies on machine kinds:
- **Where the record lives:** Body compile outcomes are facts about an operator's body on a machine kind ("the outcome of building body B of op O on machine kind M with software version V") and hold across all models and packages. The record lives in the package store under the shared repository `vxpu-body-outcomes`, independent of any specific model.
- **Identity and compiler version:** The record's identity is `(op, body, machine_kind, software_version)`. The executor software version identifies the compiler toolchain release (such as PyTorch and C++ compiler revisions). When the software version changes, records under the previous version do not apply to the new release, giving compiler bugfixes a clean opportunity to build the body without manual intervention.
- **Outcomes replace previous observations:** The record holds the last observed outcome: compiled successfully, or failed with the compiler's diagnostic message. A build that fails writes the failure with the compiler's message; a later build that succeeds writes success under the same identity, which replaces it. Nobody edits the record by hand. A wrong record gets corrected by observing again (building that body again and replacing the outcome).
- **Viewing:** `vxpu plan packages` displays observed body compilation outcomes for a machine kind under `body_outcomes`. An engine pointed at a directory containing the pre-compiled packages loads all of them and builds none.
- **Planning around outcomes:** When `vxpu plan emit` runs with a package store and a target machine, the planner reads the recorded body compile outcomes for that machine kind once and filters by the plan's software version. The planner treats "recorded as compiling" and "not known" alike. For each call site whose preferred body is recorded as not compiling, the planner selects the next applicable body offered by the kernel catalog and records the reason on `PlanSite.body_reason` and `CallSite.body_reason`, which appears as a reason badge on the tree page. If no alternative body compiles, the planner refuses, naming the call site, the body, and the failure reason. When no body is recorded as not compiling, the emitted plan is byte-identical to the baseline plan.

The executor also serves package compilation over gRPC (`service Compiler` in `proto/vxpu.proto`). The Go client (`pkg/kernelplan.CompilerClient`) connects to this service to build packages: it asks the agent for a serving compile process (`compiler`), finds the endpoint in the agent's process table, verifies readiness over the connection using the standard gRPC health service bounded by the caller's context, sends `Compile` RPCs, deduplicates concurrent requests for the same compilation inputs with shared verdicts, and stops the process when work completes or when the caller fails or is cancelled. The caller and the compile service share only the store and the agent. `CompileRequest` names the artifact by its reference in the store (`repository:tag`) and carries the plan as a message, sending no paths on the caller's disk. When the agent starts the compiler process, it stages the artifact's layer blobs via `StageBlobs`, links them into the process's artifact directory, and passes `--artifact` and `--artifact-ref` to the server. A staging failure fails the request and names the artifact, and the server refuses any request whose artifact reference does not match the staged artifact. If a process exits before serving, the client reports the agent's recorded exit status and output tail immediately rather than timing out. The server runs the same compilation pipeline as the CLI command and serializes compilation builds with a lock so concurrent requests do not race on process-global compiler settings and kernel knobs. The reply carries `PackageReport` with present packages and structured compile failures rather than RPC errors. A single connection serves multiple compilation requests. Mutual TLS will authenticate coordinator and executor connections under #662; until then, the compiler listens on the assigned port without credentials.

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

The router holds one connection to each executor it talks to, reuses it for every call, and closes it when the executor is removed. It retries calls that are safe to repeat (such as TokenizeChat, GetVocab, and NewSession) when an executor restarts. A stream already in flight when its executor dies ends with an error, because that cannot be repeated.

Kubernetes keeps the bursty, CPU-only work: compile Jobs, modelstore, the
router. A machine that runs an executor is dedicated to it.

**Direction.**
- Executors and models become separate things: a pool of executors (VMs,
  pods), each holding several models, with placement in between, and each
  executor reporting per-model state (#385).
- An agent on each worker VM lets the engine manage machines directly:
  stage weights ahead of a load, serve them peer to peer, run the executor
  (#384).
- Acquiring capacity: a multi-zone flex-start racer (`pkg/capacity`,
  #483). A flex-start request for scarce GPU VMs in nine
  zones is granted much sooner than a request in one zone. The racer
  queues one request per zone with a deterministic name and polls their
  statuses. The first zone to reach RUNNING wins. The racer immediately
  cancels all other requests in parallel and retries until each is
  confirmed gone, preventing charges for unused VMs. Requests that expire
  are recreated, and zones without the requested machine type are dropped,
  without stopping the race.

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

**Agent supervision and process execution (#657).** The agent (`vxpu-agent`,
`pkg/agent`, `proto/agent.proto`) runs on every machine that runs executors,
both on bare VMs and inside executor pods. When the coordinator tells it to,
the agent mounts weights disks, and starts, stops, and reports the processes on
its machine.

In an executor pod, the agent is process 1. As process 1, the agent reaps
orphaned children that reparent to it, and passes termination signals on to the
runtime so it stops within the pod's grace period. The router asks the pod's agent
to start the runtime process, waits for it to serve, and dials the runtime for
inference execution. The agent in a pod accepts any caller on the pod network.

The agent only runs a small set of known-safe tasks (such as `executor`,
`nodeserver`, and `compiler`), and does not run arbitrary commands. The
coordinator selects a task by name and supplies typed settings (such as a card
index). When starting a task configured to serve (such as `compiler`), the
agent allocates an available port, passes `--port <port>` to the child process,
and publishes the reachable endpoint (`host:port`) in the process table
(`ProcessStatus.endpoint`). Callers connect to the published endpoint directly
without agent forwarding. Readiness is established by the caller over the
network connection; the agent does not probe child sockets. The endpoint
remains in the process table while the process runs, and the agent clears it
when the process exits. The agent tracks each process: when a process exits or
is killed, the agent reports its exit status and recent output without the
caller polling a file.

**Machine discovery and verification (#663).** The coordinator queries the
agent (`DescribeMachine`) to learn the machine's hardware facts before
assigning work. In a pod, this answer describes the pod's share rather than the
entire host node.

**Blob staging onto local disk.** The coordinator asks the agent to make blobs
present on local disk (`StageBlobs`), each identified by its sha256 content
hash. The agent checks what is already on disk, fetches missing blobs from its
configured content-addressed store (modelstore), verifies that their bytes
hash to their names, and places them into the local blob directory
(`<work-root>/blobs`). A second call for the same blobs transfers nothing.

A single stage call transfers up to eight missing blobs concurrently, each as
one whole transfer without range splitting. Eight is a chosen bound that runs
several transfers at once while keeping stream buffer memory and store
connections bounded. When a blob transfer fails, the agent reports which blob
failed and stops the remaining transfers for that call. Any unfinished temporary
files are removed, while blobs that already finished stay on disk.

Where the agent fetches from is configuration the agent starts with, never
part of the coordinator's call. Staging takes content-addressed names, not
URLs: a root process that fetches whatever URL the network hands it is the
same hazard as running whatever command it is handed. The content address is
the integrity check: an interrupted transfer or mismatched blob is never left
under that name. The coordinator can query staged blobs and their sizes
without transferring anything (`ListBlobs`).

Staged blobs live in the layout used by the executor runtime's local weight
tier (`VXPU_BLOB_DIR`): files named directly by their sha256 under
`<work-root>/blobs`. The machine's deployment (such as a VM bring-up script
or a pod's container volume mounts) sets both the agent's work root and the
runtime's `VXPU_BLOB_DIR` to `<work-root>/blobs`. When the agent checks
whether a blob is already present, it trusts any regular file at
`<work-root>/blobs/<sha256>` with that name without re-hashing its contents
from disk on every call. The runtime places files there checked by size, and
once placed, the agent treats the file as the complete blob.

The agent only reports facts; decisions belong to the coordinator.

**Agent authentication and mutual TLS (#662).** Every control-plane and
serving connection between the coordinator and an agent or runtime will require
mutual TLS with SPIFFE identities. Using SPIFFE URI Subject Alternative Names
with no IP or DNS SANs lets the system replace its initial issuer with SPIRE or
another workload identity issuer later without changing verification code.

The `vxpu-pki-init` Job (`cmd/vxpu-pki-init`, `pkg/pki`, `k8s/pki-init.yaml`)
bootstraps this trust root inside a cluster:
- It runs in a dedicated `vxpu-pki` namespace and stores the ECDSA P-256 CA
  certificate and private key in Secret `vxpu-ca`. Isolating `vxpu-ca` in
  `vxpu-pki` prevents the router from mounting the CA key onto pods in the
  workload namespace. Neither the router nor executor service accounts hold
  Kubernetes RBAC to read Secrets in `vxpu-pki`.
- In the workload namespace, the Job writes ConfigMap `vxpu-ca-bundle`
  containing only `ca.crt`, Secret `vxpu-coordinator-tls` carrying
  `spiffe://<trust-domain>/ns/<ns>/sa/vxpu-router` with `ClientAuth` extended
  key usage, and Secret `vxpu-agent-tls` carrying
  `spiffe://<trust-domain>/ns/<ns>/sa/vxpu-agent` with `ServerAuth` extended
  key usage.
- The Job is idempotent and leaves a consistent certificate set. Re-running it
  keeps valid objects unchanged, updates `vxpu-ca-bundle` if the CA changed,
  and re-mints any missing, expired, or non-verifying leaf Secret from
  `vxpu-ca`. The Job logs every certificate's identity, expiry timestamp, and
  whether it was created, kept, or re-minted (with the reason), and never logs
  private key material.

Known limits and work not done yet:
- Writing the CA and workload certificates to a local directory for a single
  machine without a cluster will follow in a separate change.
- One shared leaf certificate (`vxpu-agent-tls`) serves every pod agent until
  per-pod identities replace it, so one compromised executor pod can
  impersonate another pod agent.
- Certificate renewal currently deletes the leaf Secret (or lets it expire),
  re-runs `vxpu-pki-init`, and restarts affected pods.
- Nothing uses these certificates yet. Agent, router, runtime, and VM
  connections remain unauthenticated until subsequent moves wire mTLS into
  `vxpu-agent`, `vxpu-router`, the Python serving children, and VM bring-up.

## 3. Where weights live

Weights are content-addressed by sha256 at every tier:

| tier | what | code |
|---|---|---|
| bucket | modelstore's blobs (`gs://…/_blobs/<sha256>`) | modelstore |
| weights disk | a checkpoint's blobs on a GCE disk, attached read-only | `pkg/weightdisk`, `pkg/agent` |
| executor's disk | the local weight tier, filled from the bucket | `localblobs.py` |
| checkpoint files (in place) | expert weights and n-gram tables | `engine.py` |
| local SSD (`external_dir`) | n-gram tables when step graph native packages require resident tensors | `engine.py` (`external_dir`) |
| GPU | dense weights; hot experts (§4) | `engine.py`, `experts_gpu.py` |

The engine addresses expert weights in place straight from the checkpoint
files, with no assembled copy on disk or in RAM (#488). Shard files are
memory-mapped read-only (`MAP_PRIVATE`), and each layer builds an
`ExpertTable` holding narrow views of its projections (`w1`, `w2`, `w3`) and
block scales alongside 1D pointer tensors of their data addresses. When the
engine places hot experts on a GPU, it copies each expert straight from its
shard memory map in parallel across experts, with no intermediate tensor.
The CPU kernels take the pointer tensors directly into the shard memory map.
Prefill routing and the expert mover read from the same table.

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

`vxpu disk build`, `attach` and `detach` run their GCE operations
through the Compute Engine API (`cloud.google.com/go/compute/apiv1`).

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

**The un-profiled timeline: in-graph stamps and the account per class.**
Wait and launch claims come only from in-graph stamps or host events with the
profiler turned off. Profiling with CUPTI serializes kernel launches on the
device, creating an artificial launch stagger that does not reflect real
execution.

Under `VXPU_PHASE_TRACE=1`, the step graph stamps the GPU's `%globaltimer`
onto each node's stream at the node's start and end using lightweight Triton
kernels (`phase_stamp`). Each stamp costs about 2 µs, adding roughly 4 µs per
node. An atomic addition (`phase_bump`) advances the device step counter at
the end of each captured replay, cycling through eight steps round-robin
(`PHASE_STEPS = 8`).

The executor reads these stamps after a run through `StepGraph.phase_table`,
which decodes the buffer into rows ordered from oldest step to newest. The
pure function `phase_account` summarizes each step's rows into the account's
columns:
- **wall**: the elapsed duration from the earliest node start to the latest
  node end.
- **covered**: the union of node execution intervals, collapsing concurrent
  spans.
- **gaps**: the uncovered time, computed as wall minus covered.
- **per-class sums**: the total execution span for each class. A graph node's
  class is the vxpu op its segment calls, and an exchange node's class is its
  layer kind label. If a graph segment calls multiple vxpu ops, each op class
  receives the segment's duration. When an exchange node encloses a graph island
  running on a concurrent stream, the account reports the full span of the
  exchange node and the full span of the graph island inside it under their
  respective classes.

**The step graph description document.**
`dataflow.Graph` (`vxpu/python/vxpu/dataflow.py`) already holds the step: a
list of nodes in call order. A node is one segment of the model, carrying a
layer kind label (`dense`, `experts` or `engram`) for humans and timings. A node
executes in one of two ways:
- **graph**: the node executes on the GPU inside the captured graph as a compiled segment.
- **exchange**: the node executes through an exchange with something outside the graph, answered through a mailbox by the CPU, a peer GPU, or a GPU pool.

Each argument of a node is one of three things: a step input, a resident tensor
(weights or state that live across steps, such as a cache), or an edge (an output
of an earlier node). Some nodes write a resident in place, for example a node that
appends to a cache. Two nodes may not run at the same time when they touch the same
resident and at least one of them writes it. That rule needs exactly these facts,
and nothing else.

When `VXPU_STEP_GRAPH_FILE=<path>` is set (requiring `VXPU_DATAFLOW=1`), the
engine writes this description as JSON atomically when it builds its dataflow
graph.

Seven protobuf messages and three enums in `vxpu.plan.v1alpha1` (`proto/plan/v1alpha1/stepgraph.proto`)
record the step and its plan:
- **`StepGraphDescription`**: the step graph as facts: its nodes in call order,
  step input names, output value references, and host mappings available outside the graph.
- **`StepNode`**: one segment node with its name, execution mode (`EXECUTION_GRAPH`
  or `EXECUTION_EXCHANGE`), layer kind label, decoder layers covered, flattened
  argument reads (`ValueRef`, omitting constant arguments), in-place resident
  writes, host op target if answered through an exchange with the CPU, and who
  answers it (`answer`).
- **`HostMapping`**: a host tensor mapping outside the graph, with its name and
  whether it is backed by a file mapping on the host (`file_backed`).
- **`Execution`**: how a node executes: inside the graph on the GPU (`EXECUTION_GRAPH`),
  or through an exchange outside it (`EXECUTION_EXCHANGE`).
- **`Answer`**: who answers an exchange node: the CPU (`ANSWER_CPU`), a GPU pool
  on this GPU (`ANSWER_GPU_POOL`), a peer GPU (`ANSWER_PEER_GPU`), or a remote
  node server (`ANSWER_REMOTE`).
- **`HostCall`**: a host operation answered by the CPU, with its collection
  name and index.
- **`EdgeRef`**: an output edge from an earlier node, with the producer node
  name and an optional output index.
- **`ValueRef`**: a reference to a step input, a resident tensor, an output
  edge from an earlier node, or a host mapping.
- **`StepPlan`**: the ordered execution plan for the step graph, containing
  `NodePlan` entries in call order, the names of file-backed host mappings to
  release (`release_host_mappings`), and execution measurements for released host
  mappings (`stats`).
- **`HostMappingReleaseStats`**: resident bytes before and after releasing a
  host mapping (`bytes_before` and `bytes_after`).
- **`StepPlanExecutionStats`**: execution measurements recorded when the step plan
  is run, holding `HostMappingReleaseStats` entries in `host_mappings`.
- **`NodePlan`**: one segment node's stream assignment (`STREAM_MAIN` or
  `STREAM_PARALLEL`), sorted wait dependencies, and optional execution measurements
  (`NodeExecutionStats`).
- **`NodeExecutionStats`**: measurements recorded from a real-world run of a step node,
  including median device execution duration in milliseconds (`device_time_ms`)
  and sample count (`sample_count`).
- **`Stream`**: execution stream assigned to a node (`STREAM_MAIN` or `STREAM_PARALLEL`).

**Host mappings and exchange handlers.**
Nodes answered outside the graph interact with tensors that sit outside the
captured graph (such as expert banks or n-gram tables). The description reports
these as `HostMapping` entries with a boolean indicating whether each is backed
by a file mapping on the host (`file_backed`). Exchange nodes reference their
external tables in their argument reads as `ValueRef.host_mapping`.

Who answers each exchange node is reported directly on `StepNode.answer`:
`ANSWER_CPU` when answered by the CPU (including partial placements where cold
items run on the CPU), `ANSWER_GPU_POOL` when wholly handled by a pool on this
GPU, `ANSWER_PEER_GPU` when wholly handled by peer GPUs on the same machine, and
`ANSWER_REMOTE` when served by a remote node server. Graph nodes leave `answer`
unspecified. These facts are reported directly by `External` when pools and
remote nodes are attached, without altering node placement or execution order.

**Releasing host mappings.**
The planner decides which host mappings to release from the step graph description
alone. It writes their names to `release_host_mappings` in the step plan. A mapping
is released when it is file-backed, read by at least one node, and read by no node
the CPU answers. A mapping with no known reader remains in host memory. A mapping
read by any node the CPU answers, or by a node whose answerer is not stated, remains
in host memory. This rule does not name layer kinds.

The engine releases exactly the host mappings the plan names. It resolves each
mapping name to its host tensors, finds their page ranges in memory, and advises
the kernel with `MADV_PAGEOUT` to drop their clean pages. Other host mappings
remain intact. Because releasing is kernel advice on an active mapping, the
process keeps the memory addresses valid. When the release completes, the engine
reports the resident bytes before and after for each named mapping and records
them in `StepPlan.stats` on the step plan it writes back.

The document holds facts and the planner derives the order from them. The description
carries execution as an explicit enum (`Execution execution = 7`) alongside `host`.
We chose separate fields with strict validation over a `oneof`: an explicit enum makes
how a node executes directly visible on every node in JSON and queryable in code
(`node.Execution == ExecutionGraph`) without unwrapping `oneof` structures. The planner
validates that a graph node cannot define a host call and an exchange node must define
one, so contradictory states are refused.

As a worked example, the 11 segments of a 4-layer MoE model produce nodes that
alternate between graph computation and exchange ops:
```
submod_0   graph    dense    layer 0      reads: input_ids, cache_position        writes: engram_hash.cache, layers.0 window_kv_cache
submod_1   exchange experts  layer 0      reads: submod_0[0..2]                   host op experts 0
submod_2   graph    dense    layer 0      reads: submod_0[3..7], submod_1
submod_3   exchange engram   layer 1      reads: submod_2[0]                      host op engram 1
submod_4   graph    dense    layers 0-1   reads: submod_3, submod_2[1], submod_0[8,9]
                                          writes: layers.1 window_kv_cache, compressor kv_state and score_state, indexer k_cache, compress_kv_cache
submod_5   exchange experts  layer 1      reads: submod_4[0..2]                   host op experts 1
submod_6   graph    dense    layers 1-2   reads: submod_4[3..9], submod_5, and the resident layers.1 compress_kv_cache (written by submod_4)
                                          writes: layers.2 window_kv_cache
submod_7   exchange experts  layer 2      reads: submod_6[0..2]                   host op experts 2
submod_8   graph    dense    layers 2-3   reads: submod_6[3..8], submod_7         writes: layers.3 window_kv_cache, indexer k_cache, compress_kv_cache
submod_9   exchange experts  layer 3      reads: submod_8[0..2]                   host op experts 3
submod_10  graph    dense    layer 3      reads: submod_8[3..7], submod_9         -> the step's output
```
Every graph segment that owns an attention layer writes that layer's caches.
`submod_6` reads the cache `layers.1 compress_kv_cache` that `submod_4` wrote.
The cache travels as a resident, not as an edge. In this model `submod_6` also
reads edges from `submod_4`, so the order between them is the same either way;
the document still states the write, because the planner must not depend on
that coincidence.

**The step plan: ordering rules and stream assignments.**
The Go planner (`vxpu/pkg/stepplan`) computes each node's wait dependencies
and stream assignment from the `StepGraphDescription`.

A stream is a queue of GPU work that runs in order; work on two streams can
run at the same time. A wait is a dependency where a node starts only after
every node it waits on has finished; nodes placed on one stream are in order
as well.

Two nodes may not run at the same time when they touch the same memory and
at least one of them writes it. From the step graph description, for each
node in call order:
- it waits on every earlier node whose output it reads (an edge);
- if it reads a resident, it waits on the last earlier node that wrote that resident;
- if it writes a resident, it waits on the last earlier writer of it and on every node that read it since that write.

Nothing else orders two nodes. In particular, how a node executes (graph or exchange)
does not order them.

A step runs on a main stream. A graph node may run on a second, parallel stream
when nothing it waits on is the node placed last on the main stream. An exchange
node stays on the main stream. The main stream joins the parallel stream before
the step produces its output.

Consider a step graph with five nodes in call order:
- `H`: an exchange node that writes a resident cache.
- `D1`: a graph node that reads an edge from `H`.
- `D2`: a graph node that reads an edge from `D1` and reads the resident cache that `H` wrote.
- `D3`: a graph node that reads only a step input.
- `D4`: a graph node that writes the resident cache.

The planner computes the following waits and streams:
- `H`: waits on nothing (`[]`). Because `H` is an exchange node, it stays on the main stream. The last node placed on the main stream is now `H`.
- `D1`: reads an edge from `H`, so it waits on `[H]`. Because `D1` waits on `H`, which is the node placed last on the main stream, `D1` stays on the main stream. The last node placed on the main stream is now `D1`.
- `D2`: reads an edge from `D1` and the cache that `H` wrote, so it waits on `[D1, H]`. Because `D1` is the node placed last on the main stream, `D2` stays on the main stream. The last node placed on the main stream is now `D2`.
- `D3`: reads only a step input, so it waits on nothing (`[]`). Because `D3` is a graph node and none of its waits is `D2` (the node placed last on the main stream), `D3` runs on the parallel stream. With no waits, `D3` starts at once on the parallel stream and runs alongside `H`, `D1` and `D2`. The last node placed on the main stream remains `D2`, and the main stream joins the parallel stream before the step's output.
- `D4`: writes the resident cache, so it waits on `H`, the last writer of that cache, and on `D2`, which read the cache since that write: `[D2, H]`. Because `D4` waits on `D2`, which is the node placed last on the main stream, `D4` stays on the main stream. The last node placed on the main stream is now `D4`.

**Executing a step plan during capture.**
The Python runtime executes a step plan without making ordering decisions of
its own. When `VXPU_STEP_PLAN_FILE=<path>` is set (requiring `VXPU_STEP_GRAPH=1`),
the engine reads the plan and hands it to the step graph capture.

`StepGraph._capture_all` captures each node on the stream the plan names
(`STREAM_MAIN` or `STREAM_PARALLEL`). The capture loop iterates over the ordered
nodes: graph segments run on the GPU as compiled AOTInductor runners, while
exchange nodes execute through an exchange (`_exchange`) with the outside that
records the copies to pinned memory, the flag the CPU thread waits for, the wait
for the reply, and the copy back. For cross-stream waits, the stream waits on the
recorded events of the dependency nodes before launching the node's kernels.
Same-stream waits rely on stream execution order. Before producing the step's output,
the main stream joins the parallel stream. Without a plan, every node runs on the
main stream in call order with no extra streams or events.

The capture verifies that the plan matches the graph. The plan reader refuses a
plan whose node names do not match the graph's nodes in call order, including
any duplicate, missing or extra node. It also refuses any stream that is
unspecified or out of range, and any wait that names an unknown node, a later
node, or the node itself. The capture does not judge whether the waits are
sufficient; it relies on the planner's decisions.

While a graph is captured across multiple streams, the CUDA caching allocator
can return intermediate scratch memory from a kernel on one stream to its
capture pool. If another stream allocates memory before the first kernel
completes, the allocator hands that same memory to the second stream and
corrupts concurrent computations. To prevent this, the capture holds all
intermediate scratch tensors (`capture_scratch` in `triton_ops.py`) until the
graph is built: no scratch address is reused while the graph is being captured.

**Step actuals attached to the plan (EXPLAIN ANALYZE for the step).**
After a run with `VXPU_STEP_PLAN_FILE=<path>`, the engine writes the step plan with
actual execution times attached to each node (`NodePlan.stats`). A plan that no
run has touched carries no statistics sub-message at all. This produces `EXPLAIN ANALYZE`
for the decode step (§1, The plan documents): the same plan the planner wrote,
with actual device times beside each node.

By default, the engine writes the analyzed plan back in-place to `VXPU_STEP_PLAN_FILE`.
When `VXPU_STEP_PLAN_OUT=<path>` is set, the engine writes to that separate destination
instead. This handles three operational cases:
1. *Multiple processes per card or host*: when multiple worker processes share a
   single input plan, each process can point `VXPU_STEP_PLAN_OUT` to its own
   card-specific path (for example `stepplan-card0.json`), preventing concurrent
   processes from overwriting each other's measurements.
2. *Read-only plan delivery*: when the input plan is mounted read-only (such as
   from a Kubernetes ConfigMap or a container image layer), setting `VXPU_STEP_PLAN_OUT`
   directs actuals to a writable scratch volume. If `VXPU_STEP_PLAN_OUT` is omitted
   and writing in-place fails due to a read-only filesystem or permission error,
   the engine logs a warning without aborting the inference run.
3. *Preserving the planner's input*: setting `VXPU_STEP_PLAN_OUT` keeps the original
   plan file untouched on disk while emitting the measured actuals document alongside it.

We group all measurements taken from a real run into one sub-message, `NodeExecutionStats`
(`stats` on `NodePlan`), apart from the planner's decisions (the stream and the waits).
Across plan documents the field is `stats`, typed as `NodeExecutionStats` on nodes,
`SiteExecutionStats` on call sites, and `PlanExecutionStats` on the plan file.
An unexecuted plan omits the message entirely, so checking whether a plan has run is a single
presence check (`node.Stats != nil` in Go, `node.HasField("stats")` in Python). Inside `NodeExecutionStats`,
`device_time_ms` records the median device execution duration in milliseconds, alongside an
integer `sample_count` recording how many step executions contributed to the median.
Optional fields preserve presence, so an unmeasured metric remains unset and distinct
from a measurement of zero. When in-graph phase stamps are active (`VXPU_PHASE_TRACE=1`),
the engine records device timestamps for each node across the steps of the call. It takes
the median duration in milliseconds for each node over those recorded steps (`select_phase_rows`),
excluding steps from earlier calls in the same process. We choose the median because GPU execution
times occasionally experience host scheduling or CUDA driver outliers. The median
captures typical node cost, matching `median_step_ms` and `per_class_device_times` in
the run record, while `sample_count` allows the planner to weigh sample reliability.
When stamps are inactive, the run completes normally and the engine clears `stats` on
every node so readers see that times were not measured. Handing a previously measured plan
back to an engine clears any existing statistics so that output actuals reflect only the
current run.

The Go tool displays the plan actuals using `vxpu plan show [--graph FILE] PLAN_FILE`.
It prints one line per node with the node name, where it ran (its stream), its
execution mode (`graph` or `exchange` when `--graph` is supplied), and its measured
device time.

**Mailboxes as message rings.**
In `vxpu/python/vxpu/mailboxes.py`, a mailbox is the communication boundary
where an exchange posts inputs to owners of computation and collects their
replies. The message ring (`csrc/ring.h`, `rings.py`) is the concrete,
lock-free transport that mailboxes will use between processes on one machine;
no mailbox uses it yet. They are one vocabulary, not two: an inbox and outbox
across a placement boundary, with the ring serving as the zero-copy, timed
transport under the mailbox interface.

A message ring is a single-producer, single-consumer queue of fixed-size slots
in shared memory (or pinned host memory), backed by a memory-backed file (such
as in `/dev/shm`, created readable and writable by its owner only with mode
`0600`; unlinked by the coordinating process or by calling `close(unlink=True)`
when the exchange finishes). It guarantees:
- **Lock-free and system-call-free message passing**: producers publish with an
  atomic release store, and consumers poll with an acquire load. No mutex,
  semaphore, or system call sits on the fast path.
- **Intact and ordered delivery**: every message arrives intact and in published
  order, including when the ring wraps repeatedly and when the reader runs slower
  than the writer.
- **Timers on every crossing**: each message slot header records steady-clock
  timestamps (`written_ns` and `read_ns`), so every crossing duration is directly
  measured and reported rather than inferred.
- **Liveness detection without spinning forever**: if a writer or reader dies,
  the survivor detects the dead process (via process state in `/proc` or `ESRCH`)
  and aborts with a clear message instead of spinning indefinitely.
- **Busy-wait polling**: a waiting process spins with CPU pauses (`_mm_pause`
  or `yield`) and then yields the thread, deliberately occupying a core until a
  message arrives, an attached peer dies, or its timeout expires. A peer that
  has not attached yet is not an error; the waiter keeps waiting until the peer
  attaches, or until the caller's timeout expires.

A Go process uses the ring by mapping the shared memory file into its address space
with `syscall.Mmap`, indexing the little-endian `RingControl` and `SlotHeader` structures
at the 64-byte aligned offsets that `ring_layout` reports, and reading and publishing
messages using `sync/atomic` (e.g. `atomic.LoadUint64` and `atomic.StoreUint64` on
`head`, `tail`, and `seq`, and `atomic.LoadUint32` on `closed`) with no locks or cgo.
Because Go atomic loads and stores are sequentially consistent, they fully synchronize
with the C++ acquire/release memory orders across the process boundary.

## 7. Numerics

Every implementation declares whether it is bit-identical to the model's
own computation or equal up to rounding, and the selftest holds it to
that: token-exact against a reference, and with logits compared, for
every tiny model. The kernel check is where a variant's class is first
held to the reference, before any engine runs it. Where a choice changes
the result (an expert computed on the GPU instead of the CPU rounds
differently), that is a property of the plan, recorded with every result
the engine returns.

**Direction.** Sums whose order does not depend on placement (§6), so
that moving work between devices changes the speed and not the answer.

## 8. Benchmarking

The `vxpu bench` command runs the league benchmark, measuring served models behind an OpenAI-compatible chat completions endpoint (`POST <base>/v1/chat/completions`) using the method from the sparkDash benchmark tool used in published DeepSeek evaluations. It measures throughput and latency across concurrency levels (1, 2, 4, 8, and 16 streams) for prose and code prompt sets.

**What the tool measures and excludes.**
The tool measures aggregate tokens a second across each concurrency wave, tokens a second for each stream, their median, and the median time to first token (TTFT).

The throughput calculation excludes both the model prefill and the first generated token. The client records the arrival timestamp of the first stream event carrying content as `tFirst`, and the arrival timestamp of the last event carrying content as `tLast`. The server reports total generated tokens in `usage.completion_tokens`. For each stream, throughput is `(completion_tokens - 1) / (tLast - tFirst)`. For a wave of concurrent streams, aggregate throughput sums `(completion_tokens - 1)` across all streams and divides by the wave span from the earliest `tFirst` to the latest `tLast`. The request start time to `tFirst` defines the time to first token.

Times come from the client's own clock as stream events arrive. Token counts come from the server's reported usage. When a single stream event carries several tokens, the arrival timestamp of that event marks `tLast` (or `tFirst` if it is the initial content event), and the server's reported `completion_tokens` provides the exact token count. Multiple tokens in one event do not change the count, though they quantize event timing to network chunk boundaries.

**Prompt format.**
The benchmark takes prompts from an external JSON file rather than hardcoded strings:

```json
{
  "prose": {
    "base": "Describe the water cycle and weather patterns across different climates."
  },
  "code": {
    "tasks": [
      "task_00\ndef task_00(): pass",
      ...
    ],
    "tail": "Return runnable Python with no markdown fences.",
    "warmup_task": "echo_value\ndef echo_value(x): return x"
  }
}
```

The prose benchmark runs a warm-up request with the base prompt and `max_tokens: 32`. At concurrency 1, it sends the base prompt alone. At concurrency n greater than 1, stream i (counting from 1 to n) receives the base prompt with ` (stream i/n)` appended. The code benchmark runs a warm-up request combining the warm-up task, a newline, and the instruction tail. For measured levels, stream i (counting from 0 to n minus 1) sends task i from the task list followed by a newline and the tail. The file must provide at least as many code tasks as the highest concurrency level requested.

**Reaching our own engine.**
The router provides an OpenAI-compatible HTTP endpoint for our engine so `vxpu bench` and standard clients can reach it without speaking internal gRPC. An OpenAI-compatible endpoint keeps benchmark tools protocol-agnostic, allowing them to measure vXPU and other serving engines identically while opening vXPU to standard LLM clients, evaluation suites, and frontends.

**Chat tokenization and the vocabulary fact.**
The router provides the OpenAI-compatible HTTP chat endpoint, but the model artifact owns the tokenizer and its Jinja chat template. Tokenization therefore runs on the executor holding the model rather than in the router. When the router receives a chat completion request, it sends a `TokenizeChat` RPC to the executor holding the model. The executor renders the chat messages into prompt token IDs using the model's chat template and template options. Template options are carried as JSON so boolean and numeric types are preserved for the Jinja template renderer.

Decoding generated tokens into text avoids per-token RPCs to Python. Instead, the router fetches the model's vocabulary table once via `GetVocab` when the model loads. The vocabulary fact reports the exact byte slice each token ID contributes to the output text, defined as what decoding adds after a prompt: `decode(prompt + reply)` minus `decode(prompt)`. Special tokens and control tokens produce empty byte slices. During streaming generation, Go concatenates the byte pieces directly from the table, handling multi-byte UTF-8 sequences and prefixes without calling into the Python runtime.

Benchmark suites such as league benchmarks require generating an exact number of tokens. The `GenerateRequest` carries an `ignore_eos` boolean flag. When set, the generation loop continues past end-of-sequence tokens until it reaches the requested token limit, reporting `length` as the finish reason.

