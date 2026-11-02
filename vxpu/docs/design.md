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
rotary, quantization, index arithmetic). Four regions of the attention
block and the MoE are now ops, each the fork's computation as a
bit-identical reference, a fused tensor body the lowering may choose, and
Triton kernels built from two templates:
- a GEMV over a block-scaled fp8 weight with an optional prologue (RMS
  norm, rotary, quantization to the fp8 grid per 32) and epilogue (rotary,
  or a SwiGLU over two weights' products with the quantization of the
  result), `_gemv_pe_kernel` and `_dual_gemv_swiglu_kernel`;
- a per-row finishing kernel (norm, rotary, quantization), `_row_post_kernel`.

| op | replaces | kernels before | after | on an L4, one row |
|---|---|---|---|---|
| `vxpu::shared_ffn` | the shared expert: three fp8 linears and the SwiGLU | 9 | 3 | 152 µs against 154 |
| `vxpu::attn_q` | q down-projection, q_norm, up-projection, rotary | about 9 | 4 | 208 µs against 206 |
| `vxpu::window_kv` | kv projection, kv_norm, rotary, act_quant | about 6 | 3 | 21 µs against 25, bit-identical |
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
hero machine measures for itself.

**Hero run 7: layer sharding and the region ops on the real model (Oct 30,
GKE, four RTX PRO 6000; measured).** Each GPU holds a contiguous range of
layers whole, dense weights and experts (`VXPU_LAYER_DEVICES`, ranges
0:8, 8:20, 20:32, 32:40 and the drafter's 40:43 back on the first; 65,
87, 87 and 58 GB of experts), and the step crosses GPUs once per range
instead of fanning every layer's experts out to four GPUs and back. The
aim was the one-GPU floor at one row, then written as 10.3 ms (8.9 ms
once the dense bytes were counted from the artifact, below). It was not
reached:

| one row, no drafter | step | kernels a step |
|---|---|---|
| hero 6's code, every layer's experts over four GPUs | 21.2 ms | about 2,090 |
| the same placement, the region ops in the graph | 21.3 ms | |
| sharded, the region ops | 20.3 ms | 1,742 |
| sharded, the ops' Triton kernels forced | 20.2 ms | |
| floor as then estimated (10.5 GB dense + 4.5 GB of experts at 1,460 GB/s) | 10.3 ms | |
| floor from the artifact's bytes (8.5 GB dense + 4.5 GB of experts) | 8.9 ms | |

The region ops took 350 kernels out of the step and no time: on this
card the small kernels they fold away were already cheap, and the GEMVs
they wrap were already near their bytes. The per-kernel profile of the
sharded step (torch.profiler over the captured step) says where the 10
ms went: the dense GEMVs 7.7 ms against a 7.2 ms floor; the expert
kernels 6.9 ms against 3.1; the FFN and hyper-connection front kernels
1.9 ms against 0.2; small kernels 2.3 ms; gaps 1.5 ms. The sharding
itself works, token-identical to one GPU on two L4s, and it cost two
more fixes on the real model than on the fixture: a cache written by a
layer on one GPU and read by layers on another (the compressed KV that
layers 20 to 39 share, which no single card can hold with its experts)
is replicated and refreshed once a step; and torch.export reads a
graph's "cuda" factories as cuda:0 whatever the current device, so each
segment's graph gets its factories pointed at its GPU when it is moved.

**7(b): the fp4 experts go to the tensor cores as they are.** The expert
kernels decoded every fp4 nibble to bf16 in software before a bf16 dot,
because they were written on an L4, whose tensor cores take no fp4, and
carried to Blackwell unchanged. At one token the gate/up kernel read its
75 MB in 113 µs (666 GB/s); the same loop without the decode ran at 914
GB/s and the loads alone at 1,660 (above the card's copy bandwidth: the
128 MB L2 was holding part of the 75 MB, which a benchmark that replays
one layer's weights must not let it). DeepSeek's expert format is MXFP4,
e2m1 with an e8m0 scale per 32, which Blackwell's tensor cores take
directly. Each expert kernel now has two bodies and the card chooses:
compute capability 10 and later pass the packed bytes and the scale
bytes to `tl.dot_scaled` unchanged; earlier cards keep the software
decode. Nothing configures it. With a cold cache (eight layers' experts
a replay, 600 MB) at one token, gate/up: software body 117 µs, native
75.6 µs with its own tile shape (the best of 108), floor 52 µs. In the
step the expert kernels fell from 6.9 to 4.9 ms and the one-row step
from 20.5 to 18.2 ms (p10/p90 18/24; the same pod, the same code, the
body switched by `VXPU_FP4_NATIVE` for the A/B). Still 1.8 times the
floor. The lesson is the thesis in one row: a kernel written on one card
and carried to another silently loses the new card's formats, and the
library needs a native body per format per card with the choice made
from the card, later from the planner's measurements. The same audit
found the batched fp8 GEMV templates converting MXFP8 weights to bf16
and applying scales in software before a bf16 dot (a native body for
hero 9), the fp8 round trip emulated in fp32 arithmetic in six kernels
(the hardware cast exists on every card we run on), and the compressed
KV's fp4 quantization running as the aten emulation. The remaining gap
at one row is the FFN and hyper-connection fronts (1.9 ms for 0.2 ms of
bytes), the expert kernels' last 1.8 ms, the small kernels and the gaps.

**7(b), the rest, as four parallel workstreams (Oct 31, each on its own
branch and GPU, each measured cold on the RTX PRO 6000 with argument sets
larger than the L2; merged into one step afterwards).** The per-kernel
profile separates their effects, which is what lets them run at once.

| workstream | change | per call at one row, before → after | launches a step |
|---|---|---|---|
| fronts | the FFN and HC fronts split the row across programs: a per-chunk statistics kernel, a chunk × group dots kernel, the finishing kernel summing partials (`gpu-triton-split`, chosen ahead of the one-pass pair) | ffn_front 31.5 → 13.5 µs, hc_front 14.7 → 6.8 µs | +39 |
| experts | a decode step's experts in two launches a layer: the gate/up kernel looks its slot up itself, the down kernel forms h from g and u in its loads (SwiGLU + per-32 quantization), the token's last program sums its replies; the step graph skips the combine when one pool holds the layer | whole op 127.9 → 107.3 µs | −240 |
| glue | the fp8 round trip is the hardware cast; wo_a's GEMV quantizes in its epilogue; the narrow projections (attn_q, window_kv) and the shared expert's dual GEMV split K across programs with the per-row kernel summing partials | attn_out 62 → 60, attn_q 56 → 49, window_kv 17 → 9, shared_ffn 44 → 42 µs | −40, and −120 if the lowering takes the 4-kernel shared_ffn body |
| drafter | the drafter's segments get their layer ids so sharding places them; a state tensor written away from its owner GPU is detected | correctness (1 of 310 drafts accepted) | |

Measured and dropped on the way: quantizing a GEMV's input in its
prologue costs 7 to 10 µs at one row (every program redoes it on the
padded tile); quantization belongs in the producer's epilogue. The
expert kernels' tail stays: the loads alone reach 1,400 GB/s, the dot
with one live row in a 16-row tile adds 18 µs, and no tile shape moves
it. The merged step, one row, sharded, the drafter's stages on the last
GPU (measured): 15.4 ms median (p10/p90 15/25), 1,422 kernels a step
(hero 7: 1,742), 14.2 ms of GPU time in 15.2 ms of wall; by part: dense
GEMVs 6.8 ms, experts 4.5 (floor 3.1), fronts 0.8 (floor 0.2), the rest
2.1. The dense GEMVs came in under the 7.2 ms written above as their
floor, so that floor was wrong. Recomputed from the artifact's binding
(rows-b1, the bound tensors that are not state): the 40 layers' dense
weights are 6.9 GB, the two engram layers 0.3 GB, and the head 1.3 GB in
bf16 (the embedding is read one row at a time), 8.5 GB a step, 5.8 ms at
1,460 GB/s; with the 4.5 GB of selected experts the one-row floor is 8.9
ms, not 10.3. Hero 7 to here: 20.3 to 15.4 ms, 1.7 times the one-GPU
floor; the dense GEMVs run at 86% of theirs. Still visible in the profile: two
compiler-fused reductions at 154 µs each, the greedy sampler at 100 µs
with a random draw it does not need, and nine top-k gathers of 11.5 µs
(the indexer's, #484). The sampler became its own op (171 to 55 µs for
the choice). The two reductions are the engram layers' fp8 projection
([25600, 6144], 157 MB) running the compiler-fused body although the
lowering measured and chose the GEMV kernel for that site (112 against
150 µs): with `fp8_linear` taken out of the fusion list the two kernels
vanish and the step's GPU time drops by 0.17 ms (14.17 to 14.00; the
median step is unchanged within noise). So the per-site choice is lost
for sites inside those segments, most likely a package cache key that
does not carry the choice: a lowering bug to fix, since a planner whose
choices are not applied is no planner.

Batched rows and the drafter, sharded, the final 7(b) code (Oct 31, the
clean rerun on an idle pod; measured; "same" is one prompt in every
row, "different" a prompt per row; hero 7's numbers, where they
existed, in parentheses):

| rows | one prompt | same prompt per row | different prompts |
|---|---|---|---|
| 1 | 15.2 ms, 60 tok/s (20.3) | | |
| 1 + drafter | 28.5 ms, 37 of 130 drafts accepted, 2.46 tokens a step, 86 tok/s | | |
| 9 | 24.8 ms | 20.5 ms, 422 tok/s | 39.4 ms, 228 tok/s |
| 8 + drafter | 69.2 ms, 2.78 tokens a step | 48.7 ms, 320 of 920 accepted, 452 tok/s | 116 ms, 2.89 tokens a step, 164 tok/s |
| 33 | 40.6 ms | 39.9 ms, 802 tok/s | 112 ms, 295 tok/s |

Against the floors: one row is 1.7 times 8.9 ms; one row with the
drafter is 86 tok/s against the six-card recipe's 285 at two GPUs'
bandwidth per layer and against about 120 at one; the batched floors
wait on the distinct-expert count. A run on a pod another job was
time-slicing gave flat medians at round numbers (20.0, 30.0, 60.0 ms,
14 ms of GPU time in a 235 ms wall) and was discarded: a step
measurement needs an idle pod, checked with nvidia-smi before the run.

Every same-prompt run has token-identical rows and accepts rows times
what one row accepts. Two bugs came out on the way to this table. The
batched drafter had collapsed (73 of 2,155 drafts at eight rows) because
the drafter's step returns a strided view of its tokens and the native
loop copied it to the host with one flat memcpy: only row 0 read its
own drafts. The capture makes the output contiguous now, and the loop
refuses a strided tensor. And the experts' direct path (two launches a
layer) streamed an expert once per pair, so one prompt in nine rows read
each of its six experts nine times (34 ms a step where the grouped path
took 26); it now serves one row's pairs only.

**The drafter's step is mostly bytes.** The premise that a six-token
verify reads what a one-token step reads holds for the dense half only:
the MoE half reads every expert any of the six tokens selects, about 25
distinct experts a layer instead of 6 (inferred from the expert kernels'
bytes at bandwidth: 215 + 119 µs a layer). Per-kernel profile of one
speculative step at one row (measured): the verify step 24.5 ms of GPU
time (experts 13.3, dense GEMVs 7.2, fronts and attention 1.7, small
kernels 2.3), the drafter 2.3 ms (its 1.3 GB bf16 head 0.9 of it, five
sequential Markov heads and argmaxes 0.4), and 3.2 ms of host gaps
between the drafter's graphs and the verify's (the native loop syncs,
copies the drafts to the host, writes the verify's input ids, syncs
again; the plain loop has 0.8 ms of gaps) in a 30 ms step. Floors,
estimated: the speculative step about 21 ms (dense 5.8, experts about
13, drafter about 1.7), 8.5 ms a token at 2.46 accepted, against 8.9 ms
a token without the drafter. So with five drafts, speculation is nearly
byte-neutral for this model at one row on one GPU's bandwidth; its gain
today (12.2 against 15.0 ms a token) comes from covering latency, and
the draft length is a bytes-against-acceptance trade for the planner to
measure. The same arithmetic explains the six-card recipe: at two GPUs'
bandwidth per layer its verify step's bytes take about 9.3 ms, which is
the 9.6 ms it reports with and without the drafter; a single row past
about 120 tokens/s on these cards needs more than one GPU's bandwidth
per layer. One change landed from this: on Blackwell the one-weight GEMV
gives each program 32 weight rows (26.6 to 24.2 µs a launch at six rows,
bit-identical; the two-weight and prologue kernels keep 16, which they
measured faster with). The host round trip is the next removable piece:
building the verify's input ids on the device from the drafter's output
would take the two syncs out.

The floor with different prompts depends on how many distinct experts
the rows select a layer, which is not measured yet: at most 198 for 33
rows (150 GB a step, 100 ms), about 156 if routing were uniform over
the 384 (120 GB, 80 ms), and fewer with the skew real routing has. So
the 120 ms step is somewhere between 60% and 85% of its bytes, and the
distinct-expert count per step is a statistic the planner must record
(section 5) before any of these floors is more than a bound. The
same-prompt rows are
token-identical to each other; on hero 6's placement, with every layer's
experts over four GPUs and the replies exchanged between them, they were
not (and that placement's 9-row step was 60 ms in the new code against
31 in hero 6's, a regression the sharded placement does not have). The
drafter under sharding is broken: one row accepts 1 draft of 310 where
the unsharded run accepts 40 of 115. The drafter's stages (layers 40 to
42) sat on the first GPU while the backbone ends on the fourth, and the
decode's last segment writes the drafter's rings (state tensors the
drafter owns on another GPU): the runner's replica rule covers a state
written by its owner and read elsewhere, not one written elsewhere, so
the writes went into the replica and the drafter read a ring that never
changed. The placement with the drafter on the last GPU is the fix being
measured; the rule needs the other direction too. The batched drafter on
hero 6's placement accepted 49 of 2,285 drafts at eight rows with the
same prompt in every row: the exchange path's rows differ, and that is
the suspect for it.

**Hero run 8's acceptance, agreed Oct 31.** Hero 8 is the planner in
Go, and it is judged on the real model at one row, not on the fixture:
the step from 15.3 ms to 11.5 ms or better (floor 8.9), with the
drafter line, 9 and 33 rows reported in the same table as always. The
remainder of the one-row gap is structure, not bytes, and the planner
owns structure, so hero 8 must deliver these rather than hand them to
hero 9:
- op boundaries in the fork so that a layer runs about a dozen kernels
  (35 today): the attention input quantized once and passed to both
  consumers, the rotary table indexed inside the kernels, the post-mix
  and the sparse attention's index preparation folded into their
  neighbours; the planner instantiates the templates at those
  boundaries;
- independent kernels on parallel branches of the step graph (the
  narrow projections and the small kernels run alone on one stream per
  GPU today);
- three crossings a step, one per range boundary, with the replica
  refreshes and step inputs staged ahead;
- the per-site choice applied and reported (#490): the step says which
  body each site ran, checked against the record;
- the statistics the floors need: distinct experts a step, and a cost
  record with two kernel bodies per op measured per site.
The fronts' finishing fold and the expert kernel's last 20 µs a layer
are kernel items that may land alongside but are not the test. This is
the fourth time "fewer kernels, fold the glue, fewer crossings" has been
written as the plan; the first three delivered part of it from the
Python side and deferred the fork's boundaries each time, and estimated
small kernels at zero when they cost about 2 µs each. Hero 8 does not
get to do that again.

**Hero 8, first measurements (Oct 31, night).** The planner's round trip
works on the real model: `vxpu kernels plan` read the record of a one-row
step (307 sites) and wrote a table of 13 entries plus the knobs; a step
run with that table applied measured 14.5 ms (p10/p90 14/16), its record
checked against the table reports 0 sites that ran another body, and the
profiling check found no heavy compiler-fused kernel: the engram
projection's 154 µs reductions (#490) are gone when the choice is given
explicitly, so the default path loses a choice the table path honours.
The step graph's crossings went from 23 flags to 3 and one row from
15.3 to 14.5 ms; the step is a chain at the node level (85 of 85 nodes
on the critical path), so parallel branches need the planner to cut the
shared expert and the attention's narrow projections into their own
nodes before they buy anything. The distinct experts a step are now
measured (one row 6.0 a layer; 33 different prompts 90 to 113 a layer,
depending on the prompts and steps): the 33-different-prompt step reads
about 70 to 85 GB of experts, and its per-kernel profile shows the
grouped expert kernels at their bytes (1,024 + 459 µs a layer for about
2.1 GB), the dense multiplications at 33 rows at about 60% of bandwidth
(33 µs a launch against 21 at one row: the software MXFP8 conversion,
hero 9's item), and 3.4 ms of GPU-to-GPU copies carrying 33 rows' state
across the range boundaries. The drafter at shorter verify blocks: 5 tokens
a step (four drafts verified) measured 26.8 ms at 2.67 tokens a step, 99.5
tok/s at one row, against 29.8 ms at 2.46 and 83 tok/s with 6 (the
model's own block of five drafts; the fifth draft is rarely accepted and
its verification costs bytes). The first run at 5 tokens read 210 ms
flat with a 346 s load: the page-cache thrash, not a kernel; the 3- and
4-token runs (29.9 ms at 2.21, 30.0 at 2.37) predate the fix and are
being rerun. The draft length is the planner's knob, measured as such. The engine now releases the
checkpoint files' page cache after loading on the pods too, which ends
the mid-step faults that contaminated three matrices.

**Hero 8, the step-graph and planner code measured clean (Oct 31, early
morning; benchmark pod, one process at a time, the page-cache releases
in the engine; measured):**

| rows | one prompt | same prompt per row | different prompts |
|---|---|---|---|
| 1 | 14.5 ms, 69 tok/s | | |
| 1 + drafter, 4 drafts verified | 26.8 to 30.0 ms, 2.67 tokens a step, 90 to 99 tok/s | | |
| 9 | 24.7 ms | 20.4 ms, 440 tok/s | 37.5 ms, 240 tok/s |
| 8 + drafter, 4 drafts | 64.5 ms | 45.4 ms, 467 tok/s | 99.3 ms, 202 tok/s |
| 33 | 40.8 ms | 40.1 ms, 800 tok/s | 105 ms, 300 tok/s |

The 33-different-prompt step reads 90 to 113 distinct experts a layer
(measured), about 63 ms of bytes: its profile puts the grouped expert
kernels at their bytes, the dense multiplications at 60% of bandwidth
(33 µs a launch at 33 rows against 21 at one), 3.4 ms in the copies that
carry 33 rows' state across the three range boundaries, and about 4 ms
of compiler glue. The planner's records now exist for 1, 9 and 33 rows;
on the 9-row record the planner picks the fused body for `attn_q` where
the one-row record picks the kernel, which is the per-shape choice the
design asks for. The fork's op boundaries are the remaining workstream
and the one that moves the one-row number.

**Hero run 8, result (Oct 31, morning; four RTX PRO 6000, layer ranges
0:8, 8:20, 20:32, 32:43 with the drafter's stages on the last GPU, the
merged hero 8 code and the re-exported artifacts, one process on the
pod at a time; measured):**

| rows | one prompt | same prompt per row | different prompts |
|---|---|---|---|
| 1 | 14.0 ms, 67 tok/s (13.7 with the second round's parallel branch) | | |
| 1 + drafter, 4 drafts verified | 27.0 ms, 2.67 tokens a step, 92 tok/s | | |
| 9 | 23.2 ms | 19.0 ms, 447 tok/s | 36.9 ms, 246 tok/s |
| 8 + drafter, 4 drafts | 59.9 ms | 44.7 ms, 505 tok/s | 101 ms, 172 tok/s |
| 33 | 35.9 ms | 35.0 ms, 915 tok/s | 103 ms, 332 tok/s |

Against the floors and the production stacks: one row is 1.57 times the
8.9 ms floor (hero 7: 20.3 ms, 2.3 times; 7(b): 15.3, 1.7 times); the
six-card vLLM recipe reports 9.6 ms with two GPUs streaming each layer,
and 285 tok/s with its drafter against our 92 on one GPU's bandwidth.
The step launches 1,026 kernels (hero 7: 1,742; 7(b): 1,422) and crosses
GPUs on 3 flags (23); its GPU time is 12.9 ms in a 13.6 ms wall under
the profiler. Every same-prompt run has identical rows and accepts rows
times what one row accepts. The same matrix on the other pod, with a
two-second warm-up before the timed runs, agrees: one row 14.0 ms
(p10/p90 14/14), 9 rows 23.2 / 24.0 / 37.3, 33 rows 35.7 / 33.5 (1,000
tok/s) / 99.6, the drafter 28.9 to 29.9 ms at one row and 45.2 ms at
eight; the drafter's step varies by 2 to 3 ms between runs, its host
work.

A second round added two things to that table. The lowering traces each
fused body on the GPU of the segment it lands in (it traced on whichever
GPU was current and baked that index into the body's factories, which
killed sharded steps whenever a fused body was picked for a segment on
another GPU), and the compiled packages' key names the kernel source,
which closes the engram projection's stale-package symptom (#490). And
the attention's two projections are cut into their own nodes and run on
a parallel branch of the step graph: one row 14.0 to 13.7 ms (on/off
back to back: 14.0 / 13.7; 9 rows 23.2 / 22.9, 19.0 / 18.7, 36.9 /
36.7), the first overlap the step has had, at the cost of 160 more
compiled segments and about two minutes more load. The shared expert
cannot join the branch yet: its op takes the routed experts' sum for
its epilogue, so it must split into the gate/up projections (beside the
experts) and the down projection with the epilogue (after), an op and
fork change. Hero 8's one-row number is therefore 13.7 ms, 1.54 times
the floor (the second pod, with everything merged: 13.7 ms, 13.4 ms of
GPU time in 13.4 ms of wall, 1,026 kernels).

The acceptance asked for 11.5 ms and a layer of about a dozen kernels.
Hero 8 reached 13.7 ms and 21 kernels a layer, so it did not meet it,
and the gap has names. One more thing the night established and fixed:
on these pods, step times of a fresh process sometimes read flat 30 or
60 ms for a whole run (14 ms of GPU work), on either pod, with no other
process on the GPUs. A per-step host trace (`VXPU_STEP_TRACE=path`,
one JSON line per step with each phase) put all of it inside the first
n-gram lookup's host node: inductor compiles that node's dequantization
loops as OpenMP parallel regions, issued once a step from the step
host's thread, whose OpenMP workers sleep between steps, and waking
them on this 384-vCPU machine costs a quantized 10 ms (a microbenchmark:
p90 9.9 ms at a 500 ms cadence, 46 µs at a 14 ms one). Not disk, not
memory, not another process. Host node packages now build with their
loops serial: six fresh processes all at 13.6 ms with p10 equal to p90,
the host node 0.012 ms. That closed it for one row and for 33 rows (32.8
ms with p10 equal to p90, 1,021 tok/s same prompt) but not for nine rows
or the drafter's steps, where the trace found the one parallel operation
left in the host node: torch runs a copy of more than 32k elements on
OpenMP's workers, and a nine-row reply is 55k elements (one row's is 6k).
The reply is a plain memcpy now: nine rows 22.6 / 18.5 / 35.6 ms over
three fresh processes with p10 equal to p90, the drafter 25.8 to 26.0 ms
(p10/p90 25/27), host nodes 0.02 ms. Any other torch op over 32k elements
on the host path has the same exposure; the trace is the instrument. The page cache at the pod's memory limit was a
second, earlier cause and is handled by the engine. The exposure stays
for any OpenMP region issued at a long cadence from the host path (the
CPU expert path when experts are served from RAM); the trace is the
instrument. What landed: four op boundaries in the fork (the
attention input quantized once, the rotary table indexed inside the
kernels, the hyper-connection post-mix as the last GEMV's epilogue, the
sparse attention forming its selection and finishing the output path
itself), 400 launches fewer; crossings 23 to 3; the verify's input ids
built on the GPU; the expert kernels' bytes through the copy engine
(TMA); the drafter at four drafts; the planner's table and record, with
the round trip on the real model (307 sites, none off the table, the
engram bug gone when the choice is given); the distinct-expert count;
the engine keeping the n-gram tables resident and the checkpoint and
bank pages out of the cache (three matrices were contaminated by page
faults before that). What did not: parallel branches, because the step
is a chain at the node level and the planner must cut the shared expert
and the attention's narrow projections into their own nodes first;
folding the fronts' finishing kernel and the shared expert's epilogue
by a last-arriving program (both slower); the persistent expert kernel.
The 21 kernels a layer are 2 experts, 7 dense multiplications, 6 front
kernels, 2 per-row kernels, 2 attention kernels, the ring write (needs
a mutating op) and the shared expert's epilogue. The next 2.5 ms at one
row are: the dense multiplications' latency at narrow shapes (1.1 ms;
needs the node cut and parallel branches, or a native MXFP8 body), the
expert kernels' one-row dot (1.2 ms over their bytes), and about 400
launches of 2 µs. At 33 different prompts the step is 103 ms against
about 63 ms of bytes: the expert kernels are at their bytes, the dense
multiplications run at 60% at 33 rows (the software MXFP8 conversion),
3.4 ms go to the copies that carry 33 rows' state across the three
boundaries, and the compiler glue is 4 ms; that is hero 9's table.

**Hero run 8, final table (Oct 31, 13:00 UTC; four RTX PRO 6000, layer
ranges 0:8, 8:20, 20:32, 32:43 with the drafter's stages on the last GPU,
everything above merged, one process on the pod at a time, a four-second
warm-up; measured):**

| rows | one prompt | same prompt per row | different prompts |
|---|---|---|---|
| 1 | 13.6 ms, 75 tok/s (p10/p90 14/14) | | |
| 1 + drafter, 4 drafts verified | 25.8 ms, 2.67 tokens a step, 103 tok/s (25/27) | | |
| 9 | 22.6 ms | 18.5 ms, 495 tok/s | 35.6 ms, 262 tok/s |
| 8 + drafter, 4 drafts | 58.8 ms | 42.9 ms, 444 tok/s | 97.1 ms, 202 tok/s |
| 33 | 35.0 ms | 32.8 ms, 1,021 tok/s | 99.2 ms, 339 tok/s |

Against hero 7 at one row: 20.3 to 13.6 ms (the floor is 8.9; 1.53
times), 1,742 to 1,026 kernels, 23 to 3 crossings, GPU time 13.4 ms in
13.3 ms of wall; the drafter from 86 to 103 tok/s; 33 same-prompt rows
from 802 to 1,021 tok/s. Every same-prompt run has identical rows and
accepts rows times what one row accepts. One stall survives: the first
prompt of this fresh process read 30 ms flat for its 63 steps before the
second read 13.6 steady; the step trace is the instrument for it.

**Hero run 9's acceptance, agreed Oct 31.** Make all four GPUs stream at
once: several batches in flight, one per layer range, so each GPU works
on a different batch's layers while the others do the same, measured as
aggregate tokens per second against the aggregate bandwidth of the four
cards (5,840 GB/s) with per-batch latency alongside. The run proves or
disproves three things: that the pipeline reaches a large fraction of
the aggregate roofline without new hand fusion; that the planner
produces this physical plan (the ranges, the pipeline depth, the
schedule, as data) and its cost model predicts the aggregate within
about 20% before the run; and that layer sharding is not a dead end for
throughput against tensor parallelism inside a layer. Expectations,
estimated from hero 8's table: four batches of 33 different prompts,
pipelined with nothing else changed, about 1,300 tok/s (today 339 for
one batch); with the two known losses fixed, the dense multiplications
at 33 rows running at 60% of bandwidth (a native MXFP8 body, the run's
one new template) and the 3.4 ms of state copies per boundary at 33
rows, 2,000 to 3,000 tok/s; under 1,500 is a failure to stop and
understand. Four one-row batches should give about four times 75 tok/s
at an unchanged 14 ms latency each; past about 17 ms the pipeline is
interfering. The planner's cost model, below, turns these into the
numbers the run is judged against. Risks: the host side (four
steps' worth of host nodes and crossings per step time; the step trace
is on from day one), GPU memory for four batches' caches and inboxes,
cross-batch correctness (the same-prompt check per batch from the first
run), and range balance (the ranges were chosen for memory, not equal
work). The acceptance table: four batches of 33 different prompts, four
of 33 same prompts, four of one row, plus hero 8's one-row and drafter
lines unchanged, each against its floor and against the six-card
recipe's 494 tok/s at 24 streams. Workstreams from day one: the pipeline
in the step graph and native loop; the native MXFP8 dense body; the
boundary copies; the planner's pipeline cost model and plan output; the
aggregate measurement harness; and, riding along, the shared expert's
split for the one-row overlap and the first-prompt stall.

**The physical plan as data, and the pipeline's cost model (hero run 9,
the planner workstream).** The sharding switches and the hand placement
file become one plan the engine reads, `vxpu-plan/1`
(`pkg/kernelplan/pipeline.go`): the card it is for, the layer ranges in
step order with their GPUs, where the ranges' experts live
("with-layers": every expert of a range's layers on its GPU), the
pipeline (batches in flight and the schedule rule, "range-order": a
batch enters range k when GPU k is free and its range k-1 has finished,
batches admitted round-robin), and the kernel table to apply. The
engine's switch today, `VXPU_LAYER_DEVICES=0:8=cuda:0,...`, is what
`PipelinePlan.LayerDevices()` renders, so an engine that still reads the
switch is driven from the same plan.

```json
{"format": "vxpu-plan/1", "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
 "ranges": [{"gpu": 0, "layers": "0:8"}, {"gpu": 1, "layers": "8:20"},
            {"gpu": 2, "layers": "20:32"}, {"gpu": 3, "layers": "32:43"}],
 "experts": "with-layers", "pipeline": {"batches": 4, "schedule": "range-order"},
 "kernels": "table-rows11-b1.json"}
```

The cost model (`pkg/planner/pipeline.go`, `vxpu plan pipeline`) prices
a batch's step per range: the range's dense weights (the head on the
last range) over the GPU's bandwidth, the experts as the distinct
experts a layer times the expert's bytes times the range's layers over
the bandwidth (at 87% of it with a prompt per row, where the grouped
kernels run), and the step's cost besides bytes, 4.1 ms fixed plus 0.61
ms a row, by the range's share of the layers. The two constants come
from hero 8's same-prompt table (one row 13.6 ms, 9 rows 18.5, 33 rows
32.8: the same 13 GB of bytes each), and the model then reproduces the
table within 3% for the same prompt, within 6% at 9 rows with a prompt
per row (37 distinct experts a layer, 35.6 ms) and within 10% at 33 (100
distinct, 99.2 ms; within 3% at the 112 the profiled run selected). The
per-layer bytes come from the artifact's binding (`ReadLayerBytes`; the
checked-in inventory of DeepSeek-V4.1-Flash is
`pkg/planner/testdata/deepseek-v41-flash-layers.json`). With N batches
in flight over K ranges, a batch step completes every max(slowest range,
sum of ranges / N) and a batch's own step takes max(sum, min(N, K) ×
slowest).

The predictions hero 9 is judged against (the test
`TestPredictPipeline_Hero9Predictions` prints them; four batches, the
current ranges, nothing else changed):

| four batches of | aggregate | a batch step every | per-batch latency | one batch alone | floor at the slowest range's bytes |
|---|---|---|---|---|---|
| one row | 259 tok/s | 3.9 ms | 15.5 ms | 13.6 ms | 406 tok/s |
| 33 rows, same prompt | 3,394 tok/s | 9.7 ms | 38.9 ms | 33.2 ms | 13,391 tok/s |
| 33 rows, a prompt per row (100 distinct experts a layer) | 1,243 tok/s | 26.6 ms | 106 ms | 89.3 ms | 1,943 tok/s |

Two things the model says before the run. The aggregate floor is not
four times a single GPU's: with four batches in flight each GPU reads
its own range's bytes for one batch per period, so the ceiling is the
rows over the slowest range's bytes alone, 1,943 tok/s for 33 different
prompts (not the "near 10,000" an earlier estimate in this section
implied by dividing one batch's bytes by four GPUs' bandwidth; that
number is the latency floor of one batch spread over four GPUs, which
the pipeline does not do). More rows per batch raise the ceiling, since
more rows share each expert read. And the ranges are not balanced for
work: 8:20 and 20:32 hold twelve layers each, 0:8 and 32:43 eight, so
the period is the twelve-layer ranges' time and the other two GPUs idle
17 to 19% of it (`Balance` in the same package searches the cuts). The
step may only be cut at the KV and index source layers (2, 8, 14, 20,
24, 28, 32, 36), and under that constraint and 96 GB cards no four-way
split has a shorter slowest range than the current one, so the
imbalance is the model's to carry, not a placement to fix; a fifth GPU
or a cut inside a KV group would change it.

**Hero run 9, first results (Oct 31, afternoon; measured on four RTX PRO
6000, four batches in flight, each batch the same prompts; the planner's
predictions were written before the runs).** `generate_rows(batches=N)`
runs N batches at once: each batch has its own state, its own step
graphs and its own native loop on its own streams per GPU, and a batch's
range k waits only on its own range k-1, so the GPUs run different
batches' ranges at the same time with no host scheduler.

| four batches of | per-batch latency | aggregate | predicted | one batch alone |
|---|---|---|---|---|
| 1 row | 14.3 to 14.7 ms | 244 tok/s | 259 at 15.5 ms | 75 tok/s at 13.6 ms |
| 33 rows, same prompt | 32.8 to 46.0 ms | 3,224 tok/s | 3,394 at 38.9 | 1,021 at 32.8 |
| 33 rows, different prompts | 100 to 118 ms | 1,182 tok/s | 1,243 at 106 | 339 at 99 |

Every batch's tokens equal the single-batch run's and each other's. The
cost model's prediction is within 6% on every aggregate, which passes
the 20% test the run set; the per-range spans (19 / 29 / 30 / 24 ms with
different prompts) show the two middle ranges setting the period as the
model said, with the first GPU idling more than modelled (35 to 45%).
Two things had to stop being shared between steps in flight: the direct
expert kernels' arrival counters (one buffer per GPU; three of four
batches drifted to other tokens until each step graph got its own), and
the cost measurement's placement of a real weight under sharding. With
the native MXFP8 body for the dense multiplications merged (33 rows: the
wide multiplication 49 to 33 µs, 92% of bandwidth), the same four-batch
runs read: 33 same-prompt rows 3,542 tok/s at 31 to 41 ms a batch
(predicted 3,394); 33 different prompts 1,223 tok/s at 103 to 110 ms
(predicted 1,243; ceiling 1,943); one row 258 tok/s at 13.9 to 14.2 ms
(predicted 259). Batches identical to each other; against the fixed
single-batch reference the first differing token moved from 26 to 28,
the native body's rounding. Those aggregates sum each batch's own
tokens over its own decode time. The measurement harness
(`dev/hero/batches.py`, the matrix `hero9-matrix.sh`) counts the
stricter way, all batches' tokens over the longest batch's decode
window, which is what a server sustains with these four batches: in its
run, one row 232 tok/s at 13.9 to 14.3 ms a batch (predicted 259), 33
same-prompt rows 2,433 tok/s at 30 to 53 ms (predicted 3,346; one batch
drifted to 53 ms where the others sat at 30 to 42), 33 different prompts
1,148 tok/s at 103 to 108 ms (predicted 1,243), with 89.5 distinct
experts a layer measured, which puts a single batch's floor at 51.9 ms
and the pipeline's ceiling at about 2,100 tok/s; the measured aggregate
is 54% of that ceiling. A repeat of the three pipeline lines minutes later
read 33 same-prompt rows 3,594 tok/s at 30 to 37 ms a batch (the drift
was a one-off), 33 different prompts 1,077 tok/s at 105 to 121 ms, one
row 234 tok/s at 13.9 to 14.3 ms; so the same-prompt case sits 7% above
its prediction and the other two 10 to 13% below, all inside the 20%
the run set. The harness's single-batch lines agree with hero 8's (one
row 13.4 ms, the drafter 25.4 ms at 2.37 tokens a step, 33 rows 30.1 /
99.9 ms); a third pod, built identical to the second, read one row at
13.3 ms (p10/p90 13/13, 1,316 kernels: the native fp8 body's 290
activation-recoding launches included) on its first step. The boundary copies landed: the capture now records every copy the
step makes between GPUs (`VXPU_STEP_COPIES=1` lists them), and at 33
rows two of the 31 were 173 of 180 MB a step, layer 20's compressed KV
cache and its index keys, copied whole from the third GPU to the fourth
every step as the shared-state replica refresh. The replica now takes
only the window of positions the step wrote, gathered on the owner and
scattered on the reader inside the graph, with a prefill's rows
refreshed whole once at admission; a replica-against-source check shows
no differing position. Crossing traffic fell from 180 to 7.3 MB a step
at 33 rows, the 24 copies from 3.27 ms to 0.14 ms, and the 33-row steps
from 35.0 / 41.8 / 108.5 to 31.8 / 30.0 / 98.5 ms (one prompt / same /
different), tokens identical. The shared expert's split landed as two
ops with the front as an island, 12.6 ms at one row with the front on
the parallel branch against 13.6 off, but the branch publishes or reads
that island's output out of order when its consumer is not the next
node (rows differ at nine rows), so the island is off by default until
the capture orders it; that fix is in progress, as is a CPU selftest
regression on segments with two outputs.

**Hero run 9, the table with everything merged (Oct 31, night; pod B,
the merged tree with the boundary copies, the shared expert's split with
its island off, the native fp8 body, the pipeline; measured by the
harness, sustained aggregate = all batches' tokens over the longest
batch's decode window):**

| case | per-batch latency | sustained aggregate | predicted | before hero 9 |
|---|---|---|---|---|
| 1 row | 13.3 to 13.6 ms | 73 tok/s | 73 | 13.6 ms |
| 1 row + drafter, 4 drafts | 25.7 ms, 2.56 tokens a step | about 100 tok/s | | 103 |
| 33 rows, one batch: one / same / different | 29.0 / 26.9 / 95.6 ms | 345 same, 209 different | 982 / 370 | 35.0 / 32.8 / 99.2 |
| 9 rows, one batch: one / same / different | 21.2 / 17.1 / 34.5 ms | 256 same, 170 different | 474 / 241 | 22.6 / 18.5 / 35.6 |
| 4 batches × 1 row | 13.6 to 14.3 ms | 246 tok/s | 259 | 75 (one batch) |
| 4 batches × 33 rows, same prompt | 27 to 37 ms | 3,575 to 3,776 tok/s | 3,346 | 1,021 (one batch) |
| 4 batches × 33 rows, different prompts | 94 to 98 ms | 1,414 to 1,436 tok/s | 1,243 | 339 (one batch) |

Every batch's tokens match the single-batch reference (after one fix:
the prefill's replica refresh went to batch 0's graph for every batch,
so batches 1 to 3 decoded on stale replicas for one matrix until the
refresh followed the state being prefilled). The different-prompt
pipeline runs at 66% of its ceiling of about 2,150 tok/s, set by the
slowest range's bytes; the planner predicted all three pipeline cases
within 13% before any of them ran. What did not land: the shared
expert's island on the parallel branch. The ordering fault was found and
fenced: the branch's events were right, but an island's kernels ran at
the same time as the routed experts' kernels on the host node's own
side stream, which the branch's events never covered, and one corrupts
the other through a buffer not yet identified. The capture now follows
one rule, held by a unit test: an island and a host node never run at
the same time (an island after a host node waits for it; a host node
waits for the branch). The attention islands keep their overlap (one row
13.3 ms); the shared expert's front gains nothing under the rule and
stays off by default. Finding the shared buffer is the way to its
millisecond; `VXPU_BRANCH_TRACE=1` prints the capture's order. Against the six-card recipe:
its 494 tok/s at 24 streams against our 1,414 at 132 rows in flight on
four cards with different prompts; single-stream 103.8 against our 73,
with the drafter 285 against our 100.

**Hero run 10's acceptance, agreed Oct 31 (night).** The planner picks
the serving shape. Hero 9 showed the engine runs several batches in
flight and that the cost model predicts it; hero 10 asks the planner to
choose, per load, among the shapes the engine now has, and proves the
choice with measurements at every load in the table (1, 9 and 33 rows;
one batch and four in flight):
- speculation inside the pipeline: the drafter with batches in flight,
  which hero 9 did not build; the planner predicts, per load, whether to
  draft and with how many tokens (a five-token verify reads many more
  distinct experts at 33 rows than at one), and the run measures the
  crossover;
- adaptive speculation: the draft length and the acceptance per step
  driven by the drafter's confidence, and drafts preferred or dropped by
  whether their experts are already being read in the step, each as a
  policy the planner scores against the cost model and the run measures;
- one wide batch against several in flight: 132 rows in one batch
  against four batches of 33, since rows in a batch share every expert
  read and batches do not; both predicted, both measured, bounded by the
  rows' cache memory;
- per-batch efficiency: a batch of 33 different prompts runs at 54% of
  its bytes with its kernels near their bytes, so the launch count
  (about 1,700 kernels a step at 33 rows: the remaining fork boundaries,
  the activation recoding folded into its producers, the ring write as
  an op) and the pacing by the two middle ranges are the items;
- self-testing built in: golden values per model checked at load; the
  reference implementation run beside the chosen one on a sample of
  steps, on by default in development; and a standing invariant that
  pads a batch with a copy of one row and asserts the two rows agree
  every step, which would have caught this week's counter race, strided
  copy, stale replica and branch fault on their first run.
What to prove: the planner's choice within about 20% of the best
measured configuration at every load; speculation and batching
composing; the consistency checks costing a few percent at most. Risk:
the batched drafter is new code on the hardest path; the checks go in
first. Setup: four identical disk-layout pods, two per node.

**Hero 10, the planner's side (Nov 1).** The planner picks the serving
shape from a load (rows in all, the prompt mix, a latency bound per
token, whether the artifact has the drafter): `vxpu plan serve` scores
every shape the engine has (one wide batch or one batch per range, no
drafter or 1 to 5 drafts under each accept policy) with the cost model
and writes the chosen plan, whose new `speculation` section names the
draft length and the policy (`fixed`, `confidence` with its threshold,
`experts` with its budget of new experts a layer and the candidates to
choose among); the engine reads the section through `physplan` and
applies it with the pure functions in `vxpu/speculation.py`
(`draft_length`, `plan_drafts`, `accept`), whose `Stats` carry the
per-position acceptance and confidence into the record. The speculative
step's model (`pkg/planner/speculation.go`): a step with k drafts a row
verifies k+1 token rows, reads the dense weights once and the experts
once per distinct expert the token rows select, pays 0.35 ms a token row
and 3.5 ms for the drafter's graphs on the last range, and yields
1 + α + ... + α^k tokens at a per-position acceptance α of 0.68 (fitted
to the measured 2.67 tokens a step at four drafts). The distinct experts
a layer: across rows the measured curve (6 at one row or one prompt,
37.3 at 9 different prompts, 90.3 at 33, log-linear between and extended
past 33 as an estimate); along a row's verify a token curve measured on
Nov 1 (pod hero-d, the specm5 artifacts, VXPU_EXPERT_STATS=1): the five
token rows of one row read 18.3 distinct experts a layer, not the 30 of
independent draws, and eight different prompts at five tokens read 83.2,
not 8 × 18.3; so consecutive tokens share more experts than rows of
different prompts do, and the growth saturates where the rows already
read most of the layer. So calibrated, the model is within 8% of the
five measured speculative steps (one row at 2, 4 and 5 drafts; eight
rows at 4 drafts, one prompt in 40.4 ms and a prompt per row in 95.1).
What it predicts, and the run measures (`dev/hero/predictions.json`, 88
cases; "fixed" is the engine's policy today, the adaptive ones rest on
estimated parameters until the engine records the confidence histogram):
at one row four drafts yield 104 tok/s (measured 103) and the confidence
policy at five drafts 108; four batches of one row with four drafts 271
against 259 plain; eight rows of one prompt 506 with four drafts
(measured 531) against 446 plain; four batches of eight rows of
different prompts 835 with two drafts against 782 plain; at 33 rows a
batch the token rows' own cost nearly cancels the tokens they yield, and
the model calls two drafts and none within about 10% of each other
(four batches of 33: 3,394 plain against 3,402 at two drafts with one
prompt in every row; 1,329 against 1,589 with a prompt per row), which
is inside its error: that crossover is the run's to measure, not the
model's to call. One wide batch of 132 different prompts 580 against
1,329 for four of 33, because one batch streams one GPU at a time.
**Hero run 10, one wide batch against several in flight: the prediction
(Oct 31, night, written before the artifacts were measured).** The cost
model calibrated on hero 8 already describes one wide batch: it is the
single-batch case with more rows. Its bytes are the dense weights (8.5 GB)
plus the distinct experts a layer times 18.8 MB times 40 layers; its time
adds 4.1 ms fixed and 0.61 ms a row, the per-row term calibrated between
1 and 33 rows and extrapolated here (the largest uncertainty in the
table). One batch streams one GPU at a time, so its step is the sum of
the four ranges; batches in flight stream all four, so their period is
the slowest range. The prompts matter: the harness cycles the 33-prompt
list, so a 132-row batch holds each prompt four times, exactly as four
batches of 33 did in hero 9 (each prompt four times across the batches).
Rows with the same prompt decode the same tokens and select the same
experts, so the 132-row batch reads the experts of 33 distinct rows:
about 100 a layer, measured at 33 rows. For truly different prompts
(prompts-132.txt, written for this run) the distinct count is estimated
from a power law through the measured 9-row (37.3) and 33-row (100)
counts, exponent 0.76, capped at 384; the run replaces the estimate.

| shape | prompts | distinct experts a layer | step, or per-batch latency | aggregate tok/s |
|---|---|---|---|---|
| 4 × 33 in flight (hero 9) | the 33-prompt list in every batch | 100 measured | 106 ms predicted, 94 to 98 measured | 1,243 predicted, 1,414 measured |
| 1 × 33 (hero 9) | the list once | 100 measured | 89 predicted, 96 measured | 370 predicted, 345 measured |
| 1 × 66 | the list twice | 100 | 109 | 603 |
| 1 × 66 | 66 different prompts | 169 estimated | 150 | 439 |
| 1 × 99 | the list three times | 100 | 130 | 764 |
| 1 × 99 | 99 different prompts | 230 estimated | 207 | 479 |
| 1 × 132 | the list four times | 100 | 150 | 882 |
| 1 × 132 | 132 different prompts | 287 estimated | 260 | 507 |
| 2 × 66 in flight | the list twice in each batch | 100 | 109 | 1,207 |
| 4 × 66 in flight | the list twice in each batch | 100 | 130 | 2,025 |

The prediction, in one sentence: at 132 rows the wide batch loses to
four batches in flight, 882 tok/s against 1,243 predicted (1,414
measured), with a 150 ms step against 94 to 98 ms, and the loss grows
when the prompts really differ (507 tok/s). The reason is not bytes: the
cycled 132-row batch reads the same 83 GB a step as a 33-row batch, so
its bytes cost 65 ms, the same as the pipeline's four batches put
together, and it would match the pipeline's throughput if the rest of
the step cost no more than 0.28 ms a row. The model says it costs 0.61,
so the wide batch spends 85 ms a step in work that does not scale with
bytes while the pipeline spreads the same work over four GPUs. What the
measurement decides is that per-row term at 66 to 132 rows; if it falls
toward 0.3 ms the shapes tie and the planner's choice comes down to the
latency budget. Memory: the 12-layer ranges (layers 8 to 20 and 20 to
32) hold 89 GB of weights on a 96 GB card, so they are where 132 rows'
caches would run out first; the run records each GPU's memory at every
width. The planner rule this decides between is in `vxpu plan pipeline`:
given the rows to serve and a latency budget, it lays the rows out as
one, two, three or four batches, predicts each, drops the shapes over
the budget and picks the highest aggregate.

**Hero run 10, one wide batch against several in flight: measured (Nov 1,
03:33 to 04:25 UTC, pod C, the vxpu-big tree at 6430bba, one process at
a time under the pod's lock, 64 new tokens, medians over the steps).**
The wide batch lost, and by more than the model said: 132 rows in one
batch step in 297 ms (444 tok/s) against 95 to 99 ms a batch step for
four batches of 33 in flight (1,414 tok/s), a 3.2× gap where the model
predicted 1.4× (150 ms against 106). The reason is not the shape. Above
64 rows the kernel library runs out: every one-pass Triton kernel and
every fused body applies to at most 64 rows (`VXPU_GEMV_ROWS_MAX`), so
at 66 rows the lowering hands the fp8 linears to a bf16 copy, the
attention projections, the KV window and the shared expert's front to
their reference bodies and the head to an f32 copy, and the step
launches 14,686 kernels in place of 1,977. The jump is a cliff, not a
slope: 95 ms at 33 rows, 209 at 66, then 0.64 ms a row from 66 to 99
(the calibrated 0.61). Raising the cap to 132 rows does not help: the
one-pass kernels tile rows to a power of two, and at 128 and 256 rows
their tiles need 140 to 164 KB of shared memory against the 101 KB the
compiler allows, so every width above 64 failed to compile. The missing
piece is a GEMM template (rows tiled, the weight streamed once per row
tile) that the planner would choose above the GEMV's handover row count;
until it exists the planner must know the library's row limit and not
plan a batch wider than it.

| shape | prompts | distinct experts a layer (measured) | step, median (p10/p90) | tok/s, rows over the step | model at the measured count | peak GPU memory | tokens against the 33-row reference |
|---|---|---|---|---|---|---|---|
| 1 × 33 | the list once | 89.5 | 95.3 ms (77/103) | 346 | 83 ms, 397 | 88.8 GB | the reference |
| 1 × 66 | the list twice | 91.5 | 209.2 (191/218) | 315 | 104 ms, 632 | 89.9 | 4 of 66 rows match; the rest differ from step 2 to 53 (median 23) |
| 1 × 66 | 66 different | 91.0 | 212.9 (164/227) | 310 | 104 ms, 634 | 89.9 | the reference for the per-row runs |
| 1 × 99 | the list three times | 89.4 | 230.4 (202/242) | 430 | 123 ms, 803 | 90.6 | 18 of 99 match |
| 1 × 99 | 99 different | 111.4 | 240.4 (188/260) | 412 | 136 ms, 726 | 90.6 | 7 of the 66 shared prompts match |
| 1 × 132 | the list four times | 144.0 | 297.0 (237/317) | 444 | 176 ms, 751 | 90.8 | 12 of 132 match |
| 1 × 132 | 132 different | 126.7 | 286.3 (224/309) | 461 | 166 ms, 798 | 90.8 | 12 of 99 shared match |
| 4 × 33 in flight | the list in every batch | 89.4 | 95.4 to 98.6 a batch | 1,414 (the harness's strict aggregate) | 99 ms, 1,337 | 92.3 | 33 of 33 in every batch |
| 2 × 66 and 4 × 66 in flight | the list twice a batch | | out of memory | | 104 ms, 1,265 | 97.2 on GPU 1 | |

Memory: the 132-row batch fits with room (90.8 GB of 97.9 on the
fullest GPU, the 8-to-20 range) and takes less than four batches of 33
(92.3 GB), since one batch holds one set of step graphs and buffers.
Two batches of 66 do not fit: the second batch's step graph on GPU 1
ran out at 97.2 GB while being captured, so batches in flight pay about
1.2 GB a batch of 33 and more a batch of 66 on the fullest GPU. The
distinct-expert counts at 66 and 99 rows with the list cycled (91.5,
89.4) match the 33-row count, as rows with the same prompt should; the
132-row count of 144 does not, and the 132 different prompts selected
fewer (126.7) than the 99 did at 111.4 would suggest, so the counter at
that width is in question (open). The tokens above 64 rows differ from
the 33-row reference in most rows, from step 2 in some: the reference
bodies the width falls back to round differently from the kernels. The
rows do agree with each other: the duplicate-row invariant, run as its
own check after the table (the list four times in the 132-row batch,
twice in the 66-row batch), found every prompt's copies decoding the
same 64 tokens (33 of 33 prompts at both widths; 132 rows at 251 ms a
step in that run). So the wide steps are slow and differently rounded,
not wrong. The model, given the measured counts, predicts the wide
steps at 104 to 176 ms; measured 209 to 297, because it does not know
the library's row limit. What the
planner takes from this: a batch wider than the kernels' row limit is
not a candidate, and the shape choice (vxpu plan pipeline --batches 0)
now marks such widths and never picks them (--max-batch-rows lifts the
limit for the day a GEMM template serves them); the memory check should
count each batch's graphs on the fullest GPU (open). The pipeline
line reproduced hero 9 exactly (1,414 tok/s then and now).
**Hero run 10, first finding (Nov 1): the pipeline's batch 0 leaves the
engine's own graph.** The batched-drafter work found four 9-row batches
of different prompts not batch-identical on the base tree. The
reproduction, six samples per case on pod B: base 4 of 6 wrong, always
batches 0 and 1 and never 2 or 3; two batches alone 3 of 6; the wrong
tokens are plausible alternatives (coherent text), so a small numeric
perturbation, not garbage. Ruled out: the replica windows (off: 4 of 5
wrong), the parallel branch (off: 3 of 6), the islands' overlap with the
node before them (made to wait: 3 of 6), the prefill (every prompt
admitted before any step: 1 of 6), nondeterminism of one batch (two runs
token-identical), memory reuse of anything the warm-up graph captured
(an allocation-history probe found none), the engine's single-batch runs
before the pipeline (none: 2 of 6). Two facts localize it: batch 0
on a graph captured by `_pipeline_graph` like the others, instead of the
graph the warm-up captured on the resident state, is clean 6 of 6; and
the plan must have both attention islands as separate segments (attn_q
alone, window_kv alone and no islands are clean, 0 of 6, 0 of 6, 0 of
4). The fix in place is the first fact made the default: every batch in
flight gets a state and a graph of its own, and the warm-up graph serves
single batches only (one more state of 0.17 GB and one more capture at
the first pipeline call). The cause is not named yet. The two suspects:
something the warm-up graph holds is not private to it (its runner is
the engine's, the split it was captured through is re-pointed by every
admission, and the engine's own loop and prefill drove it before the
pipeline did), or the island plan's view-only segment between the two
islands, whose output is a reinterpretation of its input. A second graph
captured on the resident state through `_pipeline_graph`, run as batch
0, is clean 6 of 6: the resident state is not the carrier, the graph the
warm-up captured is. Next: diff what `_warm_up`'s capture sees against
`_pipeline_graph`'s (the runner, its placement, the counters' scope, the
per-segment graphs the pipeline's eager step captures), and the drafter
side if the warm-up's draft graph is affected the same way.

**Hero run 10, closed (Nov 1, 21:40 UTC).** The run asked whether the
planner can pick the serving shape. The answer: it can pick between the
shapes the engine has today, and the run's largest result is that the
engine's kernel library, not the planner, bounds the shapes worth
picking. Six workstreams ran in parallel on four identical pods (two per
node), each change on its own branch with its own measurement, merged
into vxpu-big through 17f5c4d. Against the acceptance list:

| item | result |
|---|---|
| speculation inside the pipeline | works: each batch in flight drafts and verifies on its own graph, acceptance per batch equal to a single batch, tokens identical across batches. Four 1-row batches 250 to 288 tok/s with the drafter against 243 plain; four 8-row batches of one prompt 1,518 against 1,455; with a prompt per row the drafter loses from eight rows up (622 against 781 at four batches of eight; 2× slower at 33), because verifying five positions a row reads many more distinct experts; four 33-row batches with the drafter do not fit in memory |
| adaptive speculation | modelled and plumbed, not yet measured in the engine: the plan file's `speculation` section carries the draft length and a policy (fixed, confidence, expert-aware), `vxpu/speculation.py` applies it, the cost model predicts the five measured speculative steps within 8%; the adaptive policies' parameters are estimates until the engine records the confidence histogram |
| one wide batch against batches in flight | batches in flight win, 1,414 to 1,462 tok/s for four batches of 33 against 444 for one batch of 132 (297 ms a step). The reason is a cliff in the kernel library at 64 rows (the one-pass kernels and fused bodies hand over to reference bodies: 14,686 launches a step in place of 1,977), not the shape; the planner now refuses to plan a batch wider than the library's row limit, and the missing piece is a GEMM template for wide rows |
| per-batch efficiency, the launch count | launches fell by a quarter at one row (1,372 to 1,045 kernels, 35 copy nodes to 12) and the step did not move: 13.3 ms before and after. The GEMV prologue recode removes 284 launches but lengthens every GEMV (1.1 ms worse at nine rows), so it is off by default and a per-shape choice for the planner; the shared-expert join is off until its epilogue and the split pair agree bit for bit; the ring write (a wave's copies and flag as one kernel) is on and neutral |
| self-tests built in | three standing checks, on by default in development: golden tokens at load (1.3 to 2.6 s), a duplicate row padded into every batch and compared every step, and the reference implementation run beside the chosen body on a sample of steps (430 ms a sample, under 2% of decode). Their cost on the step is zero at one and nine rows. The duplicate-row check passed on the wide steps; the reference check puts every op within 0.4% of its reference except attn_select at 3.3% at nine rows |
| the planner predicts | pipeline and speculation models within 8% on the measured cases (the wide batch excepted, where the model did not know the library's row limit; it does now) |

Found and fixed on the way: four 9-row batches of different prompts
were not batch-identical on the base tree (wrong in four of six runs,
always batches 0 and 1, plausible alternative tokens); the defect needs
the engine's warm-up step graph running beside another batch's graph
with both attention islands present, and every batch in flight now gets
a state and graph of its own (clean in six of six, step times
unchanged; the cause inside the warm-up graph is still open). Found and
filed: the n-gram tables' population reads a cold disk one page at a
time, 3 MB/s on a disk that reads at 1.2 GB/s (#504). Still open from
this run: the GEMM template for wide rows; the producer-side MXFP8
recode (codes written once by the producer, which the measurement says
is the form that pays); the drafter's memory at four batches of 33; the
hero harness's one-row reference (`alone-tokens.json`) is stale at
position 26 and the artifacts were re-exported, so the token checks in
this run are against same-tree controls; a per-batch latency spread in
the 1-row drafter pipeline (27 against 35 ms). What the one-row step
says about where the ceiling is: 13.3 ms against a byte floor of 8.9,
with the GPUs busy for 12.7 to 13.4 ms of it, so the remaining gap is
inside the kernels (their bandwidth efficiency), not in launches,
crossings or host work.

**Hero run 11's acceptance, agreed Nov 1 (afternoon): every kernel
against its own floor, then the planner tunes the templates.** Hero 10
left the one-row step at 13.3 ms with the GPUs busy for 12.7 of them,
against a byte floor of 8.9: the remaining gap is inside the kernels.
The profile of the control run puts it in three places: the dense GEMV
templates (gemv_rows, gemv_pe, the dual SwiGLU GEMV: 6.5 ms over 286
launches, most of the 8.5 GB of dense weights, efficiency unknown per
site because the record carries microseconds and not bytes), the fp4
expert kernels (3.8 ms over 78 launches for about 4.5 GB, about 82% of
bandwidth), and about 1,000 kernels under 5 µs (fronts, stats, post,
attention pieces: about 1.9 ms that bytes do not explain). The run:
1. Bytes per site in the kernel record: for every call site the weight
   bytes and the activation bytes it must move, derived from the binding's
   shapes, next to the measured microseconds; `vxpu kernels check` prints
   efficiency per site and per template (sorted worst first) and the step's
   byte floor; the record's stats carry the per-template totals. This is
   the roofline ledger, and it is model-agnostic: any export's record gets
   it.
2. The measuring step: `vxpu kernels measure` enumerates a template's knob
   sets (block rows, K step, warps, split-K, the experts' direct-path pair
   limit) per shape class on the card it runs on, times each, and writes
   them into the table with their bytes; the planner picks per site from
   the table; a site under a threshold of its floor without a recorded
   reason fails the check. The sweep is the system's job (selection), and
   the table is checked in per card type.
3. Where the ledger says a site is bound by structure (a long-K GEMV that
   cannot fill the card at one row), split-K is the one new template.
4. The latency-bound tail: the sub-5 µs kernels per layer folded by
   template where the fold does not lengthen the survivor, counted before
   and after.
Acceptance: one row at or under 11.5 ms (hero 8's bar), tokens identical
to the same tree's control; the GEMV and expert templates at 85% of
bandwidth or better in the ledger at one row; nine rows and 33 rows no
worse; the ledger and the measuring step land in the repository with
unit tests that run without a GPU (synthetic record and table). Four
workstreams in parallel from the start on the four pods, each change
with its own measurement, hard interim deadlines, results read from the
pods' files rather than from watchers.

**Hero 11, the roofline ledger (Nov 1).** The kernel record now carries,
for every call site, the bytes the site must move, read off the graph
alone (`lower.site_bytes`): the parameters it reads (a `get_attr`, or the
export's lifted `p_*` and `c_*` placeholders, also behind a view), the
activations it reads and writes, and the state buffers it names (`b_*`,
the caches, of which an op reads a window, so kept apart as an upper
bound). The hardware section carries the card's bandwidth as the engine
measured it at load (`plan.gpu_bandwidth`: a 1 GiB device copy, 1,462
GB/s on the RTX PRO 6000, and a 1 GiB single-tensor read, which the
floors use when present, since the step reads weights and writes little:
the head's GEMV measured 1,530 GB/s, above the copy figure), so the
floors are over the card's own numbers. `vxpu kernels check --record R` prints the ledger
(`pkg/kernelplan: Ledger`): every site against its floor (weights plus
activations over the bandwidth), worst first by lost microseconds, the
per-op totals, and the sites' summed floor against their summed time;
`--step-times` takes the step's own per-site microseconds in place of
the lowering's microbenchmark: `dev/hero/ledger.py` joins a kernel
profile to the sites one layer at a time (the experts' first kernel marks
each layer; within a layer the tail's sites, the shared expert and its
combine, claim their launches first and the next layer's head after
them, each site claiming the launches its body declares), and checks that
every (op, kernel family) received exactly the launches the sites
declare, so an off-by-one cannot pass silently (the first run of it found
the router launching 39 times where 40 sites declare it); `--threshold 0.85` fails a
measured site under 85% of its floor unless the table's entry records an
`accepted` reason (a long-K GEMV that cannot fill the card at one row).
The table's entries carry the same byte fields and the chosen body's
microseconds, for the measuring step to fill. None of it knows the model:
any export's record gets the same ledger, which is what makes it the
place where "the gap is inside the kernels" becomes a ranked list per
template and per site. The first ledger of the one-row step (Nov 1,
pod D, 1,455 kernels, 13.0 ms of GPU time): the sites' floors sum to
6.0 ms and the experts' to 3.1 (top-k times 40 layers times 18.8 MB), so
the step is at 70% of its floor; by op from the step's own time, attn_q
75% (the small wq_a GEMV and its two helper kernels; wq_b itself 98%),
the shared expert 68% (its dual SwiGLU GEMV 73%), attn_out 90%, the
head 105% (the copy figure understates reads), the experts 80%, and the
small ops under 30%: window_kv (3.7 MB in 11.8 µs), the router, the
attention select and the hyper-connection front, 1.36 ms a step between
them, latency-bound. By family, gemv_rows 86% over 211 launches, gemv_pe
98%, the dual GEMV 73%.

The ledger's first catch was not a slow kernel but a flipped choice. The
self-tests' golden check refused the merged tree's load of the one-row
artifact (prompt 0 differed at position 26 from the goldens written on
6430bba). Checks off and checks on decoded the same tokens with the same
bodies, so the checks were not the cause; the records were: at all 40
attn_out sites the lowering's measured choice had flipped. On 6430bba it
timed the fused body at 58.68 µs and the Triton kernel at 57.64 and ran
the kernel; on the merged tree 58.49 against 57.60, and with the one
launch credited to the fused body that is inside 1 µs on a 58 µs pair,
so run-to-run noise decided it. The fused body is two fp8 GEMVs with a
quant between them, 83 more kernels a step (1,455 against 1,372), and it
rounds differently, hence the token. The rule this fixes in place: a
measured choice that sits inside the noise band is not the run's to make;
it goes to the planner, which chooses once from the records and writes
the table, and the engine applies the table (VXPU_PLAN_TABLE) and writes
goldens against it. The rest of hero 11 ran on such a table
(`table-h11.json`, planned from both records without the launch credit:
attn_out on the kernel, what 6430bba ran, checked against its record at
all 347 sites), and the lowering's own tie-break grew a margin (the gemv
workstream's change).
**Hero run 11, the GEMV templates against their floors (Nov 1, measured
on pod A).** The record now names every GEMV call's shape class (the
template, the rows, the weight's shape and dtype, the prologue and
epilogue flags: `triton_ops.gemv_class`), with the blocks that ran and
the call sites a step launches (`stats.gemv_classes`); `python -m
vxpu.measure classes` gives each class its bytes and its floor, and
`vxpu kernels measure` sweeps the blocks a class can run (weight rows a
program, the K step, the K split where the caller takes partial sums,
warps, then the pipeline depth on the winner) through a micro-benchmark
that replays the kernel over weights cycled past the L2 inside a CUDA
graph, and writes the fastest per class into the table's `blocks`
entries, which the dispatchers apply (`gemv_blocks`) and `vxpu kernels
check` verifies. The verdict first: the dense GEMV templates were
already at 87% of the byte floor as a whole, the sweep found the
engine's hand-set tiles best or within 2% in every class but three, and
the step moved by 0.2 ms at one row and 1.4 ms at nine, with tokens
identical to the control. The one-row ledger, per layer (the 42 MB
classes read 1,460 GB/s as the floor):

| class (one row) | bytes a launch | floor µs | measured µs, engine's tiles | efficiency | the sweep's best | gain |
|---|---|---|---|---|---|---|
| attn_q's up-projection wq_b, rotary epilogue (gemv_pe) | 42.1 MB | 28.8 | 29.9 | 96% | the same | 0 |
| attn_out's wo_b, hc epilogue | 42.1 MB | 28.9 | 31.1 | 93% | the same, 4 stages | 0 |
| attn_out's wo_a, grouped, quantizing epilogue | 33.6 MB | 23.1 | 24.7 | 94% | 4 stages: 24.5 | 1% |
| the shared expert's dual SwiGLU GEMV | 23.6 MB | 16.2 | 21.2 | 76% | 2 warps, K split 8, 4 stages: 17.4 | 18% |
| the shared expert's w2 | 11.8 MB | 8.1 | 10.6 | 76% | 2 warps: 10.4 | 2% |
| attn_q's down-projection wq_a (1,280 rows, K split 4) | 6.6 MB | 4.5 | 7.5 | 60% | 2 warps: 7.3 | 3% |
| the engram layers' 4,096-row linear | 5.3 MB | 3.6 | 8.1 | 44% | 2 warps: 6.1 | 25% |
| the KV window's wkv (512 rows, K split 4) | 2.6 MB | 1.8 | 3.4 | 53% | 2 warps: 3.2 | 6% |
| the head (bf16, fp32 out) | 1,324 MB | 907 | 863 | 105% (reads run above the copy bandwidth) | the same | 0 |
| the engram layers' 25,600-row linear | 157.5 MB | 107.8 | 105.9 | 102% | the same | 0 |

What the ledger says, in plain words: the big weights stream at 93 to
105% of the card's bandwidth already; the loss is in the small ones (3
to 11 µs a launch, under 12 MB), which cannot cover the card: 512 or
1,280 rows give 32 or 80 programs of 16 rows for 188 SMs, and the K
split (already 4) buys nothing past that (8 and 16 measured slower)
because each split's program then does a handful of K steps and the
launch's ramp dominates. Two warps instead of four helped exactly
those, the dual GEMV most (72 programs before its split). Where the
structure binds, the fix is not a tile: the two narrow GEMVs at the
attention's front read the same activations and would be one launch,
and the dual GEMV's split epilogue could join its w2 consumer; those are
fusions across ops, the tail workstream's and the next run's. The
sweep itself: 10 decode classes a row count, 34 to 254 candidates a
class, about two minutes a sweep on one GPU; the table for this card
is checked in (`pkg/kernelplan/testdata/table-rtx6000.json`, 26 block
classes over the ledger's 93 site entries). Measured steps, both arms on
the ledger's site table (triton at attn_out), the sweep's blocks the
only difference: one row 13.2 → 13.0 ms (1,383 kernels both; the
profile's GPU time 12.77 ms); nine rows of one prompt 22.3 → 20.9, of
the same prompt 17.0 → 16.7, of different prompts 34.2 → 33.9; tokens
identical at one row and in all nine rows. The nine-row profiles say
how much of that is the blocks: the step's GPU time went 20.65 → 20.46
ms (the dual GEMV 0.94 → 0.77, gemv_rows 4.70 → 4.66, nothing else
moved), so the blocks bought 0.2 ms a step at nine rows as at one; the
rest of the 1.4 ms between the one-prompt arms is the step's gaps
outside the kernels, which that case has shown at ±1 ms between runs
all day (21.1, 22.0, 22.3, 20.9 across four loads) and which no table
sets. Found on the way and fixed:
the lowering's per-run measurement flipped attn_out between its fused
body and the kernel on 1% of noise (58.5 against 57.6 µs), moving the
tokens at position 26 between two loads of the same tree; a fused body
now has to beat the kernel by max(3%, 2 µs) (`costs.decide`), and the
record says by how much.
**Hero run 11, the fp4 expert kernels against their floor (Nov 1,
evening, pod B).** The floor first. The card reads at 1,535 GB/s (a 1 GiB
stream, plain loads or TMA boxes of 32 by 128 bytes alike; the 1,460 used
before is a copy's read plus write). A stream of the gate/up kernel's own
71 MB at its own program count (864 programs of 80 KB) reads at 1,440 to
1,470 GB/s: ramp and tail cost about 2.5 µs a launch at this size, so the
80 expert launches of a one-row step pay about 0.2 ms for being 80. The
achievable floor a layer is therefore about 51.5 µs for gate/up (75.2 MB)
and 28.5 for down (37.6 MB); the kernels stood at 60.4 and 32.9 in
isolation, 63.6 and 33.1 inside the step. What was measured (synthetic
banks of the real shapes rotating over 96 experts so the 128 MB L2 never
serves a repeat; six pairs; 108 tile, warp and stage sets per kernel): the
sweep alone finds the old tiles best, because gate/up loaded x as a
[16, BK] tile with one live row and the pipeline staged that tile per stage
(8 KB a stage, twice the weight tile), holding the resident programs to
four an SM; with x loaded as one row and broadcast in registers a third
stage wins, 57.6 µs (1,305 GB/s). The descriptor made on the host once per
bank instead of per program: about 1 µs. Down: nothing faster than its
tiles, 32.2 µs with the host descriptor. Measured and not taken: a
tile-major bank layout (each box one contiguous 4 KB) 1.3 µs more; two K
tiles an iteration, gate and up in one program (bit-identical, with the
SwiGLU, routing weight and per-32 quantization in its epilogue so that down
streams h; down then 31.8 but gate/up 70 to 82), narrow K tiles with deep
pipelines: all slower; the TMA box shape on a pure stream: no effect. The
tiles are eight table knobs (FP4_GU_BN/BK/WARPS/STAGES, FP4_DN_*) and the
sweep's choice is a kernel table with blocks per shape class
(`dev/hero/fp4-blocks-rtx6000.json`). In the step, both arms under the
pinned table `table-h11.json` in one session: gate/up 63.6 to 60.5 µs a
layer and down 33.1 to 32.6, the step's GPU time 13.01 to 12.89 ms, the
one-row step 13.2 to 13.1 ms, tokens identical at one, nine and 33 rows;
nine rows 21.2/17.0/34.3 to 21.1/17.0/34.3 and 33 rows 29.0/26.8/93.7 to
29.0/26.8/91.8 (the grouped path, untouched). Against the acceptance:
gate/up at 85% of the read peak and 89% of the achievable 71 MB stream,
down at 75% and 81%; 90% of the peak is not reachable at two launches a
layer of this size, and the next step is one launch a layer (gate/up and
down phases in one persistent kernel with a per-pair arrival count), which
is worth the ramp and tail of 40 launches, about 0.1 ms a step, and the
overlap of the two phases' tails.
**Hero run 11, the tail (Nov 1, night): the floor of a launch, and what
folding the small kernels costs.** The one-row step launches 876 kernels
under 5 µs (fronts, row finishes, the MXFP8 recode, the gate's and
indexer's pieces), 1.48 ms of the 12.7 ms of GPU time. Their floor on an
RTX PRO 6000, measured inside one CUDA graph (`dev/hero/launch_floor.py`):
0.75 µs per dependent one-element kernel, 0.69 per memset node or fill
kernel, 1.47 per one-program kernel over a 32 KB row (4.8 for three in a
chain), 0.97 per 32 KB copy. So the tail sits at its floor already (0.66
to 1.3 ms for 876 nodes, whatever they compute) and only fewer nodes
helps. Two folds were built and measured, each against the split it
replaces, bitwise-identical outputs in both:
- The front's statistics kernel folded into its dots kernel, every
  program computing the row's two norm statistics itself (the row's 32 KB
  of stream from L2 per program, 192 to 384 programs a site). Slower in
  every configuration of a 3 x 2 x 3 sweep (how the row is read, warps,
  experts per program): hc_front 5.4 µs against the split's 4.3 at one
  row, ffn_front 9.1 against 7.7; at nine rows 12 to 16 against 7.7 and
  19.1 against 16.6. Reading the row again in every program costs more
  than the 0.75 to 1.5 µs launch it removes. Dropped.
- window_kv's row finish (the sum of its split-K partials, the norm, the
  rotary, the quantization: `_row_post_kernel`, 39 launches a step)
  folded into its GEMV: every program adds one to an arrival counter
  after its store and the last to arrive finishes every row
  (`FOLD_POST`, off by default). The counter must be zero at every step,
  and no kernel may own a global for it (two batches in flight on one GPU
  replay the same package, as the expert kernels' counters showed in hero
  run 9), so the lowering now gives a segment one zeros (an intermediate
  of the compiled segment, so every captured graph owns its own), sliced
  per site that declares `Impl.counters`, and traced as one more input of
  the inlined body (`lower._Scratch`, tested without a GPU). Measured
  eagerly on the real shapes (x [rows, 4096] to kv [rows, 576], split-K
  16): at one row 7.1 µs against 7.2 for the pair (the fill launch
  included; without it, 0.8 µs saved), at nine rows 22.8 against 8.2 and
  at 33 rows 62 against 12, because the one finishing program does
  serially what the separate kernel's programs did in parallel. A variant
  where every program waits at the counter and then finishes every P-th
  row ran 8.0 and 9.4 µs at one and nine rows (the wait costs what the
  launch saved) and hung at 33 rows: the GEMV's 576 programs are not all
  resident at that row count, and a program waiting for one not yet
  scheduled never finishes. In the lowered step (pod C, the pinned kernel table,
VXPU_FOLD_POST=1 against 0 on the same tree): one row 13.2 ms against
13.2, nine rows 21.1 / 17.0 / 34.2 against 21.2 / 17.0 / 34.3 (one
prompt, nine of one prompt, nine prompts), tokens identical; and 47 more
kernels a step, not 39 fewer, because the compiler fused the segment's
zeros into a fill-and-slice kernel per site (40 of 1.1 µs) instead of
keeping one fill per segment, so the fold's first use pays a node to
save a node. Passing the scratch whole with an offset, not a slice,
takes that back; it is the next edit to the mechanism.
What this says for the tail: a finish that follows a split reduction
cannot join its producer without either serializing the rows or a
grid-wide wait, and neither pays on this card; the per-segment counters
are the right mechanism for the folds that do pay (a finish of one row's
worth of work, or one fill shared by several finishes in a segment), and
they stay in the tree for that. The remaining lever on the 1.48 ms is
fewer sites, not fewer kernels per site: the MXFP8 recode written once by
its producer (284 launches; the GEMV workstream's), and the fork's
per-layer glue the compiler emits around the gate and the indexer (197
launches, 0.49 ms; the indexer's top-k is #484).

**Hero run 11, closed (Nov 1, 23:30 UTC).** The acceptance's 11.5 ms is
not met: one row 13.2 → 12.9 ms, measured in one session on pod D with
both arms on the pinned site table (nine rows 21.1 / 16.9 / 34.1 →
20.8 / 16.7 / 34.0; 33 rows 28.9 / 26.7 / 93.5 → 28.9 / 26.6 / 93.5).
What the run bought is the instrument, and the instrument says why the
bar was out of reach tonight.

| item | result |
|---|---|
| the roofline ledger | every call site carries its bytes, floor and time in the record; `vxpu kernels check` ranks sites and templates against their floors and fails sites under a threshold without a recorded reason; a profile-to-site join checks that every op received the launches it declared (and found the router launching 39 times for 40 sites). One row: floors sum to 9.1 ms against 13.0 ms of GPU time, 70% |
| the measuring step | `vxpu kernels measure` sweeps a template's knobs per shape class on the card and writes the table; the GEMV table for the RTX PRO 6000 (26 classes) is checked in; the expert kernels' sweep is in the same format |
| the GEMV templates | already at 87% of the floor overall; the engine's hand-set tiles were the best of 34 to 254 candidates in every class but three; the dual SwiGLU GEMV went from 76% to 93% (two warps, K split 8, four stages), worth 0.2 ms a step at one and nine rows. The four narrow GEMVs stay at 53 to 62% by structure (32 to 80 programs on 188 SMs): a fusion across the ops that share an input, not a tile |
| the fp4 expert kernels | 82% → gate/up 85% and down 80% of a read peak measured at 1,535 GB/s (the 1,460 copy figure undercounts reads by 5%); the win was loading the activation row once instead of staging a masked tile per pipeline stage; 108 tile sets found the defaults best otherwise; 90% needs one launch per layer (two launches cost 2.5 µs each in ramp and tail) |
| the small-kernel tail | at the launch floor: 0.75 µs per dependent kernel in a graph, 1.47 if it touches a row, so 876 kernels cost 0.66 to 1.3 ms and take 1.48; the two folds that remove nodes do not pay (the stats-into-dots fold is bitwise identical and slower everywhere; the last-program-finishes fold ties at one row and loses from nine, the finish being serial over rows); the per-segment counter mechanism is kept, off |
| the noise band | the lowering's per-run measured choice flipped attn_out's body on 1 µs of noise between two loads and changed the one-row tokens at position 26 (83 more kernels, a different rounding); the golden check caught it on the first merged load; the choice now needs a margin of max(3%, 2 µs), and hero 11's runs were pinned to one table |

Tokens: identical between the arms at 33 rows; at one and nine rows
the merged tree moves the same near-tie token at position 26 that the
attn_out flip moved. The site tables agree at every site and the swept
blocks are not the cause (the merged tree without them gives the same
tokens and the same 1,383 kernels against the base's 1,375), so the
change is the GEMV workstream's narrow K-split dispatch, a rounding-level
change the reference check puts inside its class; the golden files are to
be rewritten against the merged tree with that reason recorded, and
position 26 of the first prompt is a known coin-flip token that any
rounding change moves. Where the one-row step's remaining 3.5 ms is, from
the ledger (measured): experts 0.77 (two launches a layer), the latency
tail about 1.4 (fronts 1.07, attention select 0.48, recode 0.31, compiler
glue 0.36), the four narrow GEMVs 0.5, attn_q and attn_out 0.5 at 75 and
90%. The next steps follow from the table, in order of size: the recode
written once by its producer and the fork's per-layer glue (fewer sites
in the tail), one expert launch per layer, the fusion of the x-sharing
narrow GEMVs. Process: the four workstreams ran in parallel with deadline
reports and results read from the pods' files, and the run closed in
seven hours.

**Hero run 12's acceptance, agreed Nov 1 (night): the throughput axis.**
The one-row step is converging (hero 11's ledger prices every remaining
item there at 0.1 to 0.5 ms); the throughput side has two cliffs that
each cost more than all of them together. Four workstreams in parallel,
each with its own measurement, controls on the same tree and the pinned
site table on both arms:
1. The wide-row GEMM template. Above 64 rows every fused body and one-pass
   kernel hands over to a reference path (one batch of 66 rows steps in 209
   ms against 95 at 33). A row-tiled GEMM for the fp8 weights with MXFP8
   scales, the weight streamed once per row tile, instantiated for the
   rows-over-64 class and chosen by the table above the GEMV's handover,
   with the same epilogues the GEMV template offers where the lowering
   asks for them. Acceptance: one batch of 66, 99 and 132 rows on the
   calibrated slope (about 0.6 ms a row above the 33-row step: 132 rows
   near 160 ms, about 830 tok/s), duplicate-row check clean, tokens within
   the rounding class of the reference bodies; the ledger's per-class
   efficiency for the new template.
2. Memory for batches in flight. Four batches of 66 (predicted 2,025
   tok/s) and the drafter at four batches of 33 ran out of memory on GPU 1
   (97 GB), whose 12-layer range holds 86.6 GB of experts; the cut points
   are fixed by the KV groups and the indexer (hero 9's Balance), so the
   levers are elsewhere. Deliver: a memory ledger per GPU for a planned
   shape (weights, experts, caches per batch and row, the graphs' scratch,
   the replicas), in the planner; the two levers costed with the model: an
   expert residency budget on the full GPUs (the hottest experts resident,
   the rest through the host path, per-step cost from the expert
   statistics) and a cut inside a KV group through the replica mechanism;
   then 4×66 and the drafter at 4×33 measured if a lever fits, else the
   ledger and the predicted cost of each lever, stated.
3. One expert launch per layer (hero 11's unfinished item): the gate/up
   and down phases in one kernel with a per-pair arrival count; the two
   launches cost about 2.5 µs each in ramp and tail and their tails do not
   overlap. Acceptance: expert time per layer at one row from 93 to about
   85 µs, tokens identical, nine and 33 rows (the grouped path) no worse.
4. Adaptive speculation measured in the engine: the confidence and
   expert-aware policies hero 10 plumbed, run for real at one row and at
   four 8-row batches of different prompts, against fixed drafts and
   against no drafter, with the model's prediction beside each; the
   policies' parameters in the record replace the estimates.
Deadlines: interim reports with numbers at 02:00 and 05:00 UTC, final by
07:00 UTC; results read from the pods' files.

**Hero run 12, the wide-row GEMM (Nov 2, 00:00 to 00:45 UTC; measured on
pod A, one RTX PRO 6000 node, four GPUs sharded, one batch, pinned site
table on the tiled arm, 64 new tokens, median step).** The cliff at 64
rows is gone. The three GEMV kernels (gemv_rows, the dual SwiGLU GEMV,
gemv_pe) run tiled over rows past GEMV_ROWS_MAX: a program owns one row
tile of at most 64 rows, the grid's first and fastest axis is the row
tile, so the programs that share a weight tile launch together and the
weight comes from DRAM about once (the next row tile's pass finds it in
L2). The one-pass launch is the same launch as before; the Triton bodies
in ops.py apply up to 1,024 rows (TRITON_MAX_ROWS) and the fused bodies
keep the GEMV's limit; the shape class names its tiles ("tiles=3") so the
table and the measuring step see the wide classes apart. At the real
shapes every tiled result is bit-identical to the one-pass kernel run
tile by tile (plain, narrow split, grouped with the quant epilogue, the
shared expert's two kernels, attn_q's pair) and within 2.5e-3 of
torch.matmul over the dequantized weight.

| rows a batch | prompts | tiled kernels: step ms (tok/s) | reference path, same tree: step ms (tok/s) | hero 10's reference path |
|---|---|---|---|---|
| 33 | one / cycled list | 29.0 / 92.3 | unchanged (one pass) | 95.3 |
| 66 | one / cycled list / 66 different | 45.3 (1,457) / 106.9 (617) / 110.4 (598) | 143.7 / 204.3 / 212.0 | 209 / 213 |
| 99 | one / cycled / 99 different | 60.8 (1,628) / 117.2 (845) / 135.4 (731) | not run | 230 / 240 |
| 132 | one / cycled / 132 different | 79.3 (1,665) / 143.5 (920) / 171.8 (768) | 195.2 / 245.2 / 284.2 | 297 / 286 |

Kernels a step: 1,660 at 132 rows against 14,449 on the reference path.
Tokens: every row identical to the others with one prompt at 66, 99 and
132 rows, and every duplicate-prompt row pair identical with the cycled
list (34 of 34, 67 of 67, 100 of 100 pairs; the pairs straddle row 64).
Against the model's slope: 132 rows of the cycled list at 143.5 ms is
under the 160 the 0.6 ms-a-row slope predicted; the slope with one
distinct prompt is 0.50 ms a row from 33 to 132, and with a prompt per
row 0.80 (the distinct experts grow with the rows). The reference
check's flags on the router and the attention select at 66 and 99 rows
were tested per row: the split front body is bit-identical whether run
on 132 rows or on 33-row slices, 0 of 132 rows differ from the reference
in the top-k set or the weights on real-shape inputs, and the duplicate
pairs across row 64 agree on the real model; the flags are the
statistic's maximum over every row of the tensor, the rows no request
owns included, and top-k indices that flip where two scores are within
rounding of each other (a self-tests item: live rows only, tie-aware
integer comparison). What the wide step costs, from the record at 132
rows of one prompt: the attention select's cache reads are the largest
byte item (17.5 GB a step, floor 11.6 ms, more than the dense weights'
8.5 GB), the router's split kernels 5.3 ms (132 µs a layer, linear in
rows) and hc_front 2.2, the engram linear at 39% of its floor (166 MB, a
weight the L2 does not hold across row tiles). The GEMM template proper,
with 128-row tiles for the tensor cores' sake, is the next step for the
weights above the L2 and for rows past about 200; the planner's wide
classes now have numbers to choose by.
**Hero 12, memory for batches in flight: the ledger (Nov 2).** The
planner now prices a shape's memory per GPU (`pkg/planner/memory.go`,
`vxpu plan pipeline` prints it, `ChooseLayout` refuses a layout whose
batches do not fit): the range's dense weights (the head and the
drafter's stages on the last), the experts resident there (every expert
of the range's layers, or a placement's counts through `--placement`), a
fixed share per GPU role, each batch's caches and graph intermediates
per row and layer, the reference bodies' bf16 copies of the range's
dense weights that every batch after the first carries above the
kernels' 64-row limit, and the replica of a shared cache on a range
that starts inside a KV group (the inventory now names the source
layers, 2, 8, 14 and 20). Calibrated on pod C's measured peaks (nvidia-smi
every 5 s): with the weights and the per-batch term taken out, one batch
of 33 rows leaves 11.3 GB on the first range, 3.0 on the middle ones and
7.5 on the last; four batches of 33 add 1.2 to 1.3 GB a batch on the
12-layer ranges, 3 MB a row and layer. Checked against four batches of
33 on two layouts the same night: within 1.1 GB on every GPU. What the
ledger says about hero 10's out-of-memory at two batches of 66 rows: the
second batch's graph carried the reference bodies' weight copies, 4.5 GB
on GPU 1 (predicted 101.1 GB against a 100.1 GB budget; measured 102.0 at
the capture), not the caches, which are 0.2 GB a batch; so the kernel
library's 64-row cliff was the memory blocker too, and the GEMM template
removes both faces of it. The levers priced: with the GEMM template, on
today's ranges, two and three batches of 66 fit (96.6 and 99.0 GB on
GPU 1) and four do not (101.4, 1.3 GB over); the residency lever, making
four batches of 66 fit by evicting experts from the full GPUs, costs 65
experts a layer on GPU 1 (780, 14.7 GB to host RAM) and, at uniform
popularity, 213 host-path misses a batch step, about 14 ms a step at
0.065 ms an expert (measured Oct 29); per-expert popularity is not
measured yet (the expert statistics count distinct experts a layer, not
selections an expert), so the share the cold experts receive is the
number that decides this lever, estimated today. The cut-inside-a-KV-group
layout (0:9, 9:20, 20:31, 31:43: layers 8 and 31 move off the 12-layer
GPUs through the replica mechanism) frees 7.6 GB on GPUs 1 and 2
(measured: GPU 1's peak 94.1 → 86.4 GB at one row, 97.7 → 89.6 at four
batches of 33) and shortens the slowest range to eleven layers (1,453
tok/s predicted at four batches of 33 against 1,332), and it decodes one
row correctly, but with four batches of 33 in flight it produced wrong
tokens in most rows from early positions. The bisection (pod C, 33-row
artifact, pinned table): one batch of 33 on the new layout is already
wrong (28 of 33 rows from positions 3 to 44), the cut at 9 alone
reproduces it exactly, the cut at 31 alone is correct, and the new layout
with the replicas refreshed whole (VXPU_CACHE_WINDOWS=0) is correct in
every row of four batches. The cause: hero 9's window refresh took every
shared state tensor shaped [rows, n > tokens, ...] for a [row, position,
...] cache, and the compressor's running state at a KV group's source
layer is shaped [rows, 2, 512]; a range starting one layer after a source
layer (9 after 8) replicates that state, and the gather at the clamped
position copied one of its two slots. The fix names the rule
(`stepgraph.positional_cache`: a position axis of at least 64 slots; the
2,048-position compress KV and index key caches keep the window, the
two-slot state is refreshed whole) was not it: the only state replicated
across the cut at 9 is layer 8's compressed KV cache, shaped like layer
20's, and it still decoded 28 of 33 rows wrong. The difference is the
group: layers 2, 8 and 14 compress with a running compressor (their
`compressor.kv_state` and `score_state`), which writes the cache at
compressed slots, so the window gathered at the token positions is not
what the step wrote; layer 20's group has no compressor and its window is
right. `stepgraph.compressed_cache` names that rule from the state's
names and such a cache's replica is refreshed whole (69 MB a step at 33
rows over PCIe), the rest keep the window. Verified on the new layout:
one batch of 33 and four batches of 33 decode every row as the reference
does (0 of 33 differing, against 28 before), at 97 to 101 ms a batch
step and 1,350 tok/s strict, against 95 to 99 ms and 1,363 on today's
ranges in the same session: the whole refresh of layer 8's cache costs
about what the eleven-layer range saves, so the layout's gain tonight is
the memory (GPU 1 at 89.6 GB against 97.7), and a window for compressed
caches (the compressor's own slot index) would give back the 2 to 3 ms. The wide batches on today's ranges with
the GEMM template merged (h12-gemm 645e5db), pinned table, packages
cached: four batches of 33 step in 95 to 99 ms each (1,363 tok/s strict,
every row matching the reference); three batches of 66 step in 116 to
138 ms each and fit, GPU 1 at 100.5 GB (the ledger said 99.0), two of 66
at 99.4 (96.6 said): the ledger's statement holds, and hero 10's
out-of-memory is gone with the reference bodies. Two cautions on the
66-row throughput: the harness's strict figure (806 tok/s for three of
66) divides every token by the longest batch's window, and at 64 new
tokens a 66-row batch's window is mostly the other batches' admissions
(a 66-row prefill takes about 8 s, a batch's 64 steps about 8 s), so the
steady-state figure is the per-step one, about 1,650 tok/s from the
medians, and the harness needs a decode at least four times the
admission time before the strict figure means steady state at wide
batches; and the 58 of 66 rows that differ from the 33-row reference are
the GEMM template's rounding, verified by the GEMM workstream row by row
(duplicate prompts straddling row 64 decode identically). Ranges cut
inside a KV group with the fix: see the verification line. The drafter's
memory, measured the same night (no table: a 33-row batch verifies 165
token rows): one batch of 33 with the drafter puts GPU 1 at 94.8 GB
against 93.1 plain (4.3 MB a row and layer) and GPU 3 at 84.8 against
66.0 (the drafter's stages' experts, 7.2 GB, and 11 GB of their graphs
and caches); four batches of 33 with the drafter ran out of memory on
GPU 1 on today's ranges (peak 101.7 GB) and on GPU 3 on the new layout
(peak 101.9), and the ledger, calibrated on the one-batch run, says so
for both and for the cut at 9 alone (GPU 2 at 103.3): the drafter at
four batches of 33 needs about 3 GB more on two GPUs than any four-way
layout gives, which is the residency lever's size (15 experts a layer,
about 3 ms a step at uniform popularity) or a smaller per-batch drafter
graph. The steady-state runs at 256 new tokens (the merged tree with the
GEMM and the fix, pinned table, pod C): four batches of 33 on today's
ranges step in 128 to 132 ms each at positions near 300 (97 to 99 at 64
tokens: the step grows with the position, about 0.13 ms per step per 100
positions at 33 rows, a term the pipeline model lacks); three batches of
66 in 127 to 132 ms, GPU 1 at 100.5 GB; and the shape the ledger priced at
93.9 GB, four batches of 66 on the cut-inside-a-KV-group layout with the
fix, ran: 128 to 151 ms a batch step, GPU 1 at 93.8 GB measured, 1,913
tok/s by the per-step medians against 1,019 for four batches of 33 at the
same positions, every 33-row batch matching the reference and the 66-row
batches differing from it by the GEMM template's rounding. The harness's
strict figure is not usable at 256 tokens (batches decoded 255 or 511
steps as rows were refilled, so the longest window is not a steady-state
window); the per-step medians are the measurement, and a strict window
over equal step counts is the harness fix. What the night leaves for the
planner: a shape is refused when it does not fit, with the GB it is over;
the replica rule is a memory and a time term; the four-way layouts for
the drafter at four batches of 33 are all a few GB short, which is the
residency lever's size.

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

**The planner's contract (hero run 8, first cut).** What is data, what
is code, and what runs where:
- *The kernel table* (`vxpu-kernel-table/1`) is what the planner hands
  an engine: the GPU it is for, the kernels' knobs, and per call site the
  implementation body to run. A site is the module that calls the op in
  the exported graph (`model.layers.14.engram.wkv`), which is the same
  across exports of one model; an entry may instead name the op at a
  shape key (the engine's own, `1x6144->25600` for an fp8 linear) or the
  op alone, and the engine takes the most specific entry that matches.
  The knobs are the numbers `triton_ops` set by hand (`GEMV_ROWS_MAX`,
  `FP4_DIRECT_MAX`, the fronts' chunk and group sizes, the fp4 bodies'
  overrides), typed and checked. `VXPU_PLAN_TABLE` (a file, or inline
  from the coordinator with the load) makes the engine apply the table
  exactly: the dispatch and the lowering return the named body, nothing
  is measured, and a body that cannot run at a site the lowering reaches
  is an error, not a fallback.
- *The kernel record* (`vxpu-kernel-record/1`) is what the engine writes
  (`VXPU_PLAN_RECORD`): the hardware, the choices, the knobs in effect,
  and per call site the candidates, which of them the compiler fuses,
  the microseconds measured, the body chosen and why (the table, a
  measurement, the registry's order, or kept and what kept it), and the
  kernels inlined for it as the traced graph names them (empty for a
  fused body: the compiler's kernels). Per-step statistics go in the same
  file by name (`Plan.stat`): kernels a step, crossings, the distinct
  experts a layer, and `sweeps` of a knob where a run swept one.
- *The planner* (`pkg/kernelplan`, Go; `vxpu kernels plan`) reads
  records of one GPU and writes the table: per (op, shapes) the
  measured-fastest body, a launch credited to a fused one as the engine's
  own costing does, named per site where the record ran different bodies
  at one shape; the native-format knobs from the card's compute
  capability (fp4 on the tensor cores from Blackwell on); the other knobs
  from the record's sweeps, else as the record ran. It is a function of
  the record, unit-tested on fixture records with no GPU, so a costlier
  policy (regions costed whole, the graph cut here) replaces the function
  behind the same table.
- *The check*: `vxpu kernels check` and `python -m vxpu.plan
  --check-record` compare a run's record against its table, and every
  site that ran another body is listed; `python -m vxpu.plan
  --profile-step` runs a step under the CUDA profiler on the engine's GPU
  and reports what launched next to the table, flagging compiler-fused
  kernels heavier than 50 µs where no site chose a fused body. Issue
  #490 is the live case for the first check: the engram layers'
  `fp8_linear` sites measured and chose the GEMV and the step ran the
  fused body. The graph these artifacts export calls `vxpu.fp8_linear`
  at both sites and the lowering records the GEMV for all ten; where the
  fused body enters is still open, so the check runs with the table on
  the hero machine as the acceptance for that bug.
Of hero 8's acceptance items, the table and record drive the per-site
body choices, the knobs, and the report that the choices were applied;
the op boundaries, the step graph's branches and the crossings are the
other workstreams' and the planner takes their measurements through the
same record.

**The roofline, and the production stacks.** A decode step at one row
reads every dense weight once and the selected experts once, so its floor
on one GPU is those bytes over the GPU's copy bandwidth. On an RTX PRO
6000 (copy measured at 1,460 GB/s) DeepSeek-V4.1-Flash reads 8.5 GB of
dense weights (6.9 GB in the 40 layers, 0.3 GB in the two engram layers,
the 1.3 GB bf16 head; counted from the artifact's binding) and about 4.5
GB of selected experts a step: a floor of 8.9 ms, 112 tokens/s (hero
runs 6 and 7 were written against an estimate of 10.5 GB dense, 10.3
ms). We measure against that floor and against the
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
