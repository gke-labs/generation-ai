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

**Hero run 12, one expert launch per layer (Nov 2, night; measured on pod
B).** The item's premise was that the direct path's two launches a layer
pay about 2.5 µs each in ramp and tail and that their tails cannot
overlap, so one launch would be worth the ramp of 40 launches and the
overlap. Built and measured, at the real shapes (six pairs, K 5120, I
2304, 96 slots rotated so the L2 serves no repeat, CUDA events over 48
calls): a persistent kernel per layer (`_fp4_direct_layer_kernel`: the
programs take the gate/up items and then the down items from a work
counter, a down item spinning on its pair's arrival count, the counters
self-zeroing; bit-identical to the two kernels) runs 93.5 µs a layer at
its best of 24 configurations against the two launches' 92.9: it ties,
because the merged kernel compiles to 188 registers against 132 and 128
for the two, so each phase alone runs slower in it (gate/up 65 against
57.5 µs, down 50 against 32) than the overlap recovers (13 µs); a register
cap makes it spill and lose more. The cheaper form of the same idea pays
a little: the down kernel as a programmatic dependent launch (Triton
3.8's `gdc_launch_dependents` in the gate/up kernel, `launch_pdl` on the
down kernel), each down program waiting on its own pair's arrival counter
rather than on the whole primary grid, so a pair's down tiles overlap the
other pairs' gate/up at each kernel's own register budget: 90.8 µs a
layer (the whole-grid wait 91.3), bit-identical over 200 calls; at one
pair and at eight neither variant moves (47.6 and 140 µs). In the step,
both arms under the pinned table in one session: one row 13.1 ms control,
13.0 with the dependent launch, 13.2 with the one launch; nine rows
21.2 / 17.0 / 34.3 against 21.2 / 17.0 / 34.2 and 33 rows 29.0 / 26.8 /
91.9 against 29.0 / 26.8 / 93.5 (the grouped path, untouched; the
different-prompt case moves by its usual 2 ms); tokens identical to the
control in every arm. The profiler reads a dependent launch's duration
from its early start (down 91 µs with a gap of -57 before it), so the
step's own clock is the measure, not the per-kernel times. The expectation
stated before measuring, about 85 µs a layer, was wrong; the acceptance
(expert time from 93 to about 85 µs) is not met. Both variants stay
behind table knobs, off by default (`FP4_ONE_LAUNCH` with
`FP4_ONE_PER_SM`, `FP4_ONE_WARPS`, `FP4_ONE_STAGES`; `FP4_PDL` 0, 1 or
2), and the arrival-count pattern across two dependent launches is the
reusable piece: two kernels with their own register budgets whose phases
overlap per pair, deadlock-free because the dependents are scheduled only
once every primary program has started. Seen on the way, outside this
item: eight pairs a layer (the drafter's verify shape) run at 53% of
their floor (140 µs for 98 MB), which the ledger should rank.
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

**Hero 12, adaptive speculation measured (Nov 2, pod D, the merged tree
at cfee0bd plus this branch, no plan table since the drafter artifacts'
shapes are not in it, checks in warn mode, 64 new tokens after a 4 s
warm-up, one process per case).** The engine now records what the
policies need and replays them exactly; it cannot yet apply them,
because the exported verify graph has a fixed number of token rows (one
captured graph per draft length is the missing piece, and the
experts policy also needs the router's choice for a candidate token
before the verify, which only the step itself computes). What landed:
the drafter's output carries each draft's confidence beside its tokens
(the confidence head through a sigmoid, times 1e6, in the same int64
buffer, at no step cost: 25.2 ms without it against 25.3 with it), the
native loop keeps per decoding row and step the drafts accepted and
their scores, `speculation.stats_from_events` turns them into the
per-position acceptance, the confidence of accepted and rejected
drafts, and per threshold the precision (first rejected drafts it would
have stopped before) and loss (accepted drafts it would have forgone),
`speculation.evaluate` replays fixed 1..k and every threshold over the
same steps, the record carries it all as the `speculation` statistic,
and the Go model takes the measured positions and the replays through
`vxpu plan serve|matrix --speculation-stats RECORD` in place of the
fitted alpha and the estimated precision and loss.

| case | step ms | tok/s | tokens a step | acceptance | model |
|---|---|---|---|---|---|
| 1 row, no drafter | 13.1 | 77.7 | 1.0 | - | 73 |
| 1 row, 2 fixed drafts | 22.4 | 99.0 | 2.21 | 0.62 | 93 |
| 1 row, 4 fixed drafts (tokens only) | 25.2 | 94.1 | 2.37 | 0.34 | 89 |
| 1 row, 4 fixed drafts, confidence recorded | 25.3 | 89.7 | 2.29 | 0.32 | 89 |
| 1 row, confidence 0.3 / 0.5 / 0.7 (replayed over the 4-draft run's 28 steps) | - | about 93 / 92 / 88 estimated from the replay at the measured 2.15 ms a token row | 2.21 / 1.79 / 1.61 | 0.38 / 0.63 / 0.81 | 85 at 0.5 |
| 4 batches of 8 rows, different prompts, no drafter | 38.7 / 52.5 / 37.9 / 32.9 a batch | 633 strict | 32.5 | - | 782 |
| 4 × 8 different, 2 fixed drafts | 80 / 114 / 120 / 81 | 357 | 49.3 | 0.65 | 906 |
| 4 × 8 different, 4 fixed drafts | 103 / 151 / 118 / 96 | 399 | 59.4 | 0.43 | 879 |
| 4 × 8 different, 4 drafts, confidence recorded | 103 / 155 / 145 / 98 | 370 | 58.5 | 0.46 | 879 |
| 4 × 8 different, confidence 0.5 and experts 8 / 16 | not measured: the pipeline's loops did not pass the drafter's block width (fixed after the run), so the four-batch records carry acceptance but no confidence; the experts policy has no measurement path yet | | | | |

Measured statistics: per position the share of steps whose first i
drafts were all accepted is 0.68 / 0.39 / 0.18 / 0.04 at one row (28
steps) and 0.81 / 0.52 / 0.32 / 0.18 over the four-batch run's 768
steps; the confidence head separates accepted from rejected drafts at
the first position (mean 0.78 against 0.55) and not at the fourth (0.27
against 0.38); at a threshold of 0.5 it foresees 52% of the first
rejections and forgoes 31% of the accepted drafts, so on these steps
stopping on confidence buys less than two fixed drafts do. Verdict: at
one row two drafts are the best shape (99 tok/s against 78 plain and 90
to 94 at four), the model within 7%; in four batches of eight different
prompts every drafter shape loses to plain by 35 to 45%, and the model,
which predicts the drafter winning there, is wrong by about 2.3×: the
per-batch medians spread from 80 to 155 ms against 33 to 53 plain, so
the cost of a speculative step beside three other batches is not the
0.35 ms a token row the model carries. Tokens: the drafter's verify
rounds like a wider batch, so its tokens leave the plain run's at a
near-tie position (26 to 43 at one row; 0 to 4 of 32 rows identical at
four batches), the same batch-width effect hero 10 recorded; between
drafter shapes the engine's replay is exact by construction. Next: a
verify graph per draft length so the confidence policy can act; the
drafter in the pipeline measured per batch with the step trace; the
four-batch confidence run with the fixed loops; and the experts
policy's predictor (a per-token-id expert prior) if the engine is to
prefer candidates before the verify.

**Hero run 12, closed (Nov 2, 13:30 UTC).** The throughput axis. Four
workstreams on four pods overnight, merged into vxpu-big through this
commit; every number below is measured on the hero machine with both
arms of a comparison on the same tree and the pinned site table.

| item | result |
|---|---|
| the wide-row GEMM | the 64-row cliff is gone: the three GEMV kernels tile over rows, bit-identical to the one-pass kernel per tile; one batch of 132 rows steps in 79 / 144 / 172 ms (one prompt / the 32-prompt list cycled / 132 different prompts) against 195 / 245 / 284 on the reference path of the same tree, 1,660 kernels a step in place of 14,449; every duplicate-prompt row pair identical. Above about 200 rows and for the weights the L2 cannot hold, a GEMM with 128-row tensor-core tiles is the next template, and the expert kernels need a GEMM form where many rows share an expert |
| memory for batches in flight | a per-GPU memory ledger in the planner, calibrated within about 1 GB; it explains hero 10's out-of-memory at two batches of 66 as the reference bodies' 4.5 GB bf16 weight copies per extra batch, not the caches, so with the GEMM three batches of 66 fit on today's ranges. The ranges cut inside the KV groups (0:9, 9:20, 20:31, 31:43) free 7.6 GB on the two full GPUs; they decoded wrong because hero 9's window refresh gathered a compressor-written cache (layers 2, 8, 14 have a running compressor, 20 does not) at token positions; such a cache's replica is now refreshed whole, correct in one and four batches, at about the throughput the shorter range gains. The expert residency lever is priced at about 14 ms a step for 4×66 and not taken; the drafter at 4×33 is about 3 GB short on every four-way layout |
| the throughput numbers, 256 new tokens | four batches of 33 → four batches of 66 on the cut layout with the GEMM: with the 32-prompt list cycled per batch (each prompt twice in a 66-row batch, so the expert bytes of 32 distinct rows) 1,023 → 1,913 tok/s by per-step medians; with a distinct prompt in every row 1,143 → 1,608 (+41%), GPU 1 at 94.7 GB (ledger 93.9). The step grows with position (97 → 130 ms from position 60 to 300 at 33 rows), a term the pipeline model lacks, and the harness's strict figure is not steady state at wide batches (a strict window over equal step counts is the fix) |
| one expert launch per layer | ties: the merged kernel compiles to 188 registers against 132 and 128 for the pair, so each phase runs slower alone than the overlap recovers; the down kernel as a programmatic dependent launch waiting per pair gives 2.1 µs a layer, 13.1 → 13.0 ms at one row, nothing above; both behind knobs, off. Eight pairs a layer (the drafter's verify shape) run at 53% of their floor: the ledger's next expert target |
| adaptive speculation | the engine records the drafter's confidence and per-position acceptance (cost 0) and the model takes measured inputs; two fixed drafts are the best one-row shape (99 tok/s against 78 plain, 90 to 94 at four drafts; model within 7%). In four batches of eight different prompts every drafter shape loses to plain by 35 to 45% (357 to 399 against 633 tok/s), where the model predicted a gain: the speculative step beside three other batches costs far more than 0.35 ms a token row (per-batch medians 80 to 155 ms against 33 to 53), and it is worse than hero 10's 622 for the same case, so the merged tree's pipeline (per-batch graphs on the last GPU) needs a per-batch step trace. The policies cannot yet be applied: the verify graph has a fixed number of token rows (one captured graph per draft length is the missing piece) and the experts policy needs the router's choice before the verify; the confidence head separates accepted from rejected drafts at the first position (0.78 against 0.55) |

What the drafter is, on this machine and this model: a latency tool for one to a few users. The marginal cost of a row is its experts, and with different prompts every extra verified token row routes to about six experts of its own, so the step's bytes grow with k+1 while the yield stays near two tokens; DeepSeek's own DSpark paper says the same about static multi-token drafting under high concurrency and makes the verification length a per-request, load-aware schedule, which is the planner's job here. Process: two of the four agents finished at about 00:30 UTC and sat on expired watchers until 12:54 UTC, the second time this week; the coordinator must read every pod's result file on a clock, not the agents' messages. Next, by size: the per-batch step trace of the drafter in the pipeline; one captured verify graph per draft length and the expert-aware policy; a strict steady-state window and the position term in the model; the 128-row GEMM and the expert GEMM; the compressor-slot window for the cut layout; the plain 4×8 case's slow batch (52 ms against 33 to 39).

**Open variants (Nov 2), kept because a word had hidden them.** Each
request names one instance of a goal; these are the neighbours that were
not sized when the instance was measured, with the size known today.
- The persistent executor: the mailbox model carried down to the SM, one
  resident kernel per GPU draining the step graph from a work queue with
  device counters; today every node is its own graph launch. Upper bound
  from the measured launch floor and ramps: 1 to 1.5 ms of the 12.9 ms
  one-row step, more at wide rows; risk measured in miniature by the
  one-launch expert kernel (188 registers, slower than the pair).
- Tensor parallelism of the dense path only, experts still sharded by
  layer: at one row the dense weights are 8.5 of 13 GB and the all-reduce
  carries 10 KB a layer; up to about 4 ms of the one-row step, estimated,
  a planner shape for the latency scenario; held until the planner has
  shapes to choose between.
- The drafter: scheduled verification length per request (DSpark), a
  verify graph per draft length, the conditioning between draft positions
  (acceptance 0.83, 0.41, 0.19, 0.04 by position against about 0.8 at
  every position in DeepSeek's production), and its placement on the GPU
  every batch also finishes on.
- Numerics classes as the acceptance bar instead of bit-identical tokens:
  the shared-expert join and the recode prologue are off for rounding, not
  speed; the reference-alongside check is the arbiter the self-tests built.
- Expert residency by measured popularity and router-predicted prefetch:
  the lever was priced at uniform popularity (14 ms a step) and dropped;
  per-expert popularity has never been measured.
- Prefill as part of the step: a 66-row prefill takes about 8 s, rows one
  at a time through a program with its own dense weights, as long as that
  batch's 64 decode steps; batched prefill would be a few hundred
  milliseconds. Probably the largest unexamined cost in steady-state
  throughput at today's concurrency.
- Admission that groups rows whose expert sets overlap, the batching form
  of expert-aware drafting; the per-batch distinct-expert counters price it
  in one run.

**Hero run 13's acceptance, agreed Nov 2 (afternoon): the drafter.** The
drafter is the DSpark head the checkpoint ships (three stages, a block of
five, a Markov head and a confidence head), and in our engine it accepts
0.83, 0.41, 0.19 and 0.04 of its positions where DeepSeek's production
accepts about 0.8 at every position and 4.0 of 5 live; in four batches
of eight different prompts every drafter shape loses 35 to 45% to plain
decode where hero 10 measured a smaller loss and the model predicted a
gain; the engine runs k fixed drafts through one verify graph whatever
the draft length. Four workstreams in parallel, controls on the same
tree, the pinned site table where the shapes allow:
1. The drafter's correctness. For the same prefixes, the fork's reference
   drafter and ours compared position by position (tokens and
   confidence), on the tiny fixture and on the real model; the export of
   the drafter stages checked for the conditioning between positions and
   the inputs the stages read (the attention input of layers 37 to 39, the
   previous draft position); any defect fixed; the per-position acceptance
   re-measured at one row on the 32-prompt list and on a code-prompt list.
   Acceptance: drafted tokens equal the reference's, and the acceptance
   curve explained (a defect fixed, or the workload shown to be the cause
   with the code list).
2. A verify graph per draft length and the confidence policy applied.
   Captured verify graphs for k = 1 to 5 token rows so a shorter draft
   costs its own step, not the five-row step; the confidence policy chooses
   k per step from the drafter's confidence (the pure module's rule); the
   step cost per k measured. Acceptance: step time at k drafts within 10%
   of the model's line in k; confidence at 0.5 measured against fixed 2 and
   4 at one row and at four batches of eight different prompts.
3. The drafter in the pipeline. A per-batch step trace of four batches of
   eight different prompts with the drafter (per-batch medians 80 to 155 ms
   against 33 to 53 plain tonight; hero 10 measured 622 tok/s against 781
   plain for the same case, hero 12 357 to 399 against 633); the regression
   between the two trees named and fixed; the drafter's graphs placed by
   the planner (today on the last GPU, which every batch also finishes on;
   the first and last GPUs idle 17 to 19% of a period). Acceptance: the
   4×8 drafter case at or above hero 10's number with the cause named, and
   the placement choice recorded as a plan field.
4. The model and the expert-aware policy, from measurements. The
   speculative step's cost in flight measured (distinct experts per token
   row per batch, the drafter's own step, the contention) and the model
   corrected to within 15% on the 4×8 case; the expert-aware policy's upper
   bound measured offline from the recorded router choices (how many new
   experts each draft position would add, how many drafts a budget of 8 or
   16 new experts a layer keeps, and their acceptance); the harness reports
   acceptance per prompt set.
Deadlines: a one-line STATUS file in each workstream's pod directory
every 30 minutes, read by the coordinator on a clock; interim reports
with numbers at 17:00 and 20:00 UTC; final by 22:00 UTC. The six runs
after this one are the coordinator's to order from the open variants.

**Hero 13, the speculation model and the expert-aware policy, from
measurements (Nov 2, pod C, the tree at 0286e20 plus this branch, no plan
table since the drafter artifacts' shapes are not in it, checks in warn
mode, 64 new tokens after a 4 s warm-up, one process per case; the expert
statistics and the new expert trace on, which cost about 0.8 ms a step at
one row).** Why the model said the drafter wins four batches of eight
different prompts (906 tok/s at two drafts against 782 plain) where hero
12 measured 357 against 633 is three things, two of them the harness's:
(1) a fixed-length run steps until its slowest row has its tokens, and
with different prompts the rows accept very differently (one row of the
prose list accepts 0.68, 0.39, 0.18 and 0.04 of its four positions; one
row of a code list 0.80, 0.65, 0.40 and 0.30; the same step of 26 ms
yields 2.29 and 3.20 tokens), so eight rows with four drafts took 42
steps for 64 tokens with a mean of 4.9 rows still active where the mean
row needs 23 (the planner now predicts this: `vxpu plan serve
--new-tokens`, `Acceptance.Completion`, a ceiling since it draws rows
independently, 27 steps for the slowest of eight); (2) the strict figure
divides by the longest batch's window; (3) two of the four batches
stepped at 114 and 120 ms against 80 and 81 for the other two (the
pipeline workstream's). The steady-state per-step cost was within about
10% of the model for the two unimpeded batches, and the model, refitted
below, now says the drafter loses that case (686 tok/s against 782), as
it does.

Where a verify's token rows cost, from the step's GPU time by kernel
family (`dev/hero/kprof-rows.py`), eight rows of different prompts: plain,
48 (token row, expert) pairs a layer, the grouped expert kernels take 16.7
ms for 25.8 GB, at the 1,535 GB/s read peak; two drafts, 144 pairs, 50.0
ms for 51.5 GB that stream in 33.3; four drafts, 240 pairs, 63.8 ms for
64.1 GB that stream in 41.5: the kernels do a GEMV per pair and the extra
pairs cost 2.1 to 2.6 µs each beyond the bytes. The dense families grow
0.28 ms a token row, the drafter graph alone replays in 2.5 ms at one row
and 3.8 at eight, and past eight token rows the step's GPUs wait 10.5 ms
for the experts host node (eager fp32 GEMMs and about a thousand small
kernels for the routing of 24 or 40 token rows: an engine defect, not a
cost of speculation, priced in the model until it is removed; none at
eight rows plain, none with one prompt in every row). At 33 rows with
the five-token block (990 pairs a layer, 172 distinct experts) the expert
kernels take 125 ms for 129 GB that stream in 84, 1.0 µs a pair: the
per-pair cost falls as more token rows share an expert (2.6 µs at 30
pairs, 2.1 at 240, 1.0 at 990), so the model takes 2.75 µs a pair over
one plus the pairs over 600. The token curve's exponent falls with the
rows too (0.70, 0.60 and 0.49 reproduce the measured 18.4, 85.3 and 171.7
distinct experts at one, eight and 33 rows with five token rows). That
33-row step is 228 ms today, 326 tok/s in a fixed-length run against
about 350 plain, where hero 10 measured 355 ms on reference bodies above
64 rows: the GEMM tiles of hero 12 serve the verify's 165 token rows, and
the drafter at 33 rows is now at par rather than half. The model with
these terms (`pkg/planner/speculation.go`: bytes at the read peak plus
pairs times `ExpertPairUs` over its saturation, `HostGapMs` past
`HostGapRows`, `PerDraftRowMs` 0.28, the drafter 2.8 ms plus 0.13 a row)
predicts 25.3 ms at one row with four drafts (measured 25.3 to 26.1), 21.7
at two (22.4), 40.3 at eight rows of one prompt (40.4), 81.7 at eight rows
of different prompts with two drafts (78.6), 102.8 with four (95.1 to 97.3;
7 ms of that is the pipeline model's plain eight-row step, 34.9 against a
profiled 27.8), 223 at 33 rows with four (228). For four batches of eight
different prompts it predicts 764 tok/s at two drafts and 758 at four
against 782 plain in steady state, where hero 12's per-batch medians give
about 750 and 790, within 3%; for four batches of 33 it predicts a 13%
gain from four drafts, unmeasured (the drafter at four batches of 33 does
not fit in memory, hero 12). `vxpu plan matrix` was regenerated with it
and the eight-row measured acceptance (`dev/hero/predictions.json`).

The expert-aware policy, measured offline. `VXPU_EXPERT_TRACE=1` keeps the
experts every token row selected, per layer and step, and
`speculation.experts_replay` walks the run as the policy would: the plain
step reads every row's first token, then each draft is kept while the new
experts its row's drafts add, averaged over layers, stay within a budget,
and a kept draft yields its token when the run accepted it. At eight rows
of different prompts with four drafts, a draft adds 2.1, 1.6, 1.1 and 0.9
new experts a layer by position (the rows already read 34 of the 384, and
consecutive tokens overlap more than rows do), so a budget of 16 keeps
every draft (81.5 distinct experts against the fixed policy's 85.3), 8
keeps 1.0, 1.0, 0.99 and 0.87 of the positions (79.5), and 4 keeps 0.99,
0.53, 0.22 and 0.12 (58.2 distinct, 22.8 token rows, 2.12 tokens a step
against 2.82). Priced with the refitted step: plain 287 tok/s (the
profile's 27.8 ms), fixed four drafts 231, budget 16 234, budget 8 235,
budget 4 240. The upper bound is a 4% gain over fixed drafting and still
under plain, because a draft token costs its six pairs (0.5 ms a step over
the 40 layers) and its share of the host gap, not the one or two experts
it adds; the lever that would change the sign is the expert kernels'
per-pair cost (a GEMM form where token rows share an expert) and the host
node's gap, after which the same replay says fixed four drafts would reach
about 340 tok/s against 287 plain at eight rows. Acceptance per prompt set
is in the harness and the record (`prompt_set`, `prompt_rows` beside the
`speculation` statistic).
**Hero 13, the drafter's correctness (Nov 2, measured).** The question
was whether the steep acceptance by position hero 12 saw (0.83, 0.41,
0.19, 0.04 at one row) is a defect in how the engine runs the DSpark
head. It is not; it was a 28-step sample of one prompt, and the rest is
the head and the workload.
- The export is the fork's drafter. `draft_check` feeds the fork's own
  model the engine's output one token at a time and drafts at every
  (token, position) the engine drafted at: on the tiny fixture, over three
  rows with prompts arriving at different steps and 32 tokens each (the
  window rings wrap, drafts follow partial acceptances), 77 of 77 drafts
  are bit-identical to the reference's, the engine's tokens are the fork's
  greedy continuation, and the confidence agrees within 1e-4. The drafter
  takes the argmax (temperature 0; five argmax nodes, no sampling) and
  conditions each position on the previous draft through the Markov head
  as the fork does. A selftest check now.
- The native loop runs it the same way. On the real model, 32 prompts of
  the prose list and 32 of a coding list (`dev/hero/prompts-code-32.txt`),
  64 tokens each, four fixed drafts: the native loop and the Python-driven
  rows give the same acceptance to the third decimal and the same tokens.
- The verify is exact. With the drafter replaced by the plain run's own
  next tokens, every draft is accepted while the tokens agree (20 of 20
  over 24 tokens; across 12 prompts, acceptance equals the share of
  tokens before the run's first divergence from the plain run); with
  every draft wrong, the output still follows the plain run to the same
  kind of position. The divergences from the plain run themselves (27 of
  32 prompts, first differences from position 3) are the step's row count
  rounding differently, not the verify: a plain nine-row step with no
  drafter diverges from the one-row step in 29 of 32 prompts, first
  differences from position 3 as well. Comparisons across row counts
  belong in the rounding class, not the bit-identical one.
- What the acceptance is (765 and 627 drafting steps): prose, by position
  0.744 / 0.458 / 0.298 / 0.176, conditional 0.74, 0.62, 0.65, 0.59, 2.68
  tokens a step, 85 tok/s against 63 plain; code 0.820 / 0.624 / 0.475 /
  0.354, conditional 0.82, 0.76, 0.76, 0.75, 3.27 tokens a step, 95 tok/s
  against 60 plain. DeepSeek's live figure of about 4 of 5 is the code
  curve at five drafts. The confidence head separates accepted from
  rejected drafts at every position (prose 0.81 against 0.56 at the first,
  0.68 against 0.40 at the fourth; code 0.86 against 0.65), so the
  confidence policy has the signal it needs once the verify graph per
  draft length exists.
**Hero 13, a verify step per draft length (Nov 2; measured on pod A, one
row, specm5c-b1 and per-length exports, 64 new tokens after the warm-up,
checks in warn mode).** The export fixes the verify graph's token rows (the
block plus one), so a shorter draft could not cost less and hero 12
recorded the confidence policy without applying it. Two things changed.
First, the plan's policy is applied in the step: a row issues the drafts
the policy allows (a fixed count, or up to the first draft under the
confidence threshold), and the verify's positions past them carry the
last issued token again, so the tokens stay those of one token a step
(a padded position the verify agrees with is a correct greedy token) and
the statistics count only the issued drafts (`DecodeLoop.set_policy`,
`speculation.compose_step`; a named policy takes the host path for the
drafts, which costs nothing measurable: 25.3 ms either way). Second,
what a padded position saves was measured against graphs exported with
fewer token rows:

| drafts a step | padded, through the 5-row graph | its own graph (per-length export) |
|---|---|---|
| 0 | 23.5 ms (43 tok/s) | 13.1 (plain, no drafter) |
| 1 | 24.4 (75) | 20.0 (89) |
| 2 | 25.0 (88) | 22.4 (99) |
| 3 | 25.1 (93) | 23.7 (100) |
| 4 | 25.3 (90) | 25.3 (90) |

A padded position costs almost what a drafted one does (a repeated token
at a later position routes to experts of its own), so the policy is
exact through padding but cheap only through a graph with that many
rows. The per-length line is not the acceptance's straight line from 13.1
to 25.3 (16.1, 19.2, 22.2 would be): the first draft costs 6.9 ms, the
next three 2.4, 1.3 and 1.6, so about 4 ms of the first draft is the
drafter's own step and its hand-over to the verify, not a token row; the
best fixed length at one row is three or four drafts (100 tok/s) and
two (hero 12's 99) is within noise of them. The confidence policy, padded:

| policy | step | tok/s | tokens a step | issued / accepted (of 4 a step) | acceptance by position |
|---|---|---|---|---|---|
| fixed 4 | 25.3 | 89.7 | 2.29 | 112 / 36 | 0.68, 0.39, 0.18, 0.04 |
| confidence 0.3 | 25.1 | 101.6 | 2.56 | 79 / 38 | 0.80, 0.42, 0.29, 0.23 |
| confidence 0.5 | 24.7 | 89.4 | 2.21 | 49 / 34 | 0.90, 0.60, 0.38, 0.60 |
| confidence 0.7 | 24.4 | 77.0 | 1.88 | 37 / 29 | 0.87, 0.70, 0.50, 0.50 |

The confidence head predicts acceptance: at 0.5 the policy issues 44%
of the drafts and keeps 94% of the accepted tokens. At 0.3 it is the best
one-row number of the week, 101.6 tok/s, 13% over fixed 4, from yield
alone (2.56 tokens a step at the same step time). With a graph per
length and the measured per-length costs, the same issued lengths would
step in about 24.0 ms at 0.3 (107 tok/s, estimated) and 21.9 at 0.5
(101); so the per-length graphs are worth about 5% more at one row, and
the drafter's fixed 4 ms is the larger target. Four batches of eight
different prompts at 64 tokens (admission, not steady state): plain 670
tok/s strict, fixed 2 271, fixed 4 372, confidence 0.5 312. At 256 tokens,
steady state, per-batch medians and strict tok/s: plain 35 to 39 ms, 419;
fixed 2 116 to 130 ms, 267; fixed 4 126 to 149 ms, 324; confidence 0.5
115 to 125 ms, 309, issuing 1.7 drafts a step where fixed 4 issues 4 and
accepting 0.80, 0.62, 0.50, 0.41 by position against 0.72, 0.43, 0.25,
0.14. So in flight the confidence policy beats fixed 2 and loses to fixed
4 by 5% through padding, and every drafter shape loses to plain by a
quarter or more: a speculative step beside three other batches costs 3 to
4 times the plain step where its experts explain about 2. The same run's
four batches of one row show the mechanism: plain 22 ms a batch step
(13.1 alone), with the drafter 85 to 88 ms a batch step (25.3 alone), the
drafter's graphs on the last GPU serializing against every batch's last
range; that regression is the pipeline workstream's (hero 10 measured 27
to 36 ms there). What a graph per length needs: an
export per token-row count (the artifacts are 50 MB each and export in
under a minute), the dense weights shared between the programs (today
each program holds its own), and the loop choosing the graph by the
drafts issued; the policy's padding stays for the rows a graph does not
have.

**Hero run 14's acceptance, agreed Nov 2 (afternoon), pulled forward by
the user: the dense layers across the cards, the experts per card.** The
comparison every published number is made at is four cards streaming
together, and our one-card-at-a-time step caps single-stream decode near
78 tok/s plain and 102 with the drafter while TP4 stacks report 250 to
494 on the same cards; the pipeline's steps slow each other 1.3× when
they overlap; the two 12-layer cards hold 86.6 GB of experts while the
other two idle. The shape: every dense weight split four ways (columns
into a block, rows out of it), two cross-card sums per layer, the router
run on the replicated sum so each card knows the routing without a
message, each card holding a quarter of every layer's experts and running
only its pairs, its partial joining the FFN's sum. Measured on pod A
before the run (`dev/hero/crossings.py`): flag latency 2.0 µs on a
switch and 2.7 across the sockets, a 10 KB peer copy 10 µs, a captured
chain of 86 one-shot all-reduces with a GEMV between them costs 0.9 ms a
step at one row, 1.4 at 8 rows, 4.4 at 33 and 16.7 at 132 (bandwidth
bound at about 40 GB/s a card: wide rows need a ring or reduce-scatter,
or TP on fewer cards); the eager path costs 22 ms a step and is
unusable, so the TP step is captured as the step graph already is.
Expected from the measured bytes (estimated): one row plain about 5 ms
(13 GB over four cards plus 0.9 ms of crossings plus the small-kernel
tail per card), about 200 tok/s against 78; the speculative verify about
6 ms for 2.6 tokens, near 400 tok/s; one 132-row batch against the full
aggregate bandwidth with no pipeline phasing; memory about 74 GB a card,
balanced. Four workstreams:
1. The dense path split four ways: weight slices per card at load from
   the plan (`tp` for the dense ops), the GEMV templates on the slices,
   attention split by heads (64 heads, 16 a card) with the compressed KV
   latent replicated, the two sums per layer as a captured all-reduce op,
   the drafter stages the same way; the one-row step end to end,
   tokens within the rounding class of the layer-sharded step.
2. The experts per card: a placement of each layer's experts across the
   four cards (by id, then by measured popularity), the direct and grouped
   fp4 kernels running each card's pairs, the partial output into the
   FFN's sum, the expert host node's glue per card (the 10.5 ms gap past
   eight token rows in view); expert time per card per step at 1, 8, 33
   and 132 rows against today's.
3. The all-reduce as a kernel template: one-shot with peer stores and
   flags for small payloads, ring or reduce-scatter for wide rows, chosen
   by the planner from the measured curve, captured, fused into the
   producing GEMV's epilogue where the partial can be written straight
   into the peers' inboxes (the ring write of hero 11 is the precedent).
4. The planner's shape: `tp` and `ep` in `vxpu-plan/1`, the memory ledger
   for the layout, the cost model of a TP4 step (bytes over four cards,
   crossings by rows, the tail per card), predictions for 1, 8, 33 and
   132 rows plain and with the drafter, and `vxpu plan serve` choosing
   between layer sharding with the pipeline and TP4 with EP4 per load.
Acceptance, as an outcome (Nov 2, afternoon; the user: heroes are defined
by outcomes, not work): on four cards, single stream, DeepSeek-V4.1-Flash,
plain decode at or above 150 tok/s and with the drafter at or above 200
tok/s, measured with the harness's own control on the same tree, tokens
in the rounding class; and one 33-row batch of different prompts at or
above 1,000 tok/s. Pass or fail on those three numbers. If the TP4 + EP4
step does not reach 150 tok/s plain at one row, tensor parallelism on
PCIe-only cards is dropped as a direction for this box, and no later run
is promised to rescue it. Also to report: a TP4 + EP4 step running end to end on the real model with
tokens in the rounding class; the four-card table (one row plain and with
the drafter, 8, 33 and 132 rows) measured against the layer-sharded
numbers and against the published TP4 numbers, with the predictions
beside them; the planner's choice per load recorded. A run of more than a
day: interim reports with numbers at 19:00, 23:00 and 03:00 UTC, final
by 07:00 UTC Nov 3; STATUS files every 30 minutes; the coordinator polls.

**Hero 14, the planner's side (Nov 2, written before the measurements).**
The plan file gained a `parallel` section (`vxpu-plan/1`: `dense_tp` 1, 2
or 4, `expert_ep` 1 or `dense_tp`, the group's `gpus`, the expert
placement `by-id` / `by-popularity` / a file, the `reduce` form `one-shot`
or `ring`; a tensor-parallel plan has one range, the whole step, on the
group's first card; `physplan.py` validates it the same way and renders
it as `VXPU_DENSE_TP`, `VXPU_EXPERT_EP`, `VXPU_TP_GPUS`, `VXPU_EP_EXPERTS`
and `VXPU_REDUCE` for the engine). The tensor-parallel step's model
(`pkg/planner/parallel.go`): the dense bytes over the cards together at
an efficiency of 0.90 with the whole weight (the ledger) and an estimated
0.75 at a quarter of the columns; the experts of a layer spread over the
cards with the fullest card serving the mean share plus 0.9 standard
deviations (2.45 of 6 at one row, 36 of 127 at 132 rows), at the kernels'
measured 0.82; two cross-card sums a layer at the measured one-shot cost
(10 µs a reduction at one row, 16 at 8 rows, 52 at 33, 194 at 132, from
`dev/hero/crossings.py`) or an estimated ring at half the one-shot past
33 rows; the small kernels' launch floor on every card (1.5 ms, not
divided), the host node, and 0.50 ms a row (the GEMM path's measured
slope) over the cards; the drafter's terms from the speculation model.
With one card the same formula reproduces the measured steps within 12%
(one row 13.1, four drafts 25.3, 33 rows of one prompt 29.0, 132 rows
79.3 and 171.8; `parallel_test.go`), and the distinct-expert curve now
carries hero 10's wide-batch points (91 at 66 rows, 111 at 99, 127 at
132). `vxpu plan serve --parallel auto` chooses between the layer-sharded
shapes and the cards together per load and prints both with why;
`ChooseServing` prices the memory of each (the tensor-parallel layout:
74 GB of experts and 2.3 GB of dense weights a card, the caches split by
heads; 132 plain rows fit at 89.6 GB, 132 rows with the drafter do not by
0.6 GB on the first card). What it predicts for the run, the numbers it
is judged against (`dev/hero/predictions.json`, keyed by `parallel`):

| load | layers sharded (today) | the four cards together |
|---|---|---|
| 1 row plain | 13.7 ms, 73 tok/s (measured 13.1, 78) | 6.2 ms, 161 tok/s |
| 1 row, 2 / 4 drafts | 99 / 105 tok/s (measured 99 / 90) | 10.5 / 11.7 ms, 203 / 228 tok/s |
| 8 rows, one prompt | 18.0 ms, 446 | 7.6 ms, 1,055 |
| 8 different prompts, plain | 34.9 ms, 229 (4 batches in flight: 782) | 13.8 ms, 580 |
| 8 different prompts, 2 drafts | 210 (4 batches: 704) | 36.1 ms, 474 |
| 33 rows, one prompt | 33.2 ms, 994 (4 batches: 3,394) | 11.8 ms, 2,790 |
| 33 different prompts, plain | 83.6 ms, 395 (4 batches: 1,329) | 29.3 ms, 1,128 |
| 132 different prompts, one batch | 165.5 ms, 798 (measured 171.8, 768) | 54.3 ms, 2,431 |
| 132 rows, one prompt | 93.6 ms, 1,410 | 29.8 ms, 4,429 |

So the model says the cards together win every single-batch load by 2 to
3×, and lose to four batches in flight only where the pipeline's steady
state is the comparison (8 and 33 different prompts: 782 and 1,329 in
flight against 580 and 1,128 in one batch), with the caveat that the
pipeline's in-flight numbers are the model's, which hero 13 showed
overlapping steps undercut by 1.3×. The uncertain terms, in order: the
dense efficiency at a quarter of the columns (0.75 estimated; each 0.05
is 0.13 ms at one row), the per-row term under tensor parallelism
(assumed to divide by the cards), the ring reduction (estimated), and
the host node's gap, which is kept whole. The drafter's gain shrinks
under tensor parallelism (161 → 228 tok/s, 1.4×, against 1.3× today):
its own step and hand-over (about 4 ms) do not divide by the cards.
**Hero 13, the drafter in the pipeline (Nov 2, pod B; measured, the 32-prompt
list cycled per batch, the pinned table where the artifact allows).** The
question was why four batches of eight different prompts with the drafter
measured 622 tok/s strict in hero 10 and 357 to 399 in hero 12. It was
not the tree: hero 10's tree (bec0963) run today on hero 10's own artifact
(specm5x-b8) gives per-batch step medians 96/154/156/100 ms and 502 strict
against the 94/113/102/96 and 622 it recorded, today's tree 102/152/142/98
and 372, and the plain 4×9 case is unchanged on both (39.7/44.7/43.2/36.2
against 40.5/43.5/44.6/36.6). The order of the harness's modes changes
nothing. The step trace now carries each step's absolute start (t0_ns),
and the four batches laid side by side say what the numbers were
measuring:
- At 64 new tokens the batches are never all in flight: rows are admitted
  one prefill at a time (1.2 to 2 s a batch) and a batch's 28 speculative
  steps take 3 s, so batch 3 starts when batches 0 and 1 have finished; the
  steady span has one step in flight 40% of the time and two 60%, never
  three or four. A step takes about 96 ms alone and 130 to 155 ms with one
  other batch's step overlapping it, in both trees; the "slow batches" are
  the ones that overlapped most, and the strict figure (tokens over the
  longest batch's window) moves with the admission phase. Hero 10's 622 and
  hero 12's 357 to 399 are two draws of that, not two engines.
- The 1.5× slowdown of two overlapping steps is GPU 3's double role: the
  drafter's three stages and every batch's last range run there. With the
  drafter's stages and their experts on GPU 0 (today's `drafter.gpu` plan
  field, rendered as VXPU_DRAFT_DEVICE; the engine places the layers after
  the backbone's there and moves their experts there whatever the placement
  file says), the overlapping steps stop slowing each other at 64 tokens:
  medians 97/101/98/97, p90 103 to 119 against 164, aggregate 602 tok/s
  against 478 (+26%), tokens identical; one row alone is unchanged (25.4 ms
  with the drafter on either GPU). Through the plan field alone
  (VXPU_DRAFT_DEVICE=cuda:0 with the unchanged placement file) the run
  gives the same: 98/102/99/98 ms, aggregate 602, the engine reporting the
  three stage layers' experts moved from GPU 3.
- At 256 new tokens the pipeline is four deep (four steps in flight 64 to
  76% of the steady span) and a batch's step is then the pipeline's
  period times its depth, so per-batch medians inflate by design: plain
  4×8 steps at 37 to 40 ms a batch (33 to 41 alone), the drafter 4×8 at
  164 to 166 everywhere (96 alone), tokens per step falling from 2.3 over
  the first 64 tokens to 1.45 over 256 as acceptance decays with position.
  The drafter on GPU 0 is worse at this depth (200 to 202 ms: GPU 0 becomes
  the long pole with the stages' work on top of its eight layers), so the
  drafter's GPU is a choice the planner must make from the ranges' load,
  not a constant; the field exists for that.
- A per-GPU turnstile (VXPU_PIPELINE_TURNSTILE=1: each GPU runs the
  batches' graphs one at a time in launch order, a ticket per GPU per step,
  cuStreamWaitValue32 before the graph and cuStreamWriteValue32 after)
  helps the plain pipeline a little and tightens its tails (4×9 at 64
  tokens 39.2/39.6/37.0/35.3 and p90 43 to 45 against 39.7/44.7/43.2/36.2
  and p90 48 to 54, strict 913 against 818; 4×8 at 256 tokens p90 41 to 45
  against 54 to 57) and hurts the drafter badly (182 to 188 ms at 64 tokens,
  412 at 256), because the drafter's graph and the verify's last range take
  consecutive tickets on GPU 3 and every other batch queues behind both.
  With the drafter on GPU 0 and the turnstile together the run stalled
  after about 100 steps (the GPUs at 0% for an hour, the traces stopped):
  a batch then holds two tickets a step on GPU 0, the drafter's and the
  verify's, interleaved with the other batches', and the ordering admits a
  cycle the ticket argument missed. Off by default, marked experimental; a
  turnstile that gives the drafter's graphs no ticket (they are short) is
  the version to try, with a watchdog.
What stands: the acceptance's 622 strict for the 4×8 drafter case is not
reached by any configuration (best 475 strict and 602 aggregate at 64
tokens with the drafter on GPU 0), and the number itself was an admission
artefact; in steady state the drafter in flight at eight different rows a
batch yields about 300 tok/s aggregate against the plain pipeline's 400 to
580, the conclusion hero 12 drew and the DSpark paper states for static
drafting under concurrency. The harness needs a strict window over equal
step counts and an admission that prefills a batch's rows together before
the pipeline's throughput can be read off a 64-token run.

**Hero run 13, closed (Nov 2, 15:45 UTC).** The drafter. Four
workstreams in parallel, all merged through this commit; every number
measured on the hero machine.

| question | answer |
|---|---|
| is the drafter run correctly | yes: 77 of 77 drafts on the tiny fixture bit-identical to the fork's drafter (now a selftest check), native and Python paths identical on the real model, the verify exact under perfect drafts; the steep curve was a 28-step sample of one prompt |
| what the acceptance is | over 32 prompts the conditional acceptance beyond the first position is a steady 0.6 on prose (0.74 / 0.46 / 0.30 / 0.18 by position, 2.68 tokens a step, 85 tok/s against 63 plain in that harness) and 0.76 on code (0.82 / 0.62 / 0.48 / 0.35, 3.27 tokens a step, 95 against 60); DeepSeek's live "about 4 of 5" is the code curve at five drafts |
| the confidence policy | applied in the step and exact; at one row threshold 0.3 gives 101.6 tok/s, 13% over fixed four drafts from yield alone, the best one-row number of the week; it saves little time today because the verify graph has a fixed token-row count (a padded verify costs 25.1 ms whatever is issued), so a graph per draft length needs an export per count; the first draft costs 6.9 ms of which about 4 is the drafter's own step and hand-over |
| why the drafter loses at concurrency | the grouped expert kernels do a GEMV per (token row, expert) pair, 2.1 to 2.6 µs a pair beyond the bytes: eight rows plain are 48 pairs a layer reading 25.8 GB in 16.7 ms at the read peak; with four drafts 240 pairs read 64 GB in 63.8 ms where the bytes would take 41.5; past eight token rows of different prompts the GPUs also wait 10.5 ms a step for the experts host node's eager routing glue (an engine defect, to file). The model priced per pair is within 3% of hero 12's four-batch steady state and within 10% of every single-batch case |
| the expert-aware policy | measured offline from recorded router choices: a draft adds only 1 to 2 new experts a layer at eight rows, so choosing drafts by experts is worth 4% over fixed drafting and still under plain; the sign flips only when the per-pair cost goes (an expert GEMM form) and the host gap goes, after which fixed four drafts would be about 340 against 287 plain at eight rows |
| the pipeline "regression" | none: hero 10's own tree on its own artifact gives 502 tok/s strict today against the 622 it recorded; at 64 new tokens at most two of four batches are ever in flight (admission one row at a time), and a step costs 96 ms alone and 130 to 155 ms when another batch's step overlaps it; the drafter's GPU is now a plan field, and moving it off the last GPU removes the overlap slowdown at 64 tokens (+26% aggregate) but not with the pipeline four deep, so it is the planner's choice from the ranges' load; the turnstile (each GPU running the batches' graphs one at a time) helps the plain pipeline a little, hurts the drafter, and stays off |

What the drafter is, measured: a latency tool for one to a few users
(one row 78 → 102 tok/s with confidence 0.3; code prompts higher), a loss
at concurrency with different prompts until the expert kernels cost
bytes rather than pairs. Harness lessons, to fix before the next pipeline
number is quoted: a strict window over equal step counts, batched
admission (a batch's rows prefilled together), and 256 new tokens for
anything with batches in flight. Open from this run: an export per
token-row count so the confidence policy saves time; the expert GEMM
form; the experts host node's 10.5 ms gap; row ids in the loop's record;
the hero 12 confidence artifact failing to capture at eight rows on one
tree. Process: heartbeats every 30 minutes and a coordinator poll every
20 kept every workstream visible; all four reported inside the deadline.
**Hero 14, the dense layers across the cards (Nov 2, measured on pod A;
the run's outcome).** Not met. The bar was a plain one-row step at or
above 150 tok/s (6.7 ms) with TP4 + EP4; the step measured 13.1 ms, 76
tok/s, the same as the layer-sharded control on the same tree. The 33-row
bar (1,000 tok/s) was not met either: 32 different prompts decode at
52.6 ms a step, 624 tok/s, against the control's 92.2 ms. The drafter was
not built for the ranks (the four-rank export carries one drafter
program, and the ranks' decode is plain greedy); by hero 13's measured
gain at one row (78 to 102 tok/s, 1.3×) the drafted TP4 + EP4 step would
be about 100 tok/s against the 200 bar: not met by estimate. The
direction is dropped for this box, by the acceptance's rule.

What was built, all on the real model, tokens in the rounding class of
the control (equal for 26 to 42 positions of 64; the ranks round a
row-parallel sum twice, once per rank's half and once after the sum,
where one card rounds once; the selftest judges the two-rank decode of
the tiny model by `compare()`'s strict rule for that reason): the export
of a decode program per rank (`vxpu.export --tp N`, the fork built under
a stand-in for `torch.distributed` whose all_reduce and all_gather become
`vxpu::tp_allreduce` and `vxpu::tp_allgather` in the graph, the slice of
every parallel weight recorded in `binding["parallel"]`); the executor's
ranks as threads on their cards (`parallel.World`), each binding its
slices of the checkpoint's tensors, running its own program through its
own step graph, the collectives captured as peer exchanges; the sums
through the all-reduce template (`vxpu::allreduce`, one-shot; the Comm
set up by rank 0 for the step's 89 sites, 80 of them fp32 (the residual
stream) and 9 bf16 (the index scores), its inboxes bytes viewed per
call); the experts under either placement with one rule shared with the
experts workstream (a rank computes only the pairs its own card's pools
hold and hands on the partial; `_partial_experts` = `expert_ep` > 1 or a
tensor-parallel world is bound); the rows of a batched artifact decoded
together in lockstep; a step trace for the ranks (`VXPU_STEP_TRACE`: the
host's phases around the graphs and the host nodes' round trips inside
them); `reduce` = `one-shot` | `ring` | `skip` (no sums: the timing
floor, wrong tokens).

The table, one step in ms, median of 64, the prompt set of the harness
(plain, `VXPU_FUSE=` on both sides; `tok/s` is rows over the step):

| load | control (layers sharded, same tree) | TP4, experts layer-sharded | TP4 + EP4 |
|---|---|---|---|
| 1 row | 13.1 (78 tok/s) | 14.3 ring sums, 13.4 one-shot, 9.4 no sums | 13.1 (76), 10.9 no sums |
| 8 rows, one prompt / 8 copies / 8 different | 20.8 / 16.6 / 32.3 | 23.6 / 19.2 / 35.0 | not run (dropped by the coordinator once one row missed) |
| 33 rows, one prompt / 33 copies / 32 different | 29.0 / 26.7 (1,254) / 92.2 (369) | not run | 35.6 / 34.6 (954) / 52.6 (624) |

Where the one-row TP4 + EP4 step's 13.0 ms goes (the step trace, 64
steps, and the kernels from the profile of the sum-free step; a card runs
672 kernels a step):

| term | ms a step | how measured |
|---|---|---|
| host outside the graphs: inputs moved to the cards 0.1, the rank threads started 0.15 to 0.6 after the step, input buffers copied 0.2, graph launch 0.15 to 0.2, the join and the return 0.3 | about 1.0 | the trace's phases |
| the host nodes' round trips (the two engram lookups) | 0.04 | the trace, from the graph's timing events |
| kernels, a card's own, busy | about 2.1 | the profiler's kernel durations (3.2 µs a kernel) |
| gaps between a card's own nodes inside its graph, no peer and no host involved | about 7.6 | the sum-free graph's 9.7 ms less the kernels and the hops: 11 µs a kernel |
| the 89 sums, including the skew of the cards arriving at different times | 2.4 | the graph with one-shot sums (12.1) less without (9.7): 27 µs a sum, the template alone 11 |

The layer-sharded control shows the same per-kernel cost: 13.1 ms over
1,383 kernels across the four cards is 9.5 µs a kernel. So at one row
the step is paid per kernel, not per byte: TP4 cuts the bytes a card
reads four ways but every card then runs every layer's kernels (672
against about 350 a card under layer sharding), and the sums add 178
kernels and the skew on top. The planner's 6.2 ms prediction and this
workstream's 7 ms estimate both priced the bytes and the crossings and
carried the launch floor as 1.5 ms, which is what a card's kernels cost
in the floor's measurement, not what they cost in the step. What would
change the outcome is fewer kernels a card (the fused regions of hero
6's direction, the persistent executor), after which the same TP
machinery runs unchanged. At 33 different prompts TP4 + EP4 is already
1.75× the control (52.6 against 92.2 ms) because that step is paid per
byte of experts and the split pays; it still misses the bar by 1.6×.

Measured along the way and worth keeping: the ring path's sum (a ring
write per peer, the stream's flag waits, an aten sum) costs about 55 µs a
reduction in the step against the template's 27 (the skew is the same in
both; the template saves 28 µs a site, 2.5 ms a step at 89 sites); the
eager exchange on the CPU path races unless it publishes a copy (the
fork's all_reduce is in place); a torch profiler on a step whose sum
kernels spin on peer flags hangs the box's four cards (three at 100%,
one idle; killed by pid), so the profile is taken on the sum-free step;
Triton specializes the sum kernels on the pointers' dtype, so the Comm
primes them per dtype before any rank captures; the first TP capture
failed by running other ranks' expert pools through the peer path, which
allocates inside another card's graph.

**Hero 14, the all-reduce as a kernel template (Nov 2, measured on pod D,
four RTX PRO 6000 over PCIe, no NVLink).** `vxpu::allreduce(x, site)`
(ops.py; `allreduce.py`) is the tensor-parallel dense path's cross-card
sum: every card calls it on its partial, and every card gets the sum in
rank order (fp32 accumulation, rounded once), the same bits under either
form and as the reference, so the class is bit-identical. It runs with
no host and no allocation inside the capture: `allreduce.setup(devices,
sites, max_elems)` makes the Comm once (inboxes, flags and counters per
call site on every card), a writer raises a flag to 1 and the reader
resets it to 0 after consuming, so replays need no zeroing and no
sequence numbers. Two forms the table chooses between by rows:

| rows × 5120 bf16 | one-shot (push to every peer, sum locally; 2 kernels) | reduce-scatter + all-gather (each card sums a slice; 3 kernels) |
|---|---|---|
| 1 | 11.2 µs a reduction | 18.6 |
| 8 | 17.6 | 21.2 |
| 33 | 53.1 | 31.9 |
| 132 | 200 | 97 |
| 264 | 341 | 190 |

(a chain of 86 reductions with a GEMV between, one captured graph per
card replayed together, the GEMVs' time subtracted; 20 rounds). The
crossover is between 8 and 33 rows and the table says so
(`ALLREDUCE_SCATTER_MIN_ROWS` 33, `pkg/kernelplan/testdata/allreduce-rtx6000.json`,
written by `python -m vxpu.allreduce measure`). Per step of 86
reductions: 0.96 ms at one row, 1.5 at 8, 2.7 at 33, 8.4 at 132, 16.4 at
264. Three things learned on the way: Triton loads a kernel's module on
a card at its first launch there and that load blocks while a kernel is
spinning on that card, so a card already waiting for its peers froze the
host's launch of the peers' pushes (the Comm now primes every kernel on
every card at setup; captured graphs never meet it); the eager four-card
all-reduce costs about 245 µs whatever the payload, all of it host
launches across devices, so the sum is only usable captured; and the
fused form, the row GEMV's epilogue writing its partial straight into
the peers' inboxes (`gemv_rows(..., peers=)`, `gemv_allreduce`), saves
0.5 µs at one row (10.9 against 11.4) and loses 2× at 33 rows (107
against 53), because a GEMV program's tile is 16 columns wide and its
peer stores cross PCIe as 32-byte pieces where the push kernel's are
contiguous 8 KB chunks; the fusion would need the GEMV to write
row-contiguous tiles to the peers, which is the next thing to try if
the one-row step's 86 launches turn out to matter. The dense workstream
takes the op as is.

**Hero 14, the experts per card (Nov 2, measured on pod C).** Every
layer's 384 experts spread over the four cards, expert e on card e mod 4
(`dev/hero/ep_placement.py`; a popularity-balanced variant caps each card
at its equal share): 3,840 experts and 72.2 GB a card, placed in 35 s,
against today's 3,072 / 4,608 / 4,608 / 3,072. Measured imbalance of the
by-id split from hero 13's recorded router choices: the fullest card
carries 2.8 of 6 pairs at one row (share 0.47), 17 of 48 at eight rows
(0.36) and about a third of the pairs at 128 different rows (0.34; the
plan model had guessed 0.28); by distinct experts, the bytes term, 0.47,
0.34 and 0.28; the popularity-balanced split brings the pair share at
width to 0.29, fitted on the same traces, so an upper bound. On today's
layer-sharded ranges the spread placement hangs at the first replay after
a clean capture: each GPU then runs its range graph and a peer graph for
the other three ranges' pools (hero 6's mechanism) and the two never
complete a step, on separate streams and with the peer graph's arrival
counters keyed apart (`FP4_COUNTER_GRAPH`); two ranges with experts over
four cards hang the same way; the cause is open and the mechanism is not
the one the TP design needs. The pure-peer form runs: the dense layers on
one GPU, every layer's experts over the four cards, each layer fanning x,
the routing weights and ids to three peers and taking their per-pair
replies back (364 copies and 366 mailbox operations a step against 12).
Measured against today's layer sharding, same tree, pinned site table,
64 tokens:

| rows | prompts | experts over four cards | layer-sharded | experts per card a layer |
|---|---|---|---|---|
| 1 | one | 16.6 ms | 12.9 | 49.5 µs (1.5 pairs) against 93.4 (6 pairs) on the one card |
| 9 | one / same / nine | 22.6 / 22.1 / 26.4 | 21.6 / 17.4 / 34.6 | |
| 33 | one / same / 32 | 35.9 / 35.1 / 50.8 | 29.4 / 27.2 / 93.2 | 110 µs against 286 |
| 66 | one / same / 32 | 59.5 / 57.8 / 74.8 | 45.7 / 44.9 / 112.3 | |
| 132 | one / same / 32 | 103 / 101 / 117 | 79.9 / 76.1 / 140.7 | |

With a prompt per row the spread wins at every width from nine rows up
(34.6 to 26.4 ms, 93 to 51, 112 to 75, 141 to 117) and loses with one
prompt, where the expert bytes are small and the fan-out is the whole
cost: at one row the peer round trips add 3.7 ms where the spread saves
1.8. A card's expert kernels take 2.6× less time per step, not 4×, since
their per-launch and per-pair costs do not divide. The per-pair replies
are the pure-peer path's own cost at width: [rows, top-k, dim] fp32 is
4 MB a layer a peer at 33 rows, 10.7 ms of copies a step, where a summed
partial [rows, dim] is 0.7 MB; the all-reduce of the TP step carries the
partial. Memory: GPU 0 with the dense layers 85.8 GB at nine rows (the
ledger's number to the decimal, once it counted experts placed outside a
GPU's own range) rising to 97.1 at 132 rows; the other cards 69.7. For
the TP + EP step the engine now has `expert_ep` (plan
`parallel.expert_ep`, `VXPU_EXPERT_EP`): a GPU's expert node computes its
held pairs alone inside its own graph and hands on the partial sum, no
peer inboxes, nothing to the host; the reduce after the FFN adds the
GPUs' partials. Off by default; the tokens under it are wrong until the
dense layers are split too.

**Hero run 15's acceptance, agreed Nov 2 (evening): the league table.**
The user: a true hero run is #1 on the league table, everything proven at
once, because broken into pieces each looked small and was deferred.
Acceptance, measured the way the published stacks measure (requests
arriving, steady state over equal step counts, prose prompts, 256 new
tokens), on four RTX PRO 6000 Blackwell on PCIe: first place, not a
place on the table. The user: "hero 15 is not the league table, it is #1
on the league table. Do or do not, there is no try." So: single stream
with the drafter above the best published figure on these cards (the
vLLM fork's 252 tok/s on prose and 494 on code), 8 concurrent requests
above its 827 aggregate, 16 above its 1,150 (SGLang's 112 and 529 and
its 1,020 five-minute aggregate at 8 beside them). Met or not met on
each; if not, the shortfall by measured term and the direction that died. The
arithmetic that says it is reachable: 250 tok/s at 2.6 tokens a step is a
10.4 ms verify step; a TP4 + EP4 verify step is about 9 to 10 ms if the
plain step lands near 7. The pieces, all in parallel from the start, on
the TP4 + EP4 shape hero 14 finishes:
1. The persistent per-card executor: the step graph walked by one
   resident kernel per card, the per-layer sums a counter wait inside it,
   no per-kernel launch and no host between layers (today 662 kernels a
   card, about 1.3 ms of host hand-off and sums costing four times in the
   step what they cost alone); starts on the layer-sharded step, moves to
   the TP step.
2. The expert GEMM form: rows that share an expert as one matmul instead
   of a GEMV per (row, expert) pair (2.1 to 2.6 µs a pair beyond the bytes
   today), which the verify at 8 and 16 streams needs.
3. Speculation under TP: the DSpark block with the verify sized to the
   draft (an export per token-row count, the loop choosing the graph), the
   confidence policy, the drafter placed by the planner.
4. The serving harness: requests arriving at 1, 8 and 16 streams, a
   batch's rows prefilled together, steady-state windows, the prompt mix
   the published runs used; the planner choosing the shape per stream
   count and the memory ledger admitting what fits.
5. Continuous integration on a free pod: every piece merged as it lands,
   the three numbers re-measured each time, the table in this section.

**Hero 15, the serving harness (Nov 2).** The league table is a serving
measurement, so the run's number is only as good as the harness. Two
things changed. First, `dev/hero/serve.py` runs N streams in a closed
loop the way the published benchmarks do: each stream sends a prompt from
a prose set, receives 256 new tokens and at once sends its next, for
several rounds, through the engine's own continuous admission (a stream's
next prompt takes the row its last one freed), under whatever layout the
plan gives the engine; N is rows times batches in flight of the artifact
in use. It reports the steady-state window (from the moment every stream
has finished one request to the moment the first stream runs out of
prompts, so N streams are in flight throughout), the tokens emitted inside
it and the aggregate tokens a second over it, per-stream decode speed,
the time to first token (admission to first token) with the first round's
queueing wait apart, prefill times, step statistics per batch and the
drafter's counts; requests now carry a timestamp per emitted token for
it, and the same summary goes into the kernel record as the statistic
`serving`, so the planner's prediction is scored against it (`vxpu plan
serve` predicts the same shapes; `dev/hero/league.py` prints both beside
the published figures). The earlier harness's strict figure at 64 tokens
measured admission as much as decode (hero 12, hero 13); this one does
not quote a number until every stream is past its first request.

Second, a batch's prompts are prefilled together. The prefill program had
one row, so a 33-row batch was admitted one prompt at a time at about 100
ms a prompt alone and more under contention, 1.2 to 2 s a 33-row batch
and 8 s a 66-row one, as long as the batch's 64 decode steps; a prompt
alone reads about 150 distinct experts a layer, so 33 prompts one at a
time read the same experts 33 times. An artifact exported with
`--prefill-rows` carries a second prefill program, `prefill_rows.pt2`,
with a symbolic row count from two up to the artifact's rows (export
specializes a dimension that may be one, and refuses a dimension whose
floor is its ceiling, so two rows is a fixed shape), beside the one-row
program; a lone prompt keeps its own graph, a group of two or more takes
the many-row one (the first version had one program with a floor of two
and prefilled a lone prompt twice, which made every refill after the
burst 1.8 times slower, 233 ms against 132, and cost more than the burst
saved at 33 rows), and the engine admits every prompt due in one go: prompts of one length grouped up to the artifact's
prefill rows, one prefill per group, row i moved to its target row, the
decoding rows the prefill writes over set aside and put back, the
replicas refreshed once, the loop held once per admission round. The
one-row path is the same code with groups of one. The selftest's new
check prefills three prompts of one length together and two more later
while rows decode and gets the tokens of one-at-a-time admission; its
first version caught the engine restoring only row 0 of the two rows a
lone prompt's prefill writes.

**Hero 15 measured (Nov 2, 19:10 UTC).** The league table on the merged
tree (vxpu-big df310f3, the expert GEMM form, with this harness), four
RTX PRO 6000 layer-sharded, prose prompts, 256 new tokens, four rounds,
steady window, reference sampling off; aggregate and per-stream tokens a
second, the planner's figure for the same shape in brackets:

| streams | shape | plain | with the drafter | published |
|---|---|---|---|---|
| 1 | 1 row | 72.0 / 74.6, 13.4 ms a step [73] | 2 drafts 93.4 / 103.0, 22.2 ms, 2.19 tokens a step [95]; 4 drafts 93.2 / 100.9 [99] | vLLM fork + DSpark 250, SGLang 112 |
| 8 | 1 batch of 8 | 231.5 / 30.0, 31.8 ms [230] | | vLLM fork 814 to 827, SGLang 529 |
| 8 | 2 batches of 4 | 257.1 / 31.4, 24.8 and 27.4 ms [339] | | |
| 8 | 4 batches of 2 | 359.5 / 42.6, 19.6 ms [398] | 2 drafts 439.8 / 55.7, 33.3 ms, 1.75 tokens a row-step [405]; 4 drafts 397.3 / 51.1, 42.5 ms, 2.10 [430] | |
| 16 | 2 batches of 8 | 381.7 / 26.0, 33.6 ms [459] | | vLLM fork 1,150 |
| 16 | 4 batches of 4 | 522.5 / 33.5, 25.9 ms [583] | 4 drafts 487.2 / 30.4, 66.5 ms, 2.11 [583, drafter off]; 2 drafts not measured (the 4-row 3-token draft graph failed to capture: an inductor kernel allocated during capture) | |

The planner's pick is the measured best at every N (the drafter at one
stream, four batches of two with the drafter at 8, four batches of four
plain at 16); its figures run 10 to 20 percent high. Four batches of two
beat one batch of eight at 8 streams because a 2-row step takes 19.6 ms
against 31.8 for 8 rows and four are in flight; the same at 16. Nothing
reaches the published figures: 93 against 250 at one stream, 440 against
820 at 8, 523 against 1,150 at 16. Batched admission, one prompt at a
time against grouped, first-round queue p90 (the wait for the burst's
last row): 8 rows 1,070 to 587 ms; 66 rows 8,446 to 2,272 ms, and 33
rows 4,019 to 5,936 ms on the two-graph tree (16,224 with the one-graph
version), because prose prompts of 33 lengths make 10 to 15 groups and
each new token-row shape's Triton first-use compile lands in the queue;
the aggregate at 33 and 66 rows on the merged tree, one batch, reference
sampling on: 455.7 and 588.0 (289.8 and 354.5 before).
**Hero 15, speculation (Nov 2 to 3, measured on pods C and D).** Three
questions, each answered with numbers on the GEMM tree (vxpu-big a8769f8)
and then on the tensor-parallel tree (h15-tp-step 453703f, fork 84a5054).

*The drafter in flight at 8 and 16 streams* (serve.py, closed loop, 4
rounds of 256 tokens, prose, greedy): 8 streams as four batches of two
rows decode 367.5 tok/s plain, 429.7 with two drafts a step (acceptance
0.59) and 404.5 with four; 16 streams as four batches of four decode
533.4 plain, 503.6 with two drafts and 489.4 to 505.6 with four
(run-to-run spread 3%). The drafter helps at 8 streams and loses at 16,
where the drafted step at four rows and five token rows costs 2.6x the
plain step for 2.1x the tokens. The planner's predictions were 398/430
and 583/531. Fused bodies change nothing at these widths (530.8 against
533.4). The 16-stream two-draft case first died in the lowering's
measurement, a compiled candidate allocating inside the capture of its
replay graph; the measurement now runs every argument set once before
capturing and times a candidate eagerly when the capture still fails.

*The verify sized to the draft.* The native loop registers a captured
graph per shorter token-row count (`set_verify`) and runs, for the drafts
its rows issued, the shortest graph with room for them; the engine loads
the shorter decodes from `VXPU_VERIFY_ARTIFACTS` over the main decode's
weights and state. The first exports were wrong in a way only the
acceptance showed: a program exported for its own step sizes two rings
from that step (the window ring has window_size + window_slack slots, the
compressor ring compress_ratio + window_slack, window_slack = step_tokens
- 1), so a 2-token verify indexed the shared 5-token step's rings as if
they had three fewer slots, and its acceptance fell from 0.51 to 0.43 at
the same policy, which an exact verify cannot do. A shorter verify is now
exported with the step's slack (`--window-slack`) and with the drafter,
so its step writes the drafter's rings; the engine refuses one that
differs. Measured at one row on the layer-sharded tree (serve.py, one
stream): plain 13.4 ms, 74.6 tok/s; per-length fixed 1/2/3/4 drafts
20.8/22.1/23.7/25.4 ms = 83.4/99.6/95.4/98.0 tok/s; confidence 0.3 24.3
ms, 2.42 tokens a step, 104.4 tok/s; confidence 0.5 21.8 ms, 2.17,
102.9; the same policies padded through the five-row graph 25.1 ms / 97.0
and 24.6 / 93.3. The per-length verify is worth 8 to 10% at one row and loses 6
to 8% at 8 streams, where the acceptance matches padded (0.507 against
0.509 at confidence 0.3, 0.644 against 0.662 at 0.5) but the per-batch
step does not shrink (46.0 ms against 45.4, 41.0 against 41.4): 359.8
and 353.4 tok/s per-length against 388.4 and 376.7 padded (fixed two
drafts, 430, is the best 8-stream setting). The drafter on GPU 0 instead
of its layers' card costs a third under fixed drafts (265 against 405)
and nothing under the confidence policy (387 against 388).

*The drafter under TP4 + EP4*, the single-stream line. The four-rank
export never wrote the drafter's rings (the rank programs were exported
without the observe), which is why hero 14 estimated the drafted TP step;
now every rank's step writes them from the replicated activation after
the last sum, the engine binds the one draft program to rank 0's state
with its stages' experts placed whole on rank 0's GPU, loads the
per-length verifies as four-rank programs over the main step's slices,
and the host-driven TP loop runs the speculative step: draft on rank 0,
the policy, the shortest verify with room, the acceptance. Bringing it up
took six fixes, each measured by its own failure: the ring-write
segment's package build under torch's annotate context; per-length
graphs replacing the all-reduce Comm under captured graphs and sharing
the world's inboxes by index (a core dump); four ranks compiling the
sampler at once, and then the compile lock held across a collective wait
(a hang); and torch.export's process-wide tracing state, which corrupted
a rank building packages while another ran its eager step (mode-stack
assertions, a fake tensor among a segment's constants). The last is
answered by priming: every program is stepped and captured one rank at a
time with the world in solo mode before the ranks step together.
Measured on pod D with hero-spec2.py (one stream, 64 greedy tokens after
the warm-up), fold-3 tree: plain 10.2 ms, 97.6 tok/s; the one-token step
with the ring writes 10.4 to 10.5 ms; padded five-row verify: fixed 4
drafts 17.5 ms (drafter 2.94 + verify 14.6), 2.56 tokens a step, 146.2
tok/s; confidence 0.3 17.4 ms, 2.56, 146.9; confidence 0.5 17.4 ms,
2.00, 115.9; fixed 2 padded 17.3 ms, 2.21, 127.2; fixed 2 with the
per-length verify (the 3-row graph) 16.3 ms (drafter 2.95 + verify
13.3), 2.37, 145.7. On the fold-4 tree (h15-tp-step 453703f, fork 84a5054), the curve the
planner needs, padded, pod D (step ms / tokens a step / tok/s): plain
10.07 / 1 / 98.4; fixed 4 drafts 17.2 / 2.36 / 132; fixed 3 17.1 / 2.29 /
134; fixed 2 17.0 / 2.21 / 129; confidence 0.1 17.1 / 2.29 / 133; 0.2
17.1 / 2.29 / 133; 0.3 17.2 / 2.46 / 144; 0.5 17.0 / 1.94 / 114. The step
is the same 17 ms whatever the policy issues, because the padded verify
always runs five rows; the policy only moves the tokens a step, and the
acceptance of a 64-token run varies by 0.3 tokens a step between runs of
the same setting (1.36 to 1.67 accepted at fixed 4). The per-length
verify under TP is a wash: on pod C, same tree, confidence 0.3 decodes
18.8 ms and 2.29 tokens a step per-length (its verify 13.0 / 13.8 / 15.0
/ 16.0 ms at 2 / 3 / 4 / 5 rows) against 19.1 ms and 2.46 padded, 125
against 129 tok/s; the shorter graphs save their rows but switching
programs step to step in the host-driven loop costs about as much. Pod
C runs the five-row verify 2 ms slower than pod D (16.1 against 14.3)
and the plain step 0.3 slower (10.5 against 10.2): the pods are not
interchangeable at five rows. The tp-step workstream's expert K split
(65b302b, VXPU_FP4_GU_KS=2 VXPU_FP4_DN_KS=2) takes pod C's plain step
from 10.5 to 9.7 ms with identical tokens and leaves its five-row verify
where it was (16.09 to 16.18): the verify's row cost is not in the
direct expert kernels' K loop. The per-length run at confidence 0.5 on
that tree and pod gives the verify's cost by rows inside one process:
9.85 ms at one row (the plain step), 13.0 at two, 13.9 at three, 15.1 at
four, 16.0 at five. The first extra row costs 3.2 ms and each further
row about 1.0: the step leaves its one-row path at the second row, and
that jump, not the rows after it, is where a drafted step loses against
the plain one. Pod D after its restage, same case with the split, reads
9.43 / 12.13 / 12.57 / 13.41 / 13.79: the second row 2.7 ms, the rows
after it 0.4 to 0.8 (15.1 ms a step, 1.83 tokens, 125.4 tok/s against
114 padded). (The run itself: 16.1 ms a step, 1.83 tokens, 116.5 tok/s,
the policy issuing 1.43 drafts a step; padded confidence 0.5 on pod D
decodes 114 tok/s.) Fixed 2 per-length on the same tree and pod: 16.8 ms
(drafter 2.94, three-row verify 13.8), 2.21 tokens a step, 131.4 tok/s,
against 140.7 on pod D after its restage (15.7 ms: the three-row verify
12.76 with the split, 13.3 without on the fold-3 tree, so the split is
worth half a millisecond at three rows on that pod) and 145.7 on pod D's
fold-3 run, the pods' two milliseconds again. The best single-stream
number of the run is the same per-length path at confidence 0.3 on pod
D after the restage, K-split tree: 16.7 ms a step (drafter 2.94; the
verify 12.74 / 13.24 / 14.06 ms at three / four / five rows, taken 3 / 5 /
16 times in 24 steps), 3.54 drafts issued and 1.62 accepted a step, 2.67
tokens a step, 159.8 tok/s, against 143 to 144 padded on the same tree
and pod: on pod D the per-length verify pays 11%, where on pod C it was a
wash; pod C's slower short graphs are the difference. Against 252, 159.8
is 63%. The same case over the league table's prose prompt set (132
prompts, 256 new tokens each, one stream, pod D): 152.0 tok/s of decode
over 33,553 tokens in 13,765 steps, 2.44 tokens a step, mean step 16.0
ms, 2.87 drafts issued and 1.44 accepted a step; by prompt the decode
runs 129 to 192 tok/s. The conditional acceptance by draft position,
over every step that reached the position, is 0.72 / 0.68 / 0.67 / 0.67:
flat past the first, so a longer block would keep accepting at about
two thirds a position, and seven drafts would give about 2.8 tokens a
step at this acceptance. On pod D the same pair reads 14.23 ms with the split against 14.28
without for the five-row verify (fixed 4 padded, 17.2 ms a step both,
each rank's graph 13.56 against 13.60): the split is worth nothing to
the verify on either pod. The six-token block, every one of the drafter's
five drafts verified (a ring slack of 5, its own export family), padded,
fixed 5 drafts, on that tree and pod: 18.2 ms a step = drafter 2.92 + a
six-row verify 15.24, 2.56 tokens a step at acceptance 0.32, 141.7
tok/s: the sixth row costs 1.0 ms and buys a tenth of a token a step, a
wash against the five-token block. Seven drafts, the vLLM fork's
setting, need a second drafter pass from the target's hidden states of
the drafts, which this drafter does not have; the block is its limit. Against 252: at 2.56 tokens a step
the step must be 10.2 ms, today's plain step, so the drafter (2.9 ms,
byte floor well under 0.5) and the verify's 4.3 ms over the plain step
(expert bytes at this placement plus launches and sums growing with
rows) must both go, or the tokens a step must rise (the vLLM fork's 252
rests on 7 drafts with adaptive verification; this drafter's block is 5).
The drafter's call, accounted on the fold-4 tree (pod C, 50 calls alone,
then a profile of 20): 2.69 ms a call, of which 2.40 ms of GPU work in 257
kernels (the dense GEMVs 1.23 ms in 19 launches of gemv_rows and 0.25 in
five cuBLAS calls, the three stages' fp4 experts 0.35, the swiglu 0.07,
five argmax selects 0.13, small aten kernels 0.04) and 0.25 ms of host
(graph launches 0.18, the token copies 0.05, the output sync 0.02). Its
bytes explain it: the drafter's weights are 3.9 GB in the binding, of
which 1.32 GB is its own bf16 head ([129,280, 5,120], read whole on every
call for the full logits its Markov head draws the five drafts from),
1.32 GB its own embedding (a row lookup, not a read of the table) and
1.27 GB the three stages' fp8 attention and dense projections; 2.6 GB a
call is 1.6 ms at the card's 1.6 TB/s, so the 2.4 ms of GPU time is at
67% of the byte floor, and the levers are bytes, not launches: the head
in fp8 or fp4 (0.4 to 0.6 ms), the stages' GEMVs at their floor (0.4),
then the launch count. The block is not a lever: one pass through the
stages and the head gives the logits, and the five drafts are five cheap
Markov-head steps on them. The head in fp8, the drafter's own copy of the
target's bf16 head quantized at load as the dense fp8 linears are (one
power-of-two scale per 32x32 block; export --draft-head fp8), takes the
call from 2.69 to 2.26 ms alone (its GEMV 1.23 to 0.80 ms of gemv_rows)
and from 2.96 to 2.51 inside the step on pod C, with the drafts as good
within one prompt's noise (confidence 0.3 padded: 2.37 tokens a step
against 2.46, 127 against 129 tok/s; per-length fixed 2: the same 34 of
58 drafts accepted and the same first difference from the reference as
the bf16 head, the step 16.8 to 16.3 ms, 131 to 135 tok/s; per-length confidence
0.3 on pod C: the same 39 of 85 accepted as pod D's bf16 run of the
case, 18.4 ms, 2.67 tokens a step, 147 tok/s against 125 with the bf16
head on that pod). The drafter sharded across the four ranks like the target (a draft program per rank,
export --draft-sharded; stages and head a quarter each, the logits
gathered, every rank drafting the same tokens) reads a quarter of the
bytes per rank (gemv_rows 0.39 ms a rank) and still costs 2.34 ms a
call on pod D, because the Markov head's embedding and head were parallel
modules too and each of its five steps gathered and summed across the
ranks: about twenty exchanges a call. The fix is those two tables whole
on every rank (33 MB each). Even so the sharded drafter in the loop is
the best single-prompt figure of the run: confidence 0.3 per-length on
pod D, 16.1 ms a step (drafter 2.32; the verify 9.45 / 12.2 / 12.7 / 13.3
/ 14.0 ms at one to five rows), 3.17 drafts issued and 1.67 accepted,
2.67 tokens a step, 169.4 tok/s, with the expert-parallel placement
unchanged (the drafter's stages' experts a quarter per card like the
target's, GPU 0 no longer carrying them whole). On the prompt set (132
prose prompts, 256 tokens each, one stream, pod D) the sharded drafter
gives 158.6 tok/s (152.0 with the single-rank drafter): 15.38 ms a step,
2.88 drafts issued and 1.44 accepted, 2.44 tokens a step, conditional
acceptance by position 0.72 / 0.68 / 0.66 / 0.65. The Markov head's
tables whole on every rank remove the exchanges (pushes 48 to 28 a
call) and lose anyway, 2.65 ms a call on pod C against 2.63 for the
parallel form on the same pod: the head multiplies the previous token's
embedding against every vocab row for each drafted position (a bias over
the whole vocabulary, five GEMVs over a [129280, 256] table a call), and
the replicated table in ParallelHead's fp32 is 132 MB, read five times
at the card's bandwidth (the two reductions run at 1.5 TB/s). The
checkpoint holds that table in bf16; kept so and multiplied into an fp32
reduction the read halves (0.22 ms a call per rank). Open variant, not
built: the bias restricted to the top K base logits of each position (K
= 256: 128 KB of head rows a step), about 0.2 ms a call saved, with the
bound L_K + |embed| max_v |H[v]| <= best as the exactness test per step;
when the bound fails the draft can differ from the exact one (the verify
still decides the token), so it is a planner-grade choice to be measured
by its bound-miss rate on the prompt set, not a default. The same case runs 18.45 ms
a step on pod C and 16.1 on pod D with identical drafts (one-row verify
9.69 against 9.45, five rows 15.96 against 14.0, drafter 2.63 against
2.32): this note compares within a pod only. Seven drafts, measured and
parked (pod D, one prompt, the eight-row step with slack 7 and the
drafter drafting a second block in sequence from its own states, export
--draft-passes 2): the call 4.36 ms for ten drafts (2.34 for one block),
1,749 kernels over the four ranks against 1,045, the second block
repeating the stages, the head and five Markov steps; fixed 7 padded
20.8 ms a step = drafter 4.32 + eight-row verify 16.46, 1.42 accepted
of 7, 2.42 tokens a step, 118.7 tok/s, against fixed 4 on the five-row
step on the same pod and prompt, 16.3 ms, 1.37 accepted, 2.37 tokens a
step, 146.3 tok/s. Conditional acceptance by position at seven drafts
0.654 / 0.588 / 0.5 / 0.6 / 0.333 / 1.0 / 0.0 (positions six and seven
from one and two at-risk steps). Positions five to seven bought 0.05
accepted tokens a step for 4.5 ms; fixed 5 on the same eight-row step
gives the same 1.42 accepted at 21.0 ms, so the second block's drafts
are almost never reached. The eight-row verify costs 2.5 ms more than
the five-row one (16.46 against 14.0) and flips a near tie at token 4
of the reference where the five-row cases first differ at 34 (the row
count changes the GEMV numerics; the text after the tie is the same).
Seven drafts pay only once the drafter is cheap and the acceptance past
position four is above a half; the adaptive controller (the vLLM fork's
rule: floor(mean accepted + 1.5), down at once, up by one a window of
16) stays as the policy because it costs nothing and settles at two or
three. The league table's own measurement, reproduced (Nov 3, 03:55 UTC,
pod D): their prompts (prose: one fixed hash-map prompt; code: the
binary_search task), the chat template with thinking off (our prompt
token counts equal their server's, 32 and 59), greedy, exactly 400
tokens with EOS ignored, their 32-token warm-up first, and their
formula, 399 tokens over the time from the first token to the last
(prefill and the first token excluded). Sharded drafter with the bf16
Markov table, per-length verify, confidence 0.3 up to four drafts:
prose 176.2 tok/s (140 steps, 2.857 tokens a step, 16.18 ms a step,
drafter 2.26, 3.58 issued and 1.85 accepted, conditional acceptance
0.727 / 0.794 / 0.722 / 0.644; their published 252 at about 2.9 to 3.0
tokens a step, so the acceptance matches and the gap is the step, 16.2
ms against their 11.7); code 276.8 tok/s (86 steps, 4.651 tokens a step
of a five-token maximum, 16.76 ms a step, 3.99 issued and 3.64 accepted,
conditional acceptance 0.977 / 0.952 / 0.95 / 0.961; their 494 at 6.1
to 6.3 tokens a step with seven drafts, which the eight-row step gives
us room for). The 132-prompt prose set at 256 tokens gives 158.6 on the
same artifact: their single prose prompt accepts 1.85 a step where the
set accepts 1.44. The direct expert kernels up to 48 pairs
(VXPU_FP4_DIRECT_MAX=48, tp-step 59e1936, measured there at 5 rows of
different prompts 15.1 to 13.9 ms) on the same pair, same pod, same
artifact and policy: prose 176.2 to 188.1 tok/s, code 276.8 to 274.2.
The step moved by at most 0.2 ms (prose 16.18 to 16.07, the five-row
verify 14.22 to 14.33; code 16.76 to 16.54, 14.52 to 14.38): one prompt
gives 5 x 8 = 40 pairs of one row each, where the direct kernels and the
grouped GEMM cost the same. The prose gain is the drafts: the two kernel
forms round differently, the text diverges, and the second text accepts
2.05 a step (0.78 / 0.776 / 0.767 / 0.765) against 1.85. On one prompt
that swing is luck, not a result; the switch stays on for the batched
shapes it was measured at, and both numbers stand in the table. The drafter's call, kernel by kernel (the trace of one
call on one rank of the sharded drafter with the bf16 Markov table, pod
C): 202 kernels, 2.15 ms from first to last, 1.72 busy. The first push
and sum wait 135 us for the ranks to arrive (the four rank threads each
launch a 200-node graph; the skew lands in the first exchange). Each
stage: the front's ten small kernels; the attention's three GEMVs at
about 10 us, fifteen small kernels (norm, the RoPE gather, quant, the
ring concatenation, three index kernels, the sparse attention and its
combine), two output GEMVs, then push 21 + sum 18 us; the FFN front,
six routing kernels, the fp4 experts (gate/up 68, down 29 us, byte
bound), the shared expert's two GEMVs, push 18 + mixed sum 7. The head:
the bf16 quarter 222 us at the floor and three ring writes (the logits
gather) 77 us. The Markov head five times: the table GEMV 44 us (66 MB,
at the floor) and the add+argmax 27 to 32 us, inductor's one program per
row over 129,280 floats where the library reduction takes a few. Per
rank: exchanges 0.41 ms, experts 0.29, head 0.22, Markov 0.40,
attention GEMVs 0.15, small kernels 0.2. Measured in turn on pod C (the
call alone, four ranks): the fp8 head per rank 2.43 to 2.31 (drafts
identical, the block scales sliced per rank as the weight is); the
attention's query, key and output paths placed as the main attention's
(attn_q, window_kv, attn_core, attn_out_partial with the ranks' sum
mixing the stream; the drafter keeps its own selection, the ring's
window and the whole block for every query, where the main attention's
is causal) 2.43 to 2.29 with 685 kernels over the four ranks in place
of 817; the greedy pick as an opaque op (vxpu::argmax_last) exported
with both: the call 2.25 ms (the argmax now the library's single-block
reduction at 22 us a step; its place is the table GEMV's epilogue). On
the 64-token check the placed paths' drafts differ from the published
path's (rounding-class kernels; the output then meets a different near
tie), and on the league pair over 400 tokens the acceptance holds: with
the switch on, prose 190.6 tok/s at 15.86 ms a step (the plain sharded
drafter 188.1 at 16.07), 2.02 accepted a step against 2.05; code 284.2
at 16.32 (274.2 at 16.54), 3.64 accepted against 3.53; the drafter in
the loop 1.95 ms from 2.21. Across the 132-prompt prose set (256
tokens each, pod C, the same tree with the switch and the single
launcher) the two drafters accept the same: 1.459 a step for the full
stack against 1.443 for the plain sharded drafter, conditional
acceptance 0.722 / 0.679 / 0.665 / 0.674 against 0.723 / 0.672 / 0.669
/ 0.675, 151.2 against 148.6 tok/s on that pod; the placed paths change
which drafts come at near ties, not how many are right. tp-step's head
merged (2960f2e: the reduce-scatter form of the mixed sum, the ranks'
inputs moved on the main thread after 79edc32 had deadlocked the
sharded drafter's rank-0 inputs against a spinning sum; then 48eb1f6,
the single launcher) runs every program here without a stop; on the
league pair at one row it costs 0.1 ms a step against the pre-merge
protocol (186.5 / 272.3 against 188.1 / 274.2, inside the band), and the
single launcher takes 0.12 to 0.2 ms off the drafter's call in the loop
(2.22 to 2.01 on pod C, tokens identical). The eight-row step on the league pair (the
two-pass drafter with the bf16 table, per-length verifies to seven rows,
the adaptive controller to seven, the switch on, pod D): code 277.7
tok/s, 5.88 tokens a step (6.24 issued, 4.91 accepted) at 21.13 ms a
step, the drafter 3.85 and the verify 16.79 at seven rows (52 of 68
steps) and 18.09 at eight; prose 142.9, the controller settling at two
to three drafts (112 of 166 steps at three rows) and paying the
two-pass drafter 3.88 ms for nothing. With confidence 0.3 to seven in
place of the controller: code 292.6 tok/s, the best code number of the
run, 6.25 tokens a step (6.61 issued, 5.3 accepted) at 21.3 ms (the
drafter 3.78, the verify 17.88 at eight rows for 50 of 64 steps);
prose 167.8 at 3.33 tokens a step and 19.81 ms. The tokens a step the
seven drafts promised came on code and the step ate most of them: the
five-token step gives 276.8 to 284.2 on the same prompt at 16.3 ms. The conditional
acceptance by position at seven on code says where the drafts are
worth having: 0.985 / 0.955 / 0.922 / 0.983 / 0.948 for the first
block's five, then 0.509 / 0.429 for the second block's two, drafted
from the drafter's own states (with confidence 0.3: 0.984 / 0.968 /
0.984 / 0.967 / 0.948, then 0.596 / 0.56). So the shape for code is one pass of
five drafts on a six-row step (the per-length verifies to five rows),
not two passes to seven: at 0.95 a position, about 5.6 tokens a step at
the one-pass drafter's 2 ms and a six-row verify near 14.8, which is
the next export, measured on pod D: code 315.9 tok/s, the best code
number of the run, 5.479 tokens a step (4.97 issued, 4.51 accepted;
0.986 / 0.972 / 0.929 / 0.985 / 0.906) at 17.3 ms a step, the drafter
1.71 and the six-row verify 15.52 (the sixth row costs 1.3 ms over the
fifth, more than rows three to five did); prose 187.4 at 3.03 a step and
16.13 ms, against 190.6 on the five-token step, so prose stays on five
and code moves to six. The multi-stream rows, their way (N streams started
together, the prose prompt with " (stream i/N)" appended per stream,
code tasks 1 to N, 400 tokens each, aggregate = the sum of (completion
- 1) over the latest last token minus the earliest first token; one
stream per row of a batch-N tensor-parallel artifact, the rows in
lockstep): plain, 8 streams, pod D, 453.3 prose / 452.8 code tok/s
aggregate, 56.7 a stream, 17.6 ms a step at 8 rows (pod C 391.1 / 389.3
at 20.4); their 822 / 1,188 is a drafted step. At 8 rows the step reads
up to 64 of 128 experts a layer, half the bank, where one row reads 8.
The first eager step at batch 8 stopped once in three runs (ranks 0, 2
and 3 waiting at site 0 round 0 on rank 1's push, rank 1 finished; 3600
s; the merged tree on pod D) and ran twice (the pre-merge tree on pod
D, the merged tree on pod C): a race in the first step, not a property
of a tree. One race the code allowed is closed: the Comm grew when a
later program asked for more than the one set up, and a graph captured
against the earlier Comm would push into boxes nobody waited on; now
the prime sizes the Comm once before any capture for every program (the
most all-reduce sites any has, rows x the largest step's tokens x dim
elements, both dtypes, the hc width), a capture marks it bound, setup
raises instead of growing a bound Comm, and the sums assert their size
(4198151). The speculative loop takes the artifact's rows (8341a16): the
prompts through the one-token verify in lockstep, the drafter after
every row's next token at its own position, the policy per row with the
controller's depth shared, the shortest verify with room for the most
any row issued, acceptance per row. With it, the drafted 8-stream row
on pod C (the full drafter stack at batch 8, per-length verifies at
batch 8, confidence 0.3 to four): code 516.1 tok/s aggregate (64.5 a
stream), prose 374.0 (46.8), against plain 389 / 391 on that pod and
their 1,188 / 822. The step: 62.5 ms = drafter 6.46 (2.0 at one row;
its stages see 8 x 5 token rows) + the five-row verify 56.24 (every
step: the rows share one program, so the verify takes the most any row
issued, and some row always issues four) on code, 4.04 tokens a step a
row at 0.964 / 0.936 / 0.935 / 0.918; prose 60.1 ms, 2.82 a row at
0.786 / 0.735 / 0.709 / 0.699. Forty token rows through the dense
layers a step, 320 expert pairs a layer, so the whole bank is read every
step where plain 8 rows read half. The term is the verify's cost per
token row: on this pod the plain 8-row step is 20.4 ms, 16 token rows
31.8, 24 rows 42.4, 32 rows 40.4, 40 rows 55, about 1.2 ms a token row
past eight, where byte-bound expert reads would add about 8; the
many-row step is the tp-step shape. At that cost the acceptance says
code is flat near 500 to 550 at any depth and prose gains from a depth
of one or two (about 450); the adaptive controller, its depth shared
across the rows, is the policy at 8 streams. The drafter's call at
batch 8 alone (pod D, every row the same token): 3.51 ms against 1.7 to
2.0 at one row; over the four ranks the logits gather's ring writes
2.61 ms (the head is vocab-sharded, so the Markov loop gathers [8 rows
x 5, vocab / 4] fp32, 20 MB a rank a call, where one row gathered 2.6),
the pushes 1.82 and sums 0.91 carrying the same, the Markov add and
argmax 1.9 over 40 positions, the experts 1.23 (0.77 at one row), the
stage GEMVs 1.28 (1.0 at one row: weight-bound, the rows ride free). The
rows cost exchanges and Markov positions, not GEMV bytes; the
vocab-sharded Markov head with a partial argmax and a tiny exchange a
step, which was not worth building at one row, takes about 1.3 ms a
rank out at batch 8. The five-row verify alone at batch 8 with every
row the same token is 28.5 ms on pod D where the league run's eight
different prompts took 54 to 56 on pod C: the rows of different prompts
read more of the bank, the plain 40- and 80-row steps with one prompt
and with different prompts are the split tp-step owns: on pod D the
plain step is 17.6 ms at 8 rows, 24.6 / 31.6 at 40 rows (one prompt in
every row / a different prompt per row) and 38.1 / 47.1 at 80, so 0.22
to 0.44 ms a row past eight, with the different-prompt premium 7 ms at
40 rows and 9 at 80 (2,092 tok/s aggregate at 80 rows with one prompt,
1,695 with eighty). The drafted 16-stream row (pod C, the full drafter
stack at batch 16, per-length verifies at batch 16, confidence 0.3 to
four): prose 821.0 tok/s aggregate (51.3 a stream), code 1,073.5 (67.1 a
stream), against plain 567 / 530 on that pod and their 1,150 / 1,669;
the step 54.4 ms on prose = drafter 6.0 + five-row verify 50.0 at 80
token rows, 2.80 tokens a step a row at 0.77 / 0.73 / 0.73 / 0.72; code
57.7 ms with the verify 52.6, 3.88 a row at 0.95 / 0.95 / 0.94 / 0.93.
The verify at 80 token rows costs what the earlier 8-stream run's did at
40 (54 to 56 on the same pod), so the many-row verify is nearly flat
from 40 to 80 rows, or that run carried something extra; the 8-stream
runs on this tree decided it: the drafted 8-stream row on pod D (tree10,
the same artifact and policy) is 607.7 prose / 820.8 code tok/s
aggregate (76.0 / 102.6 a stream) at 36.99 / 39.28 ms a step, the
drafter 3.66 and the five-row verify 33.97 / 35.91 at 40 token rows,
2.82 / 4.04 tokens a step a row; the 60 ms step of the morning run was
not the shape. The attribution at 40 token rows on that pod: the plain
step at 40 different rows 31.6 ms, the five-row verify at batch 8 with
eight different prompts 34.0 to 35.9, so the verify path (the
per-length programs, every row padded to the deepest issue, the ring
writes, the verify's extra inputs) costs 2.4 to 4.3 ms at this shape and
the drafter 3.7; the plain step against a floor near 17 is the tp-step
shape. The adaptive controller at 8 streams (pod C, the same tree):
prose 630.8 aggregate at 32.2 ms a step, the depth shared across the
rows settling at three to four (48 steps on the three-row verify at
26.2 ms, 101 on the four-row at 30.2), 2.7 issued and 1.64 accepted a
row, 2.55 tokens a step a row; code 834.8 at 39.0 ms, the depth at four
(96 of 98 steps at five rows), 4.08 a row. Against confidence 0.3 at
four on pod D (607.7 / 820.8) the controller buys 4 percent on prose by
shedding a row and loses nothing on code. A fixed depth of one (two-row
verify, 16 token rows, 25.4 / 26.7 ms a step) gives 551.8 / 580.4: the
rows are cheaper but 1.75 / 1.94 tokens a row do not pay for them, so
at 8 streams the controller's three to four is the policy (pod D: 630.4
/ 832.8). The drafted 16-stream row on pod D, rerun on the tree whose
prime ends with every flag zeroed: 822.8 / 1,074.5 aggregate at 54.3 /
57.7 ms, the pod C numbers to the digit. The vocab-sharded Markov head
(ModelArgs.dspark_markov_sharded, export --draft-markov sharded: the
head's logits stay the rank's vocab slice, each rank biases and chooses
over its quarter, the ranks exchange a (value, index) pair a position
padded to four floats, the first rank with the largest value wins as
torch.argmax over the gathered vector would) drafts token for token as
the gathered form (42 of 82, first difference 44) and costs 1.68 ms in
the loop at one row against 1.73, where the pair exchange costs what
the gather saved; at batch 8 the call is 2.44 ms against 3.51, the 20 MB
a rank of gathered logits gone and the pair exchange's sums 1.24 ms
over the four ranks what remains. tp-step's fused push (VXPU_FUSED_PUSH)
on the drafted case: tokens identical, the step 16.09 to 15.71. The
first-step stop had one more cause after the Comm's sizing: the prime
runs each rank's eager step and capture alone in solo mode, and those
steps push into the peers' inboxes and raise flags nobody consumes, so a
stale round-0 flag put a card a round ahead of its peer in the first
real step (the drafted 16-stream case, rank 2's site-0 pushes missing
on every peer, all four ranks waiting); the prime now zeroes every Comm
flag and counter and every ring flag before handing over (d11c091,
d49dcfd), and eight loads since ran clean. A third class showed later
on pod C (the sharded drafter at 8 streams on tp-step's merged head,
after a 2,126 s prime): three ranks in the step's finish with their
GPUs spinning on flags and the fourth, GPU idle, inside Triton's
CompiledKernel destructor as it entered its first step, read by py-spy
on the process; the prime's solo programs leave compiled kernels to
the collector, and their destructors (a module unload) ran on one
rank's thread at the moment of its first step while the peers waited on
its pushes for good. The prime now has every rank collect garbage and
synchronize its device on its own thread, with nothing in flight,
before handing over. So the first-step stops are one list of three,
each closed at the prime: the Comm sized once and never grown, every
flag and counter zeroed, every rank's garbage collected. The first step
keeps a long allowance for its compile (the longest seen 44 minutes on
a new tree); it is 3,600 s in the chains now, since the 7,200 s default
let the third class sit silent for two hours where py-spy read it in a
second. A fourth lesson is about what the prime captures: it captures
each rank alone in solo mode and that captured graph is the one
replayed, so a node's capture must take the real path whether or not
the world is solo, as the collectives' captures do; the embedding row
fetch's first capture took its solo branch (the rank's own lookup, zero
rows for the ids other ranks own), the drafts were wrong, and the entry
read 723 against 1,048 tok/s at 16 streams while the account harness,
which times whatever graph it is given, read a clean 0.31 ms saving.
The shards are now registered on the world when each rank's graph is
built, before any prime, and every rank's capture reads all four from
that registry. The rule that follows: any new drafter or verify
artifact runs the drafted gate before any account or width entry uses
it, at the head of the chain, and the chain stops on a mismatch
(gate.py beside the chains): its accepted tokens a step within 10
percent of the reference artifact's run of the same case on the same
pod (a wrong draft leaves the text to the main model and collapses the
acceptance), its text identical to a second run of itself under the
other setting of whatever is being A/B'd (tree against tree, push form
against push form, fetch form against fetch form), the acceptance per
draft position printed, and the per-step texts kept. Identical text to
the reference is required only between runs with the same sequence of
verify programs, because each row count is a numerics class of its
own (its kernel variants and sum order): on one tree the five-row
program run every step differs from the one-to-five-row mix at
character 296 of the prose answer, and the eight-row program differs
at 340 whichever push form it runs. The class is wider than the
program sequence: it is every knob the kernel digest hashes. The
prose text of the one-stream runs split into two groups on Nov 4
along the clock, not the tree or the pod, and two hours of
file comparison (packages, fork copies, trees, venv, all identical)
ended at the chain scripts: every chain up to D46 carried
VXPU_FP4_DIRECT_MAX=48 in its shared knobs, so the five-row verify's
30 expert pairs ran the direct form; the chains rewritten after a
scratch directory was lost dropped it, the default limit of 8 runs the
scheduled form at 30 pairs, another summation order moved the prose at
character 352 (code's margins held), and the digest in the package key
is why the same tree recompiled seven segments on the same pod. The
direct form is also 0.4 ms a step faster there (14.54 against 14.94).
Rules: a chain states its full knob set per run, never through a
shared variable a rewrite can lose; the league result line carries the
knobs in effect and the kernel digest; the gate compares texts only
within one digest. A fifth lesson is about flags between cards: the embedding
fetch's push form first raised its per-peer flags with a stream write
to the peer's flag address, and a step stopped after the prime on two
pods with the same signature (three cards spinning at their first
collective, rank 0 idle in that wait) where other runs of the same code
had passed hundreds of steps; a race, not a topology. The collectives
raise their flags from inside the ring write's kernel, a device store
through the peer mapping after a fence, and the push form now does the
same over a four-byte pair per peer. The rule: a flag between cards is
raised from inside a kernel after a fence, never by a stream write to
the peer's address. The pod C figures before its
restage at 09:05 carry 2 to 3 ms a step of that node; the new pod C
runs the same case at pod D's speed. The fork's adaptive verification
at concurrency (its METHOD.md section 3 and adaptive_verification.md)
scores every (request, position) draft slot by its survival probability,
the running product of the request's per-position confidences, verifies
the highest-scoring slots across all requests until a global budget is
spent, and sets the budget from a step-cost model profiled at startup
to maximize expected accepted tokens a second; its step is a flat batch
of tokens, so a slot not taken is a token row not run. Ours is
rectangular: the verify program takes the same row count for every
request, a shorter request is padded to the deepest, and a rejected
draft costs nothing an unissued slot would not, so the choice collapses
to one number a step, the depth. The budget policy (speculation.BUDGET,
"policy": "budget") takes, each step, the depth D with a verify program
at D + 1 rows that maximizes the expected tokens a step, one a live row
plus each row's survival probabilities over its first D drafts, over
the step's cost, the drafter's measured call plus the verify program's
measured time at D + 1 rows, each program tried once to seed the costs
and the running medians after; every live row issues to D. It is the
window controller with the confidence head's forecast in place of the
last window's acceptances and the measured step costs in place of a
fixed target, and it is measured at 8 and 16 streams beside both. On
tp-step's head merged (e704f6a: the captured graphs launched from the
rank threads at once, the attention partials pushed as bf16 without the
cast, the fused push off), the five-token step's league pair on pod D
with the class off: prose 193.1 tok/s at 15.66 ms a step (drafter 2.03,
the five-row verify 13.95 for 102 of 132 steps, 3.03 tokens a step),
code 288.9 at 16.06 (the verify 14.05, 4.651 a step), against 190.6 /
284.2 on the tree before it; the same tokens. The bf16 push class
follows on the same tree for its default at width: with the class, the
FFN sites' fp32 partials cross PCIe as bf16 and each rank's partial is
rounded to bf16 before the fp32 sum; the step is 0.3 ms faster on prose
(15.36) and 0.2 on code (15.84), and the tokens move at a near tie: the
prose texts agree for 351 characters, about 80 tokens, and diverge,
giving 182.9 tok/s at 2.817 tokens a step on the second text against
193.1 at 3.03 on the first; code agrees through the sample and gives
296.3 at 4.706 against 288.9 at 4.651. On one prompt that is the same
swing the direct-expert switch gave the other way; the rate comes from
eight prompts at 8 streams and the 132-prompt line with and without
the class, both queued, and until they say otherwise the class is a
recorded variant. At 8 streams (pod C, eight prompts each) the class
takes 0.8 ms off the prose step (35.14 to 34.30) and 1.3 off code
(37.31 to 35.97), and the acceptance over the eight prompts falls 6
percent on prose (1.96 to 1.84 a row a step; 639.6 to 604.3 aggregate)
and rises 2 percent on code (3.47 to 3.55; 864.3 to 944.1), the same
directions as at one stream, on texts that again diverge; two pairs in
one direction on two prompt sets is more than one text's luck, and the
132-prompt line decided: with the class, 1.436 accepted a step against
1.459 over 132 prompts (0.717 / 0.679 / 0.658 / 0.651 against 0.722 /
0.679 / 0.665 / 0.674), the step 14.66 against 14.96 ms, 166.0 against
164.2 tok/s; a drift of 1.6 percent for 2 percent of step at one row,
a net loss on prose at 8 streams and a gain on code that grows with
the rows. So the class is a numerics variant the planner picks per
workload, and one rule covers both push forms, since the pull-form sum
puts bf16 on the wire by construction and carries the same class. The
16-stream pair (pod C, merged head, c0.3, class off to on in one chain)
settled the pick at width: prose 883.6 to 944.1 tok/s aggregate, the
step 50.53 to 47.29 ms, the five-row verify at 80 token rows 46.06 to
42.53, tokens a row identical (2.797); code 1,147.0 to 1,231.8, the
step 54.04 to 51.83, the verify 48.68 to 46.33, tokens a row 3.883 to
4.0. The class's saving per step grows with the rows (0.2 ms at one,
0.8 to 1.3 at 40, 2.4 to 3.5 at 80), the wire-bytes term's shape, and at
80 rows the prose acceptance did not move. At one stream the six-row
code entry reads 312.3 to 315.2 with the class (same tokens, within the
band) and the prose 185.5 to 169.1 (1.78 accepted a step against 2.04).
So the pick is: on for both workloads at 16 streams and for code at 8,
off for prose at 8 and for both at one stream; the entries at width
carry the class in their names. The same chain put the drafter call at
2.07 to 2.12 ms a call at one row on the merged head against 1.71 on the
tree before tp-step's launcher, thin-push and fused-push changes, class
off and on alike and with the verify programs at their earlier times:
a regression of the drafter's small four-rank program, 2 percent of the
one-stream step, under bisection by env switch on tp-step's side. The
vocab-sharded Markov drafter at 8 streams (pod C, same head, c0.3):
prose 662.6 at 33.93 ms a step against the gathered form's 639.6 at
35.14, code 890.5 at 36.21 against 864.3 at 37.31, the same tokens a
row; the 1.1 to 1.2 ms a step is the drafter call's 3.51 to 2.44 at
batch 8; with the window controller instead of c0.3 the sharded drafter
reads 679.4 / 901.5 at 29.92 / 36.13 ms a step (2.548 / 4.082 tokens a
step a row), the 8-stream prose entry of record (class off), measured
on one pod against the gathered drafter's 639.6 in the same chain. The
budget policy on top of the class at 16 streams (pod D,
same head): prose 1,022.1 tok/s aggregate at 39.53 ms a step, the
verify chosen at two / three / four / five rows for 14 / 41 / 88 / 15 of
158 steps, 2.532 tokens a step a row against c0.3's 2.797 at 47.29 with
the class alone; code 1,270.2 at 51.82, 91 of 97 steps at five rows,
4.124 tokens a step a row. Against the budget policy alone on the same
pod (896.2 / 1,177.8) that is 14 and 7.8 percent: the class's saving at
80 token rows, and on prose the policy moving to depth three once the
step is cheaper. The 16-stream row stands at 1,022.1 / 1,270.2 with
both, 89 and 76 percent of the league's 1,150 / 1,669. At 8 streams
the policy adds nothing on code: budget plus class 928.0 at 35.83 ms
and 4.167 tokens a step a row against the class alone with c0.3 at
944.1 and 4.255, within 2 percent on differing texts; on code at 8 both
sit at depth four, so the 8-stream code entry stays 944.1. The sharded
Markov drafter at 16 streams (pod C, same head, class off): with c0.3,
prose 928.2 at 48.09 ms a step against the gathered drafter's 883.6 at
50.53, code 1,202.7 at 51.53 against 1,147.0 at 54.04, the same tokens a
row and the drafter call 3.67 against 6.11 ms at 16 rows, which is the
whole 5 percent; with the window controller, prose 978.2 at 39.79
(depth three for 102 of 164 steps) and code 1,195.9 at 51.83. The
sharded drafter with the budget policy and the class, the full stack at
16 streams on the same pod: prose 1,033.7 at 37.20 ms a step, the
verify at two / three / four / five rows for 24 / 30 / 105 / 7 of 166
steps, 2.41 tokens a step a row, the drafter call 3.40 ms; code 1,322.3
at 48.77, 90 of 99 steps at five rows, 4.04 tokens a step a row. The
chain on one pod reads 883.6 / 1,147.0 (gathered drafter, c0.3), 944.1 /
1,231.8 (the class), 928.2 / 1,202.7 (the sharded drafter alone),
1,033.7 / 1,322.3 (all three), 90 and 79 percent of the league's 16-stream
bars. The gathered drafter with the budget policy and the class on the
same pod reads 1,001.9 / 1,221.9 at 40.33 / 51.22 ms (pod D's 1,022.1 /
1,270.2 for the same case, the pods' and texts' spread), so on this chain
the policy adds 6 percent on prose and nothing on code over the class
alone, and the sharded drafter 3.2 and 8.2 percent on top of both. The
16-stream entries of record are 1,033.7 prose / 1,322.3 code, the three
together, measured as one chain on one pod. The sharded drafter with the
class and c0.3 reads 998.3 / 1,297.8 at 44.72 / 49.19 ms (the class's
7.6 and 7.9 percent over 928.2 / 1,202.7, as on the gathered drafter),
and with the window controller instead of the budget policy 1,049.2 /
1,307.9 at 37.79 / 49.30, within 2 percent of the budget stack both ways,
as the two policies were at 8 streams; the entry keeps the budget policy.
The class under the budget policy on prose at 8 streams (gathered
drafter, pod C): 662.6 at 30.49 ms, 2.532 tokens a step a row, against
c0.3 without the class 639.6, c0.3 with it 604.3 and the budget policy
alone 621.5, so with the policy the class gains on prose at 8 too and the
earlier loss was c0.3's acceptance on those texts; the 8-stream prose
entry stays the sharded drafter with the controller at 679.4: with the
class it reads 692.4 at 28.46 ms with the controller and 673.2 at 29.09
with the budget policy, both within 2 percent of the entry, so the
class is a tie on prose at 8 streams and the entry keeps the simpler
configuration; on code at 8 the
sharded drafter with the class at c0.3 reads 976.5 at 34.77 ms (the
gathered drafter with the class 944.1 at 35.97, the same tokens a row),
the drafter's 1.2 ms at batch 8; with the controller 974.1 at 34.86 and
with the budget policy 958.7 at 34.68, all three within 2 percent, as
the policies sit at depth four on code at 8 streams whatever the rule.
The 8-stream code entry of record is 976.5, the sharded drafter with
the class at c0.3. One depth policy across the entries is not required:
the planner picks the depth policy per workload from these measurements,
and the three being within 2 percent on code at 8 streams is itself the
measurement the rule records (on code any policy sits at depth four; on
prose the budget policy or the controller wins by the step). Each width
entry was run twice in its exact configuration on the same pod and tree
for a noise band: 16 streams prose 1,033.7 and 1,053.6 (mean 1,043.7,
spread 1.9 percent), code 1,322.3 and 1,332.4 (1,327.4, 0.8); 8 streams
prose 679.4 and 685.6 (682.5, 0.9), code 976.5 and 973.3 (974.9, 0.3).
No repeat differed by more than 3 percent, so no third run; the table
carries the mean and the spread, and a change under the spread is not a
result.

The drafter's call at batch, by term, in the one-row account's harness
(the sharded drafter, per device, ten calls averaged from the trace):
at 8 rows the call is 2.55 ms wall and a device is busy 1.85 ms in 211
to 226 kernels, the allreduce kernels 0.52 to 0.66 (the push kernel
0.35 as seven sites at 50 us, the hc sums 0.12 to 0.19, the plain sum
0.10), the stage GEMVs, fronts and attention 0.63, the experts 0.29 to
0.43, the Markov loop's and head's small kernels 0.24, the rest gaps
and host; at 16 rows 3.30 wall, 2.65 busy, the allreduce 0.87 to 1.07
(the push 0.66 as seven sites at 95 us), the stage 0.89, the experts
flat, the Markov and head 0.37. From 8 to 16 rows the call grows 0.75
ms: the allreduce 0.41 of it, the stage GEMVs 0.26, the Markov kernels
0.14. The seven push-form sites a call, read off the drafter program's
nodes, are the token embedding's all_reduce (the vocab-parallel
embedding sums one real row among zeros at full width) and six hc sums,
an attention and an FFN sum in each of the drafter's three blocks; the
five Markov pair exchanges are allgather nodes on the ring write path
already (fifteen ring writes a call at 2 us each, their flag waits in
the gaps). The push records read 37 / 67 us a site at 8 / 16 rows on
the bf16 sites and 66 to 71 / 129 to 133 on the fp32 FFN sites, but
the push kernel alone, measured by tp-step in chains of 80 sites, copies
the fp32 partial in 20 / 35 us at 8 / 16 rows: the records inside the
captured graph carry what the push waits on at its site (the ring
counter, a peer still reading its inbox, the ranks' skew) on top of the
copy, so the drafter's allreduce term is site time on the device
timeline, waits included, not a bytes term; six of the seven sites are
the step's own sums, and the fence form of the push's release (tp-step)
takes 3 us a site off every one. The site the drafter owns is the
embedding's: one push and the plain sum (the 0.10 to 0.14 ms _ar_sum
record, its wait included), 0.15 to 0.19 ms a call,
removed by reading each id's row from the owning rank's shard instead of
summing one real row among zeros: vxpu::tp_embed_rows (the fork's
ParallelEmbedding hands the ids, its shard and its first index to a
rows hook the exporter sets under --embed-rows), a step-graph node whose
capture shares the ranks' shards once (parallel.World.share, the tensors
at their fixed addresses) and launches one kernel (triton_ops.peer_rows)
with the four shards as its pointers, each row loaded from one shard
behind an owner mask over P2P, as the ring write reaches the peers'
inboxes; no flag and no sum, since the shards never change, and no copy
of the 1.85 GB table. Eagerly, and for a rank stepping alone, the masked
lookup and the ranks' sum as before. On the drafter's call in the
account harness: 2.553 to 2.243 ms at 8 rows and 3.301 to 2.993 at 16,
0.31 ms off at both widths, the site's cost rather than its bytes; the
program has six push sites and no plain sum kernel, the stage, expert
and Markov terms per device are unchanged, and the hc sums' records
grow by what the removed site used to absorb of the ranks' skew. Those
first figures were read on a wrong graph (the solo capture, below);
on the right one the call is 2.553 to 2.421 ms at 8 rows and 3.301 to
3.363 at 16, because the first gather kernel (one 2-byte load a thread
behind four owner masks) costs 129 us a call at 8 rows and 281 at 16,
more at 16 than the site it removes; a streaming form (the owner's base
chosen once a row, the row copied in wide contiguous loads) is the next
measurement. The fetch's first form on the width rows, gate passed,
recorded beside the account and not as entries: 8 streams prose 693.5
at 29.32 ms against the band 682.5 (+1.6 percent), code 985.9 at 34.44
against 974.9 (+1.1), identical text and acceptance; 16 streams
1,055.3 / 1,307.2 against the same head's 1,047.6 / 1,311.9, inside the
band with the step 0.17 ms longer, as the 16-row account says. The
gather kernel alone, on four idle GPUs against torch at 1 to 40 rows:
both forms bit-identical, the first 19 to 23 us a call and the
streaming one 18 to 19, flat in rows, so the 129 and 281 us records in
the captured step were the kernel's P2P reads stalling behind the busy
peers' traffic and not its work; the streaming form, with fewer and
wider requests, is the default and the first form the A/B
(VXPU_PEER_ROWS=masked), and the step's account on the next tree was
the arbiter: there the streaming form's record is 233 us a call at 8
rows and 466 at 16, worse than the first form's, and the call 2.495 /
3.606 ms against 2.553 / 3.301 without the fetch. So the cost is the
P2P read itself from peers whose memory is busy with their own step,
whatever the kernel's shape (115 to 229 KB at under 1 GB/s), a wash at 8
rows and a 0.3 ms loss at 16, and the pull form closes as measured. The
engine pushes partials rather than reading them for this reason; the
form that fits the site is the push, the owning rank ring-writing its
rows into the peers' inboxes and raising the flags, no sum, a quarter
of the old site's bytes; it is recorded as the next form and not built,
the site being worth 0.1 to 0.2 ms a step; built once the entries were
in (VXPU_EMBED_ROWS=push, the pull form the A/B): one kernel on each
rank writes the rows it owns from its own shard into every rank's
output buffer at the rows' positions (the world's buffers for the node,
one per destination rank, at fixed addresses), raises a flag per peer
on its stream and waits on the three flags it is owed, as the ring path
does; each row is written by exactly one rank, so there is no sum. The
width entries on the current head are measured on the artifacts
without the fetch: on that
head (tp-step's fold at two producers, the fused push off under the
class and by token rows, the fence) the sharded drafter reads, gate
passed, 8 streams prose 698.1 at 29.12 ms against the band 682.5 (+2.3
percent, the rows rule no longer fusing the 40-row program plus the
fence), 16 streams 1,084.1 / 1,346.2 against 1,043.7 / 1,327.4 (+3.9 and
+1.4), where the step equals the previous head's and the prose gain is
the budget policy's depth path (its cost model is measured at startup
and its depth distribution moves between runs), the code gain 0.5 ms a
step; at one stream the head equals the fence head (207.9 / 310.6
against 208.0 / 310.4 in one chain). Under the rule the width entries
move to this head: 8 streams prose 698.1, 16 streams prose 1,084.1 and
code 1,346.2, each with its gate passed and its control chain on the
same pod; the 16-stream prose entry carries the caveat above and gets
its own band from one repeat on this head (inside the old band, the
entry becomes the mean of the two with the spread), and 8 streams code
with the class 989.0 at 34.33 ms against 974.9 ± 0.3 (+1.4, the
record's tokens, 4.255 a row). The repeat read prose 1,083.9 (160 steps
against 159: the depth path reproduces, so the entry is 1,084.0 with no
spread) and code 1,311.6 at 101 steps against 98, back inside the old
band, so the 16-stream code entry is the mean, 1,328.9 with a 2.6
percent spread, equal to the previous head's. The one-stream entries stay 208.4 /
335.1 with this head an equal reading. The fused push's rows rule
read the artifact binding's batch, which on the drafted path is the
stream count: every 8-stream program of 8 to 40 rows read 8 and took
the fused push, 0.4 to 0.55 ms a step at 8 streams on two pods; the
rule now reads a program's token rows, batch times step_tokens, against
tp-step's threshold of 6, and stays off under the bf16 wire class,
whose rounding the fused push's fp32 wire does not share (the 8-stream
code tokens moved with both on, 4.255 to 4.04 accepted a row). The
grouped expert form's schedule revision on the drafted one-stream step
read +0.8 / +1.0 ms a step (197.9 / 291.6 against 208.4 / 311.0, the
one-row verify 9.67 against 8.68), tokens identical; that revision ran
the member path on every block, and the form with the single-pair body
(3fb962f) runs again on the next tree as the deciding leg for the
drafted step. With the single-pair body on that tree the form is a tie
within 0.1 ms at every verify depth on the five-token step (209.2 /
311.2 against 207.9 / 310.6) and 0.23 ms ahead at six rows (340.0
against 335.2), tokens identical; with tp-step's schedule fold as well
(the pre-pass off the critical path), on another pod with the direct
control in the same chain, the form wins the drafted step: five-token
prose 212.0 at 14.26 ms against 207.9 at 14.54, code 315.2 at 14.72
against 310.7 at 14.93, the five-row verify 12.84 against 13.17 and
13.07 against 13.31, the six-row entry 348.2 at 15.70 against 335.3 at
16.30 with the six-row verify 14.03 against 14.59; 0.24 to 0.56 ms a
verify, tokens identical, one row a shade behind. So the grouped form
stands for the drafted step as the planner's form from two rows, direct
at one, as tp-step's rule has it; the win is small because the sharing
is (below). The eight-row figure of record beside the vLLM fork's
c=1 log (8 rows at 9.9 to 10.1 ms a step, 2.44 to 2.57 accepted, and
11.5 ms wall measured at depth 4.7 on our cards): seven drafts fixed
from the two-pass drafter (block 10), the eight-row program every step,
the direct expert form kept at 48 pairs, fused push, prose, 157.0 tok/s
at 20.16 ms a step (drafter 3.61, eight-row verify 16.54), 2.19
accepted a step; the plain push reads 154.5 at 20.50 with identical
tokens, so the fused push is 0.27 ms faster at eight token rows where
the rows rule turns it off. Twice their step at the same yield: vLLM's
"mean acceptance length" counts the bonus token, so their 2.96 to 3.31
at fixed depth 7 on our cards is tokens a step, the quantity our 3.19
is (2.19 accepted plus one), and their acceptance at positions 6 and 7
is 0.20 and 0.17, as weak as ours (0.18 conditional at 6; 0.17 / 0.11
unconditional at 6 and 7 in the dump below). The eight-row shape's case
against them is step time only. Our acceptance by position is 0.69,
0.77, 0.73, 0.74, 0.78, 0.18, 0.80. The per-step dump of the
verify rows and choices (VXPU_SPEC_DUMP, 282 steps) says what it is:
the share of steps where draft k equals the verify's choice before it
is 0.66, 0.72, 0.65, 0.69, 0.50 at positions 1 to 5 and 0.28, 0.25 at 6
and 7, with no agreement with the choice one position later (an
offset) and no repeats, so the second pass as a whole is the weak
predictor. In the fork's spec_draft the second block started from the
fifth draft's embedding and observed the states of the first block's
own run, whose positions after the input token held the noise token:
the drafted tokens' identities never entered its context. Now the
stages run over the first block's drafts as a sequence at their
positions and those states go into the window (fork 9e9b79c): it did
not help. The re-exported drafter (spec3m8s) accepts position 6 at
0.11 (unconditional agreement 0.17 / 0.11 at 6 and 7 against 0.72 /
0.72 / 0.66 / 0.67 / 0.55 at 1 to 5), 2.33 accepted a step on its own
text, the drafter 4.0 ms against 3.6 (the stage pass over four
positions). The reason is the input the drafter was trained on: its
target layers are 37, 38 and 39 (dspark_target_layer_ids), the last
three of forty, so the hidden states it expects at a drafted position
are the target's deep states, which nothing short of a verify
produces; the stages' own states are out of distribution however they
are formed. The vLLM fork does not chain passes at all: its drafter
runs one pass of an anchor and six noise tokens (ANATOMY.md:299) and
accepts its tail as weakly as ours. The two-pass drafter is parked,
with this dump; if the eight-row shape comes back (TP4 experts and the
sum forms cutting its 16.5 ms verify), the drafter for it is one pass
of block 7, the fork's shape, at one pass's cost (about 1.7 ms against
3.6).

The fixed shape (speculation shape "fixed", plan option, default
ragged): the drafted loop's host between its two device calls
measured 7 us a step (the policy, the ids, the acceptance), so what
the step pays at the call boundaries is the drafter's read-back sync
and the verify's launch, then the verify's read-back and the next
drafter's launch. The fixed shape relaunches the drafter's rank graphs
and the artifact's own S-row verify on input buffers written on rank
0's stream every step, the confidence mask, the verify rows, the
accepted count (the leading run of issued positions whose draft equals
the choice before it), the last kept token and the advanced position
all as tensor ops on rank 0, and a device ring of rows, choices and
counts copied to pinned memory behind an event the host reads a step
behind (StepGraph.input_buffers and relaunch(force), Decode.relaunch_raw,
Engine._tp_spec_fixed; the host path runs until both programs' rank
graphs are captured, then hands over). Its cost is the padded rows on
steps with fewer drafts than S - 1: 0.31 ms a step at prose's draft
distribution under the confidence floor (2/3/4/5 rows on 8/8/17/97 of
130 steps, 13.03 ms a verify against 13.34 for the five-row program
every step). The first pair, pod B, tree22, fixed depth 4 (both run
the five-row program every step), the scheduled expert form: prose
host path 178.0 tok/s at 15.13 ms (drafter call 1.69, verify call
13.43), fixed shape 185.0 at 14.45, text identical, 1.70 accepted both;
code 296.7 at 15.50 against 306.7 at 14.81, text identical, 3.59
accepted both: the two boundaries cost 0.68 ms a step at five rows.
The eight-row shape loses at today's slope (16.5 against 13.3 ms for
0.3 more accepted at best) and at 0.6 a row (its two-pass drafter is
3.6 ms); it pays only where acceptance is far above ours.

The tensor-parallel experts class (tp-step's VXPU_EXPERT_TP=4 with
VXPU_FP4_DN_KS=1: every card a quarter slice of every routed expert,
the down K-partials summed in the collective, a numerics class of its
own) arbitrated by the league as the bf16 wire was, on tree24 (this
branch with tp-step's head merged), every run with its knobs stated.
The class is a plan option, not a default: the user's decision is that
tensor-sharded experts are off unless a plan selects them, so every
entry below that carries the class carries expert_tp: 4 as the plan's
choice for that workload (the chains set VXPU_EXPERT_TP=4 and
VXPU_FP4_DN_KS=1 explicitly), and the 8-stream prose entry's plan does
not.
One stream, the fixed five-row shape, c0.3, the direct form to 48
pairs, pod A: class off 209.9 tok/s prose at 14.09 ms (verify span
12.77, drafter 1.3, 1.99 accepted) and 319.3 code at 14.21; class on
239.0 at 12.55 (11.34 + 1.2, 2.03 accepted) and 358.7 at 12.66 (3.59),
+13.9 and +12.3 percent, 1.54 ms off the step; the repeat identical in
text and figures; the code text identical to the class-off run, the
prose text forking at character 192 with the acceptance within the
band. Against the league's bars, 239.0 of 252 and 358.7 of 494 (the
code bar's shape is the six-row artifact, whose class figure follows).
Sixteen streams (spec2t5m, the budget policy, the bf16 wire): class
off 1,085.3 prose and 1,314.3 code (the entries of record reproduced
on this tree); class on 1,125.8 / 1,087.6 / 1,116.5 prose, mean
1,110.0, spread 1.7 percent, and 1,355.6 / 1,436.7 / 1,387.3 code, mean
1,393.2, spread 2.9 percent, texts identical across the three runs,
accepted 1.53 to 1.63 against 1.65 and 3.38 to 3.42 against 3.36; the
spread is the budget policy's depth choices, which follow the measured
costs. Eight streams (c0.3): code with the bf16 wire 991.8 to 1,069.9
and 1,067.5 (text identical, 3.55 accepted both), the entry 1,068.7;
prose, class off, 676.3 against 667.5 with the class, the step 33.85
to 32.51 ms but the class's text on that prompt accepting 1.89 against
1.96, so the 8-stream prose entry stays class off under c0.3. The
entry of record (698.1) ran the adaptive controller: on tree24 the
controller reads 699.2 class off and 703.3 / 704.1 with the class
(texts identical, the step 30.0 to 26.0 ms, but the controller settles
one draft shallower on the class's text: 1.47 accepted against 1.64),
inside the band with the acceptance at the band's edge, no move; the
budget policy reads 685.8 class off and 720.4 / 683.2 / 734.2 with
the class (texts identical, accepted 1.68 / 1.62 / 1.67: the policy's
depth choices move the figure), mean 712.6, spread 3.6 percent. That
mean is 2.1 percent over the entry of record, but the low run sits
inside the record's band (682.5 plus or minus 2.3 percent), so it is
not a consistent gain under the rules: the 8-stream prose entry stays
698.1 with its class-off plan, and 712.6 plus or minus 3.6 percent
stands beside it as the budget policy with the class. The 132-prompt
acceptance line under the fixed shape, class off: 173.2 tok/s, 2.444
tokens a step, 14.11 ms, accepted 1.469, conditional 0.724 / 0.69 /
0.672 / 0.66; class on: 194.7 tok/s, 2.433 tokens a step, 12.50 ms,
accepted 1.459, conditional 0.723 / 0.683 / 0.669 / 0.668, so the
class costs 0.7 percent of the acceptance over 132 prompts (the bf16
wire cost 1.6) and gives 12.4 percent of the throughput. The class
holds everywhere by acceptance and repeatability and shortens the step
everywhere; the one aggregate it loses is 8-stream prose, by that
prompt's text.

The drafter's closing table for hero run 15 (the one-stream artifact's
sharded drafter under the class, per rank, ms a call; the call alone
1.55 with a 20-call profile, the in-loop span 1.2 under the fixed
shape, the difference the standalone call's launch skew across the four
rank graphs and the host's launch, both hidden behind the running
verify in the loop): GPU busy 1.37 in 146 kernels (1.46 with the class
off: its six hc sums wait 0.12 less once its own experts are sliced).
The stage GEMVs 0.30 (16 _gemv_rows over about 470 MB of stage weights,
main_proj and the fp8 head slice: at the byte floor); the sharded
Markov head's five sequential positions 0.33 in about 16 small kernels
(the bias reduction, the argmax, the pair, the embedding; its 66 MB
table in L2 after the first position); the drafter's own experts 0.18
in six direct-form launches moving under a megabyte; the exchanges 0.11
(three ring writes, the pair all-gather, six hc sums at 4 us); fronts,
the PE GEMV, swiglu and attention 0.13; about 60 kernels under 5 us
(the MXFP8 casts, 22 a rank; row_post 9; small pointwise 12;
front_stats 6) for 0.12 of launch granularity. So the GEMVs are at
their floor and the rest is launch latency: the cuts are the Markov
head's position as one kernel (vxpu::markov_pick: the lookup, the fp32
bias over the slice, the argmax and the exchange's pair in one launch
and a decode; 0.15 to 0.2), the three expert stages as one launch each
(0.08), and the casts and row_post folded into their producers (0.05);
about 0.3 of the 1.2, 239.0 to about 245 tok/s at 12.55 ms and 3.0
tokens a step. The first cut measured on the step (vxpu::markov_pick,
the drafter re-exported as spec2t5p9, pod A, the class and the fixed
shape, the direct class): the drafter's span 1.2 to 0.97 ms, the step
12.55 to 12.30 prose and 12.66 to 12.42 code, 239.0 to 243.8 and 358.7
to 365.9 tok/s (+2.0 percent both), the texts identical to the
previous drafter's and to the repeat (243.6 / 365.6), the acceptance
identical to the step count; the kernel against its torch reference
on the GPU: index, value and embedding identical at four shapes with
ties, 20 us a position eager against 74 for the chain. The one-stream
entries with the class (expert_tp: 4 in the plan), the fixed shape and
the Markov pick: 243.7 prose (two runs, no spread), 365.8 code on the
five-row shape. The second cut, the direct expert path as one launch a
layer (FP4_ONE_LAUNCH, a table knob that reaches the main step's 40
layers as well), loses on the step: 12.35 against 12.30 ms prose and
12.49 against 12.42 code, and it is a class of its own (the prose text
forks at 319), so it stays off. The dependent launch of the down
kernel (FP4_PDL=2, tp-step's knob from hero run 12, measured on whole
experts then and left off: each pair's down kernel launched while the
gate-up grid drains, its programs waiting on their own pair's arrival
counter) is the day's second cut and costs nothing: on the same step
the verify span goes 11.33 to 10.90 ms, the step 12.30 to 11.84 prose
and 12.42 to 11.96 code, 243.8 to 253.1 and 365.9 to 379.7 tok/s (+3.8
percent), the repeat 252.8 / 379.2, the texts identical to the
knob-off run and to the repeat with the same acceptance (the knob
moves when the down programs start, not what they sum). The knob
reaches only the direct two-launch path (up to 48 pairs), so it
applies to the one-stream shapes and not to the width steps' GEMM
form. The one-stream prose entry: 253.0 (expert_tp: 4, the fixed
shape, the Markov pick, FP4_PDL=2; two runs, no spread), 99.6 percent
of the 254 measured on these cards for the vLLM fork's stack (the
published 252); the bars measured here for that stack are prose 254 /
827 / 1,235 and code 519 / 1,248 / 1,845 at 1 / 8 / 16 streams. On
the six-row shape the dependent launch gives 419.0 / 419.4 code at
13.07 ms (verify span 12.36 to 12.12; 4.56 accepted, texts identical to
the knob-off run and to the repeat) and 240.6 / 240.7 prose: the
one-stream code entry is 419.2 (expert_tp: 4, the fixed shape, the
Markov pick, FP4_PDL=2 in the plan), 80.8 percent of the 519 measured
here. At width the Markov pick is neutral: the per-position chain was
batched over rows already, the drafter's call reads 3.04 against 2.97
ms at 16 streams and 2.4 either way at 8, and the pairs land inside the
entries' spread with identical texts (16 streams 1,104.4 / 1,114.9
prose and 1,431.4 / 1,388.1 code against 1,110.0 and 1,393.2 of
record; 8-stream code 1,069.7 / 1,069.9 against 1,068.7), so the width
entries stand. The width drafter's account at 16 rows (the class on,
per rank, ms a call; the call alone 2.82, GPU busy 2.67 in 220
kernels): the exchanges 0.95 (the seven push sites 0.46, the first sum's
wait 0.41 on the standalone call, the hc sums 0.06, the ring writes
0.02); the Markov pick 0.35 (five positions at 70 us: the one-row
kernel read the 16.5 MB head slice once per row, sixteen times from L2,
which is why the pick was neutral at width; a row-blocked form, one
program taking 16 rows against one tile of the slice as a tensor-core
product, reads it once and runs 21 us a position at 8 and 16 rows,
c537693); the stage GEMVs 0.41 over the same 470 MB as at one row (the
floor 0.29: the rows tile costs 0.12); the drafter's own experts 0.31
in the GEMM form for under a megabyte; fronts 0.19, the swiglu GEMV
0.09, the PE GEMV 0.05, the casts 0.04. About 0.4 ms of bytes in 2.67:
at width the drafter is exchange-bound first, then latency-bound. The
cuts: the exchanges (the drafter's seven sites in the pull form,
measured on the step and not taken from the mailbox note's 2x, which
did not carry to the verify's sums; and the first sum's 0.41, the
four rank graphs' launch skew, hidden by enqueueing the next drafter
behind the running verify with its inputs written on the device while
the verify's program stays the host's choice after the drafter: the
half-fixed loop, zero padding, the policy kept; a fully fixed shape at
width is a recorded negative, 7 ms a step on prose, where the policy
ran 2/3/4/5-row programs on 28/29/94/12 of 163 steps at 16 streams,
33.4 ms a step against 40.5 for five rows always, and about 0.5 on
code); the pick's row-blocked form, 0.3; the GEMVs' rows tile, 0.12
(tp-step's table); the expert GEMM's two launches, 0.1 to 0.2. The
8-row account (pod B, the class on; the call alone 2.16, GPU busy 1.89
in 220 kernels a rank) has the same order: the exchanges 0.64 (the
first sum's wait 0.32, the push sites 0.25, the hc sums 0.04, the ring
writes 0.03), the stage GEMVs 0.31 at their floor, the expert GEMM
0.29, the pick 0.19 with the one-row kernel, fronts 0.11, the rest
about 0.3. The row-blocked pick measured on the step at 16 streams
takes the drafter's call from 3.04 to 2.81 ms (the aggregates inside
the budget policy's spread: 1,137.7 prose / 1,395.1 code against
1,104.4 / 1,431.4), 0.23 of the sized 0.3. Ranked per rank a call at 8
and 16 rows: the first sum's skew 0.32 / 0.41 (the half shape), the
push sites 0.25 / 0.46 (the drafter-scoped pull form), the pick 0.1 /
0.23 (done), the GEMVs' tile 0 / 0.12, the expert GEMM's launches 0.1
to 0.2: up to 1.1 ms a step at 16 streams and 0.75 at 8, about 2.3
percent of either step against the bars of 1,845 and 1,248.

The eight-row code shape, sized for its switch-on: code accepts about
0.93 a position conditional on our stack (0.977 / 0.953 / 0.926 /
0.947 at four drafts, 4.56 accepted of five), so seven drafts give
5.29 accepted and 6.29 tokens a step, which is the vLLM fork's 6.26;
its drafter is one pass over an anchor and six noise positions (the
checkpoint's block is 5; --draft-block 7 exports that shape, block 7
and passes 1, the Markov pick inside it), about 1.05 to 1.1 ms a call,
and positions 1 to 5 compute as the five-row drafter's do because the
block is causal; the open question is positions 6 and 7 in one pass on
our stack (their table: 0.62 / 0.53), the readiness test to run in a
pod gap. The step: the five-row verify reads 10.90 ms and the six-row
12.12 on the same tree, pod and class with the dependent launch, 1.22
ms a row between 30 and 36 direct pairs, so eight rows is about 14.5 by
that slope, the step 15.6, and 6.26 tokens over it 401 tok/s, under
the six-row's 419.2; at 0.8 a row the verify is 13.7, the step 14.8,
423, a tie; at 0.6 a row 13.3, 14.4, 435. The eight-row shape becomes
the code entry when the per-row cost falls under about 0.8 ms, which
is what tp-step's slice-amortizing expert kernel is measured against.

The drafter and verify side by side with the vLLM fork's (the user's
rule for the rest of the run: if they do it and we are not at parity,
we try it; a spent list before parity means an unaccounted
difference). Their elements, from league/fork/ANATOMY.md and its
sources, each marked adopted, measured equivalent, or not tried:
(1) one parallel pass over an anchor and six noise positions, the
Markov head sequential inside the graph: adopted at the checkpoint's
block of 5 (four drafts, the five-row shape, the Markov chain in the
graph since the pick), the block of 7 exported (--draft-block) with
its readiness test queued; the two-pass form is the recorded negative.
(2) Gumbel sampling of the drafts and rejection sampling of the verify:
at temperature 0 the Gumbel-max is the argmax and rejection sampling
accepts a draft exactly when it equals the target's argmax, so the
greedy match is the same rule and the bonus token is our choice at the
first mismatch; measured equivalent: their tokens a step on these
cards 3.07 at c=1 under their controller against our 3.08 under the
confidence floor, 2.96 to 3.31 at fixed depth 7 against our 3.19 to
3.33, their per-position acceptance equal to ours where both have it.
Sampling at temperature above 0 is not an item for the league (its
method is greedy). (3) Adaptive verification choosing the depth inside
one graph, the budget on the CPU with a host-to-device copy and one
ranking kernel between the graphs: ours is the per-length programs
chosen on the host (the confidence floor, or the budget policy from
the confidence head's survival probabilities and the measured program
costs) plus the fixed shape at one stream and the half shape at
width; measured: one program every step with the acceptance on the
device is +3.9 percent at one stream against the ragged host loop,
and at width the per-length programs beat one fixed program by the
padding (7 ms a step at 16 streams prose from the program histogram),
so ours keeps the programs and hides the drafter's launch instead
(the half shape, its pairs queued); their adaptive verifier bought
about 1 ms on prose and nothing on code in the reference run.
(4) Two graph replays a step, the whole draft step in CUDA graphs
including the sequential Markov sampling: adopted, as the drafter's
four rank graphs and the verify's four, one replay each a rank, the
Markov chain inside the drafter's graph, and in the fixed shape the
two chained on the stream with no host between. (5) The drafter's
backbone sharded with the target under tensor parallelism, six more
all-reduces and one logits all-gather a step: adopted and measured
better, the stages sharded with six hc sums and the Markov head
sharded by vocab with a (value, index) pair exchanged instead of the
logits (20 MB a rank a call at batch 8 before the pair). (6) The DSpark
confidence head: adopted (the confidence floor, the budget policy's
survival probabilities). (7) The eager work between their graphs (the
rejection sampling kernel, the CPU budget, a copy-event
synchronization, a broadcast of the confidence probabilities, the
output copy overlapped with the next proposal): ours has none in the
fixed shape (the acceptance on the device, the ring read a step
behind) and one host hop in the half shape; measured as the two call
boundaries at 0.68 ms a step at one stream. (8) Their Markov bias as a
replicated [B, 256] x [256, V] GEMV over the whole vocabulary a rank:
ours is the rank's slice in the pick kernel, one read of 16.5 MB a
position from L2, 21 us a position at any row count. (9) Not tried:
their draft graphs captured at sizes in multiples of 7 up to 112 rows
(varlen padding to the next graph size), where ours capture one graph
per program and row count; and their shared expert on a side stream
beside the routed experts (the intra-layer overlap), which is the
step's, in tp-step's build. The one element where we are behind by
measurement is none of the drafter's: at depth 7 the yield is equal
and the step is the gap, which the eight-row sizing above carries.
The varlen family sized: our width step verifies a rectangle of
streams x (D + 1) rows for one program per depth, so every stream that
issued fewer drafts than the step's D pads its rows; at 16 streams
prose the budget policy ran D = 4 on most steps (80 rows) while the
streams issued 2.4 to 2.8 drafts a step on average, so the sum over
streams of (issued + 1) is about 56 rows against 80, and at the
measured program slope at width (0.31 ms a row: the 64-row program
35.5 ms against the 80-row 40.5) that is about 7 ms of a 36 ms step,
and about 5 of 30 at 8 streams: a 15 to 20 percent item on width prose
and nothing on code, where the streams issue 3.98 of 4. Their padding
to a multiple of 7 costs 0 to 6 rows a step. It needs a verify program
over a flat batch of token rows with a sequence index and a position
per row, the acceptance per stream over a variable count, and the
family exported at the padded sizes: a new verify shape through the
exporter, the step graph and the loop, the first item on the hero 16
width list.

**Correction (Nov 4): the sizing above misread the budget policy.** Under
the budget policy every live stream issues the step's common depth, so
the 2.4 to 2.8 "issued a step" is the depth varying from step to step
(1 to 4), not streams issuing fewer drafts than the rectangle holds.
Counted from the survival logs: W29-16-sc prose (pod D), 153 steps, 150
with every row at the same depth, 8,947 token rows flat against 8,992
padded (0.995); the 8-stream budget runs read 0.997 to 0.999. The flat
program gains nothing on padding against the width entries as they
stand (the budget policy and the controller). Its real case is the
per-request depth, decomposed from the same logs: the confidence head
is calibrated on them (forecast / realized cumulative acceptance among
issued positions, 8 streams: 0.80 / 0.77, 0.58 / 0.58, 0.37 / 0.39,
0.24 / 0.31; 16 streams: 0.80 / 0.76, 0.58 / 0.60, 0.39 / 0.43, 0.23 /
0.26), so policies replay over the forecasts, each costed by the run's
own program curve (8 streams: step 16.7 + 0.419 ms a token row, the
drafter included, fitted to the 16 / 24 / 32 / 40-row programs; 16
streams: 14.7 + 0.358). The model reproduces the controller: common
depth 3 at 8 streams models 699 tok/s against the measured 698.1. The
vLLM fork's c=8 prose figures read back consistently: depth 3.13 is
drafts a request (their coverage 1.00 0.89 0.60 0.34 0.16 0.08 0.04 sums
to 3.11), so their step verifies 33 rows at c=8, their 2.91 tokens a
step is 1 + the sum of coverage times acceptance (2.90), and their step
is 8 x 2.91 / 827 = 28.1 ms at 33 rows.

| policy (model over the logs, our slope, four draft positions) | 8 streams prose: rows a step, tokens a row, tok/s | 16 streams prose |
|---|---|---|
| common depth 1 / 2 / 3 / 4 | 14.6 / 22.0 / 29.3 / 36.6 rows; 1.80 / 2.39 / 2.77 / 3.00; 578 / 674 / 699 / 686 | 30.5 / 45.8 / 61.1 / 76.3; 1.80 / 2.39 / 2.78 / 3.01; 1,071 / 1,171 / 1,159 / 1,094 |
| per-request budgeted slots, budget 3.0 / 3.5 / 4.0 / 4.5 rows a request | 2.55 / 2.75 / 2.89 / 2.97 tokens; 714 / 729 / 726 / 710 | 2.57 / 2.77 / 2.90 / 2.98; 1,245 / 1,236 / 1,203 / 1,154 |
| measured | controller 698.1 (29.6 rows, 2.548); budget 673.2 | half shape, budget: 1,208.1 |

(a) The common depth is the gap in tokens a step: at their 33 rows the
selection gives 2.89 to 2.97 tokens a row against their 2.91; at our
slope the best budget is 28 rows at 8 streams (729 modeled, +4.2
percent over the controller) and 48 to 50 at 16 (1,245, +6.3 over the
best common depth, +3 over the half entry). Per-position acceptance
among issued under our selection at their coverage (the top 89 / 60 /
34 percent of rows at positions 2 / 3 / 4): 0.630 / 0.511 / 0.432
forecast against their realized 0.605 / 0.520 / 0.505; equal at two and
three, and position four is where the head under-forecasts. Their
positions 5 to 7 add 0.118 tokens a step for 0.28 rows a request,
about one percent net here. (b) The depth choice at a common depth:
the controller sits at the optimum at 8 streams (depth 3) and the
budget policy within one percent at 16 (depth 3 on 105 of 156 steps
where depth 2 models one percent better); depth 4 everywhere loses 2
percent at 8 and 7 at 16. Nothing to run. (c) Acceptance per position
at width against one stream: position one realized 0.773 at 8 streams,
0.763 at 16, 0.78 at one, theirs 0.757; the conditional bands at
positions 2 to 4 are the one-stream bands at every width. Zero. What
remains of the width-prose gap after (a) is the step: 30.5 ms modeled
at 33 rows (27.5 measured at 32, 30.6 at 40) against their 28.1, 8
percent, the step per row; with our per-request tokens at their step
time the model gives 823 against their 827. The build for (a) is the
flat token-row verify with the budget as the program's row count (never
padded; the selection a host sort of R x 4 confidences a step): stage
1's fork patch (every per-sequence state indexed by a seq input, the
writes a group's tokens share taking the run's last row's value so one
index_put_ carries one value per slot, the kernel's per-row cache row
and newest) is written and stashed on the fork branch, paused Nov 4 on
the corrected sizing; its first measurement is 8 streams prose, flat
programs at 24 / 28 / 32 rows with the selection against the
controller's 698, 729 expected, and the model's assumption to verify
there is that a flat program at T rows costs what the rectangle costs
at T rows.

The flat verify landed (5702217, the fork at 9df98d1) and its CPU check
passes: the tiny model on two ranks, flat programs of 6 and 9 rows
against the per-length ones, fill pad gives the rectangle's tokens with
36 of 37 steps flat and the same 23 of 102 drafts accepted, fill drafts
the same tokens in 31 steps (32 of 115), the slots policy in 30 steps
(36 of 92); the flat decode program carries attn_select_rows at every
layer and 8 rotary ops where the rectangle carried 20. The first
measurement runs on pod D (chainD67): the flat programs at 24 / 28 / 32
rows exported with the family's flags, the controller's rectangle as the
control on the same pod and main artifact, then the cost pair (fill pad),
fill drafts under the controller and the slots policy at 28 and 32 rows,
two runs each.

Two one-stream and width results of the same afternoon. The reduce-scatter
sums at 8 streams prose (pod B, tree29, the controller, class off): forced
off 670.0 tok/s (29.41 ms a step), on 704.8 and 707.0 (27.95 / 27.87),
text identical across the three and acceptance 1.56 a step on all, the
same programs (3 rows on 74 steps, 4 on 86); the 8-stream prose entry is
705.9 (85.3 percent of 827), above the old band (694.0 to 698.9). The
one-pass block-7 drafter at one stream in the fixed 8-row program every
step (spec2t7p-tp4-b1, confidence 0.3 over seven positions, the class,
the dependent launch; pod D, tree29): prose 399 tokens in 128 steps at
14.93 and 14.92 ms, 208.8 and 208.9 tok/s against the five-row entry's
253.0; code 61 steps at 15.01 and 15.00 ms, 435.8 and 436.1 against
419.2 (+4.0 percent), texts identical between the runs. Issued 5.3 /
6.93 a step, accepted 2.14 / 5.67; conditional acceptance by position
prose 0.77 0.80 0.64 0.58 0.64 0.54 0.50, code 0.97 0.93 0.96 0.94 0.96
0.94 0.82: the one-pass drafter's tail holds where the two-pass one's
fell to 0.18 / 0.11, and on prose the 8-row verify's 13.8 ms costs more
than the 2.14 accepted tokens buy against the five-row program's 12.9
ms for 1.8; the sizing's 401 / 423 / 435 on code was met at the top of
its range. Code at one stream takes the block-7 drafter as its entry
candidate (436.0 mean, 84.0 percent of 519) once it is run as the
entry's full chain.

The width entries after the reduce-scatter sums (tp-step's rule from
16 token rows, which lived only in the kernel table until 0d25c39 made
it the module's default, so no width entry had taken it) and the half
shape, on tree29 (this branch with that head merged), every result line
now carrying the knobs in effect: 16 streams, the budget policy, the
bf16 wire, expert_tp 4, the row-blocked pick: the scatter forced off
1,128.5 prose / 1,426.1 code (35.21 / 47.19 ms a step), the scatter on
1,145.9 and 1,172.8 / 1,454.0 and 1,457.6 (the code step 45.14 / 44.98,
the 2.1 ms; prose's step rose with the deeper programs the policy
chose under the cheaper sums, its tokens a step with it), the half
shape on top 1,209.0 and 1,207.2 / 1,476.8 and 1,481.2 (35.76 and
33.86 / 44.40 and 44.24 ms), every text identical to the run before
and to its repeat: the 16-stream entries are 1,208.1 prose (97.8
percent of the 1,235 measured here; 1,110.0 this morning) and 1,479.0
code (80.2 percent of 1,845; 1,393.2). Eight streams code, c0.3, the
class, the bf16 wire: the scatter off 1,067.7 (32.48 ms), on 1,093.4
and 1,099.3 (31.62 / 31.58): the entry is 1,096.4 (87.9 percent of
1,248; 1,068.7 this morning). The drafter's seven sums in the pull
form, scoped to the drafter alone (DRAFT_ALLREDUCE_PULL): the drafter's
call 2.81 to 2.73 ms at 16 rows and 2.19 to 2.11 at 8, texts identical,
a tenth of the sized saving, as the pull form's PCIe reads cost what the
push saved; it stays a knob, off. The
code entry's shape is the six-row artifact (five drafts, c0.3; 348.2 of
record with the grouped form): on tree24 in the direct class, the host
path 337.3 code / 199.4 prose, the fixed shape 343.0 / 212.9 (code text
identical, the prose text its own program class), and the fixed shape
with the class 403.7 and 404.0 code at 13.57 ms (verify span 12.37,
4.56 accepted, 5.48 tokens a step; the code text identical to the
class-off run's) and 231.9 / 232.0 prose at 13.44: the code entry moves
to 403.9 (two runs, no spread), 81.7 percent of the 494 bar, and prose
stays on the five-row shape (243.7 against 232.0). With the Markov
pick on this shape as well (spec2u6p9, tree25): code 410.6 / 410.3 at
13.33 ms (4.56 accepted, the texts identical to the previous drafter's
and to each other), prose 235.9 / 235.8 at 13.21: the one-stream code
entry is 410.5 (the six-row shape, expert_tp: 4 in the plan, the fixed
shape, the Markov pick), 83.1 percent of the 494 bar, and the drafter's
0.23 ms shows at both shapes. How much there is to share was
counted on the drafted step's own rows (the prose prompt, a fixed depth
so every step runs one program, the distinct experts a layer a step
from the four rank graphs, the prefill's one-row replays taken out): at
3 verify rows 13.4 distinct of 18 pairs a layer (26 percent shared), at
4 rows 16.6 of 24 (31), at 5 rows 19.4 of 30 (35), the layers ranging
from 12 to 28 distinct at five rows. An fp4 expert with its scales is
18.2 MB, so the expert bytes a card a step are 1.09 GB at one row, 2.44
at three, 3.01 at four and 3.54 at five: at 1.6 TB/s a floor of 0.68,
1.52, 1.88 and 2.21 ms against steps of 8.9, 11.2, 12.4 and 13.4 ms.
From one to five rows the step grows 4.5 ms and the bytes' floor 1.5,
so bytes are a third of the per-row slope and the rest is the pushes,
fronts, sums and launches that grow with rows; the grouped form's
ceiling on this path is the shared share of the bytes' time, at most
0.8 ms at five rows, which fits its measured tie to small win; and at
one row the bytes' floor is 0.68 ms of an 8.9 ms step, so the
single-stream step is mostly not bytes, in its start and in its slope.
The reference stack measured on the same cards (the vLLM fork, their
formula, two repeats identical, the SSE path included) runs the plain
one-row prose step in 7.18 ms against our 8.6 to 8.9, and its logged
8-row step near 10 ms puts its slope at 0.35 to 0.4 ms a row, the
expert-byte floor itself: their rows cost only bytes where ours cost
1.1 ms. So the single-stream gap at five rows, about 4.7 ms, is about
1.5 ms of start and about 3 ms of slope, and the slope, the larger
term, is not bytes.

The one-stream entries of record move to tp-step's merged head (the
tree with the pinned staging at its default, the grouped form off, the
fused push by rows): prose 204.0 tok/s at 14.82 ms a step (the
five-token full stack, from 193.1) and code 331.3 at 16.50 (the six-row
step, from 312.3), the same tokens as before, the six-row verify 14.78
against 15.36 and the drafter 1.68 against 2.07; the pinned staging's
legs read 203.6 / 331.5, a tie, and the grouped form's leg at one
stream is the 22 ms step recorded above. One repeat each in the entry's
configuration: prose 206.5 (mean 205.3, spread 1.2 percent), code
332.7 (mean 332.0, spread 0.4), so the one-stream rows carry a band as
the width rows do. tp-step's next head (one fence then relaxed flags at
the end of every push, the push's 512 block, the fused push by rows)
on the same pod with those repeats as the control in the same chain:
prose 208.4 at 14.50 ms a step against 206.5 at 14.64, five-token code
311.0 against 307.6, the six-row code entry 335.1 at 16.31 against
332.7 at 16.43, same tokens, the six-row verify 14.59 against 14.72:
0.13 to 0.17 ms a step, the fence's few microseconds a site over the
step's sites, consistent across the three cases and at the edge of the
bands. The one-stream entries of record are 208.4 prose / 335.1 code on
that head, the previous head's 205.3 / 332.0 kept as its figures: the
rule is the current head measured with the control in the same chain,
and a consistent gain with identical tokens is the head's number
whether or not it clears the band. At width the same head reads, each
pair in one chain on one pod: 16 streams 1,066.6 / 1,337.0 against
1,055.3 / 1,307.2 (the fetch artifacts, the previous head as the
control), +1.1 and +2.3 percent with the same tokens; 8 streams prose
686.4 at 29.62 ms against 696.0 at 29.21 with the fused push forced off
on the control, the same tokens, 0.4 ms slower; 8 streams code with the
class 886.4 at 4.04 tokens a step a row against 987.0 at 4.255, the
tokens different. The head leaves the fused push to the rows rule (on
for programs of 16 rows or fewer, which at 8 streams are the 8- and
16-row verifies), and on the class path that form is not
token-identical to the plain push, so the width entries on this head
wait for the fused push off on the class path or its fix; the fence
itself gains everywhere the tokens are the same. At width the same head reads
inside the entries' bands (16 streams 1,047.6 / 1,311.9 against 1,043.7
and 1,327.4; 8 streams gathered 638.0 / 859.6 against tree12's 639.6 /
864.3), so the width entries stand on their tree13 chain with the merged
head as an equal reading.

tp-step's head e5bd3a1 (the pinned staging for the step's host copies,
the grouped direct expert form off by default, the pull-form sums, the
fused push on by default) merged into this branch and staged as one tree
puts the drafted one-stream step, same texts as the entries, at 14.82 ms
prose / 15.19 code with the pinned staging off and the grouped form off:
204.0 / 305.4 tok/s against the entries' 193.1 / 288.9 on the earlier
head, the five-row verify 13.39 against 13.95 and the drafter call 1.70
against 2.07, so the drafter's regression of the previous head is not
on this one. The pinned staging on reads 14.84 / 15.14, within the
band: the drafted loop's host copies are the drafter's (its inputs are
on device and its argmax read-back goes through the opaque op), so the
step's pinned copies have little to pin on this path. The grouped
direct expert form (each distinct expert read once for the pairs that
chose it, off by default) on the same chain reads 22.38 / 23.83 ms a
step, tokens identical: by verify length +3.0 ms at one row, +3.6 at
two, +5.4 at three, +6.7 at four, +8.5 at five against the direct form,
a per-launch cost that grows with the pairs rather than a saving, so it
stays off for every entry; tp-step's ladder on the plain step is the
measurement to set beside it. The 8-stream trio with the fused push
off and the six-row code trio follow on the same tree. The budget policy
at 16 streams (pod D, on tp-step's merged head): prose 896.2 tok/s
aggregate at 43.17 ms a step, the chosen verify at two / three / four /
five rows for 10 / 40 / 101 / 14 of 165 steps, 1.63 accepted a row,
2.424 tokens a step a row; code 1,177.8 at 53.67, 93 of 101 steps at
five rows, 3.46 accepted, 3.96 a row; against the c0.3 entry's 822.8 /
1,074.5 that is 9 percent, part of it the tree (the five-row verify at
80 token rows 49.0 ms here against 52.6 before the merge), and the
policy's own share read against c0.3 on the same head (pod C, 883.6 /
1,147.0 at 50.53 / 54.04 ms, 2.80 / 3.88 tokens a step a row) is 1.4
percent on prose and 2.7 on code, the merged tree 7.4 and 6.7. On prose
the policy trades depth for step (2.42 tokens a step a row at 43 ms
against 2.80 at 50.5); on code it sits at depth four as c0.3 does. The
16-stream row is 883.6 / 1,147.0 at c0.3 and 896.2 / 1,177.8 with the
budget policy. The budget policy at 8 streams (pod D): prose
621.5 tok/s aggregate at 30.94 ms a step, the chosen verify at two /
three / four / five rows for 16 / 31 / 93 / 26 of 166 steps, 1.66
accepted a row, 2.41 tokens a step a row; code 846.4 at 36.61, 93 of 103
steps at five rows, 3.41 accepted, 3.883 a row; against the window
controller's 630.4 / 832.8 on the same pod, within 2 percent both ways.
The confidence head's forecast and the window's acceptances agree on
these prompts, and the shape's cost curve (24 token rows 25 ms, 32 rows
28, 40 rows 32 to 34) is flat enough between depths three and four that
the choice hardly matters; what the fork's budget has that this cannot
is the ragged step, a token-row count a step rather than a rectangle,
and that is a verify-program change. The vocab-sharded Markov drafter on
the 132-prompt line (pod C, new node): 167.9 tok/s at 14.64 ms against
the gathered form's 167.0 at 14.72 on the same pod and tree, the
acceptance identical to the third digit (1.459 a step, 0.722 / 0.679 /
0.665 / 0.674 both); its one-stream league pair 193.8 / 291.8, the same
tokens. It is the drafter for the concurrency rows (2.44 against 3.51 ms
a call at batch 8); at one row it is worth 0.1 ms. Next structural
item, whatever the ragged step's estimate says: a request slot per
token row. Every per-request state in the fork is indexed by batch row
(fourteen sites in model.py: the window ring's seed and index_put, the
KV cache's index_copy and index_put, the compressed cache's, the
indexer's keys, the engram's kv and score states, the DSpark rings'
observe, and every `[:bsz]` view those paths hand the placed kernels),
and the placed attention kernels (window_kv, attn_select, attn_core,
the compress path) read row b for program b. A `cache_rows` [rows]
int64 input threaded through Transformer.forward to every write (the
slot in place of the row) and read, with a row indirection in those
kernels (program b reads slot cache_rows[b]), makes the step's rows
independent of the requests' slots: the ragged verify (one program per
total token-row count, each request's slots flattened into the row
list, the per-length programs the case where every request has the
same depth), and admission and eviction at concurrency without copying
a cache row, both need it. The gather-based shortcut (a scratch batch
copied per step) is out: at 40 rows it moves about 4 GB of cache a
step. Sized at a day or two of fork, kernel and exporter work with a
token check at each step, and sized for its gain before that: the
8-stream budget run rerun with the confidence head's survival
probabilities logged per step and row, the fork's slot selection
replayed over them against the measured cost curve (8 token rows 17.6
ms, 24 rows 25.0, 32 rows 28.0, 40 rows 31.6, 80 rows 47.1, the drafter
3.9 and the verify path's 3.0 added), gives the ragged step 638.5
expected tok/s on prose against the rectangle's 614.4 on the same log
(3.9 percent; the ragged step settles at 29 to 33 token rows where the
rectangle takes 32 or 40) and 907.0 against 901.0 on code (0.7 percent;
36 to 40 of the 40 slots taken nearly every step), an estimate, not a
measurement. The logs say why: first-position survival averages 0.80 on
prose and 0.97 on code with a spread across the eight requests of 0.44
and 0.20 within a step, so the slots a ragged step would drop are few,
and a token row costs 0.4 ms against a step base near 24. The rectangle
is near-optimal on this benchmark; the ragged step stays the structural
item for admission and eviction, and the concurrency rows' remaining gap
is the step at 40 and 80 token rows and the drafter at batch. What remains for the 1.5 ms target is the exchanges and the
graph's size: the fused-push GEMV form (allreduce.gemv_allreduce,
measured in hero 14, not in the lowering) and fewer nodes to launch. The draft tokens' trip to the host and back costs
the 0.07 ms above; keeping them on the device (the compose path the
one-GPU native loop has) buys nothing until the drafter itself is under a
millisecond. The verify's row cost by the step trace (fold-4 tree, pod D): the plain
step 10.07 ms = inputs moved 0.11 + the ranks' run 9.97, each rank's
graph 9.45 ms on its GPU with 0.6 ms of host around it (thread hand-off
0.15, input copies 0.11, launch 0.08, the wait after); the five-row verify
14.23 ms = 0.11 + 14.12, each rank's graph 13.6 ms with the same 0.6 ms
around it. The 4.2 ms the five rows add are all inside the ranks' graphs
(expert bytes at this placement, the GEMV rows and the sums at five
rows); the host and the hand-offs do not grow with rows. Splitting the
4.2 by kernel family is the tp-step workstream's ledger to run on this
artifact; this workstream's share of the step is the drafter and the
loop around the verify.

**Hero run 14, closed (Nov 2, 17:30 UTC): not met on all three
numbers.** Tensor parallelism of the dense layers across the four cards
with the experts split per card, built as one captured CUDA graph per
card meeting at 89 one-shot sums a step, measured on the real model
against the layer-sharded control on the same tree, tokens in the
rounding class:

| load | control | TP4 + EP4 | bar |
|---|---|---|---|
| 1 row plain | 13.1 ms, 78 tok/s | 13.1 ms, 76 tok/s | 150 tok/s |
| 1 row with the drafter | 102 tok/s (hero 13) | about 100, estimated (no TP drafter built) | 200 |
| 33 rows, 32 different prompts | 92.2 ms, 369 tok/s | 52.6 ms, 624 tok/s | 1,000 |
| 33 rows, one prompt / copies | 29.0 / 26.7 | 35.6 / 34.6 | |
| 8 rows (TP4, experts layer-sharded) | 20.8 / 16.6 / 32.3 | 23.6 / 19.2 / 35.0 | |

Where the one-row step's 13.0 ms goes (step trace over 64 steps, 672
kernels a card): host around the graphs about 1.0; kernels busy about
2.1 (the profiler's figure on the sum-free step, with the caveat that it
could not be aligned to one step); gaps between a card's own nodes inside
its graph about 7.6 (11 µs a node, where hero 11 measured 0.75 µs between
tiny dependent nodes: a quarter-width GEMV reads a quarter of the bytes
but keeps most of its ramp and tail, so "gaps" and short kernels are not
separable here); the 89 sums 2.4 including about 1.4 of skew (27 µs a sum
in the step against 11 alone). The direction that died: tensor
parallelism as four independent per-card graphs with a rendezvous per
sum on PCIe-only cards; the bytes split four ways and the fixed costs per
step doubled. What survives, and only inside hero 15's resident executor:
the TP4 + EP4 shape itself, whose byte floor at one row is 2.2 ms, with
the per-node and rendezvous costs removed. Also measured and kept: the
one-shot and reduce-scatter all-reduce as a table-chosen op (11 µs at one
row, 97 at 132 rows), the per-GPU memory ledger for the layout (72.2 GB
of experts a card, balanced), the by-id and popularity placements, the
expert spread's own gains through the peer path at wide rows (93 → 51 ms
at 33 different rows), the crossings benchmark, the plan's `parallel`
section and the chooser. The planner's prediction of 161 tok/s at one row
was wrong by the per-kernel term it did not have; the model now carries
it.
**Hero 15, the expert GEMM form (Nov 2, measured on pod C).** The pair
tax was not the bytes: the grouped fp4 expert kernels already read each
expert once per block of pairs, but loaded the weight tile with plain
loads and did gate and up in one program, and at 240 pairs a layer they
ran at 60% of the read peak. The GEMM form keeps the grouped schedule
(blocks of up to 16 pairs sharing an expert) and streams the packed tile
through a tensor descriptor with gate and up as separate programs, one
tensor-core dot per load, the block's x rows loaded per K step: the
direct kernel's recipe applied to blocks of pairs. Bit-identical to the
grouped kernels. Micro-benchmark (`dev/hero/bench_grouped.py`, synthetic
banks of the real shapes, the pairs' experts drawn as a step's, bytes per
distinct expert at 1,535 GB/s), the swept tiles (16 pairs a block, 32
expert rows a program, K 128, two warps, three stages; 48 sets in
`dev/hero/fp4-gemm-rtx6000.json`): 48 pairs 558 → 532 µs (87 → 90% of
the floor), 240 pairs 1,780 → 1,183 (61 → 93%), 990 pairs 2,076 → 2,526
at a real step's 191 distinct experts (57 → 93%). The real step, same
tree, grouped → GEMM, tokens identical in every case:

| case | pairs a layer | step ms, one prompt | same prompt | a prompt per row |
|---|---|---|---|---|
| 9 rows | 54 | 21.6 → 20.5 | 17.4 → 16.8 | 34.6 → 33.5 |
| 33 rows | about 200 | 29.4 → 28.3 | 27.2 → 27.5 | 93.9 → 67.5 |
| 8 rows, four drafts | 240 | 55.1 → 45.4 | 39.9 → 36.1 | 96.8 → 71.7 |
| 33 rows, five-token block | 990 | 109.5 → 102.6 | 105.9 → 101.0 | 228.3 → 169.9 |

So the step with eight different prompts and four drafts is 71.7 ms
against the 63.8 ms of expert time it had before (the bytes alone 41.5);
the 33-row verify is 170 ms against 228; the plain 33-row step with a
prompt per row 67.5 against 93.9, which puts one 33-row batch at 489
tok/s and four in flight well above hero 9's 1,414. The form is the
table's above `FP4_GEMM_MIN_PAIRS` (9, where the direct kernels stop);
the knobs `FP4_GG_*` and `FP4_GD_*` are table entries. Per-family
profiles (`dev/hero/kprof-rows.py`, the expert kernels' GPU time a step,
scaled to 40 layers from the profiled window): eight rows with four
drafts 67.8 → 44.3 ms (the bytes 41.5: 94% of the floor, from 61%); 33
rows with the five-token block 125.0 → 79.2 ms; 33 rows plain with a
prompt per row 44.4 → 29.1 ms. What remains above the bytes at these
widths is the step's other work; the expert kernels themselves are at
their floor.

**Hero 15, the fused push (Nov 3, measured on pod B, four RTX PRO 6000 over
PCIe).** The tensor-parallel step's 80 mixing sums (`vxpu::tp_allreduce_hc`)
ran as two kernels a site: a push of the partial into the peers' inboxes,
then the sum with the block's `hc_post` in its epilogue. The fused push moves
the push into the GEMV that makes the partial: the attention's `wo_b`, and
the shared expert's `w2` with the routed experts' fp32 sum added in its
epilogue as `forward_routed` adds it. The engine rewrites the exported pairs
(`lower.fuse_pushes`; the plan's `fused_push`, on by default since Nov 3,
`VXPU_FUSED_PUSH=0` the A/B back): `tp_allreduce_hc(float(attn_out_partial(..)))` becomes
`tp_gemv_allreduce_hc(attn_out_mid(..), wo_b, sb, None, ..)` and the FFN's
`y + float(shared_ffn_front(..))` becomes
`tp_gemv_allreduce_hc(shared_ffn_mid(..), w2, s2, y, ..)`, so the producer
and the sum are one collective node and the Comm's site numbering cannot
drift between them. In the step graph the GEMV (`gemv_rows(peers=)`) writes
its partial into the peers' inboxes from its epilogue and raises their flags
from its last program, and the sum kernel only waits and mixes
(`allreduce.gemv_allreduce_hc`); the attention sites push bf16, half the
bytes, which the sum loads as fp32 exactly, and the FFN sites fp32. The
numbers are the one-shot form's, added in the same order: tokens identical to
the control at 1, 2, 3 and 5 rows, and the two kernel forms agree bit for
bit on four cards (`allreduce_test.FusedPushTest`). Measured on one tree in
one chain, control → fused, median step ms: one row 9.7 → 9.5; two rows (the
direct expert kernels to 48 pairs) 11.5 → 11.1 with a prompt per row and
11.2 → 10.8 with one prompt; three rows 13.1 → 12.8 and 12.8 → 12.5; five
rows 16.2 → 15.8. The per-class account of card 0 at one row (profiled, per
step): the 80 push kernels (7.9 µs each, 705 µs) are gone and the sum
kernels fall from 18.1 to 15.1 µs mean (1,449 → 1,211 µs; most of a sum
kernel is the wait for the peers), but the 80 pushing GEMVs grow by 7.1 µs
each (the step's 211 GEMVs 2,114 → 2,684 µs): a program's tile is 16 to 32
columns wide, so its peer stores cross PCIe as 32 to 64 byte pieces, and the
pass the push kernel made is paid inside the GEMV. Net about 0.4 ms a step
against the 0.6 sized, with the byte floor of about 8 µs a site still far
below the 23 µs the two kernels took. A note from the measuring: in the
eager four-card harness a cross-device `.to()` copy of a card's partial came
back stale while the kernels' sums on the card were right, so the test forms
its expected total on each card and copies nothing.

**Hero 15, the pushing GEMV's tile (Nov 3, measured on pod B).** The span a
program's epilogue stores into each peer is its tile width, BN weight rows a
program, so the tile is the knob (`GEMV_PUSH_BN`, the class key `push=1` for
the table; `python -m vxpu.allreduce measure-push` measures it at the step's
two mixing shapes: chains of 80 sites per card, captured and replayed
together, bit-checked on each card against the one-shot form). Measured on
a quiet pod, µs a site, GEMV + one-shot against the fused push at BN 16 /
32 / 64 / 128:

| site, rows | pushed a site | GEMV + one-shot | fused, BN 16 / 32 / 64 / 128 |
|---|---|---|---|
| attention (wo_b, bf16 partial), 1 | 30 KB | 20.8 | 16.8 / 15.6 / 16.1 / 18.6 |
| attention, 3 | 90 KB | 23.5 | 18.5 / 17.1 / 17.6 / 20.1 |
| attention, 5 | 150 KB | 26.6 | 21.9 / 18.9 / 18.9 / 21.4 |
| FFN (w2 with the routed sum, fp32), 1 | 60 KB | 16.1 | 15.2 / 14.7 / 14.9 / 15.6 |
| FFN, 3 | 180 KB | 19.2 | 18.9 / 17.6 / 17.9 / 18.5 |
| FFN, 5 | 300 KB | 22.2 | 22.8 / 20.6 / 20.9 / 21.4 |

(2 and 4 rows lie between.) BN 32 is best or equal everywhere and is the
pushing GEMV's default; 128 loses the GEMV's parallelism (40 programs for
5,120 columns). Against the GEMV's own tile of 16 that is 1 to 3 µs a site
at the attention sites and 0.5 to 2 at the FFN's, 0.07 ms a step at one row
and 0.2 at five. The ladder on one tree in one chain (pod B after its node
was replaced, launcher off, the direct expert kernels to 48 pairs above one
row), control → fused, median step ms, one prompt / a prompt per row: one
row 9.5 → 9.3; two 11.0 → 10.6 / 10.9 → 10.5; three 11.8 → 11.5 / 12.0 →
11.7; four 13.1 → 12.6 / 13.1 → 12.7; five 14.4 → 13.9 / 14.2 → 13.7;
tokens identical to the control at every row count, and the old tile within
0.1 ms of the new at one and three rows, as the per-site numbers say. The
sum part now costs 9 to 12 µs a site at the attention sites and 10 to 16 at
the FFN's against byte floors of 2 to 12: what remains is the flag round
trip and the wait for the peers, which only a step that no longer meets at
every site (the resident executor) removes. A four-row crash seen with the
fused push on this pod was not the fused push: the control aborted the same
way (glibc's "double free or corruption" in the step host's package loading
right after the capture, three of four first runs of the four-row artifact),
and the TP step workstream fixed it by constructing one AOTInductor package
loader at a time in the process.

**Hero 15, the direct expert kernels under expert parallelism: the compact
grid (Nov 3, measured on pod A).** The question was a few-pair kernel for the
drafted verify (2 to 5 rows, 12 to 30 pairs a layer), where the TP step's
profile put the routed experts 0.7 ms a row above their floor. Sizing first,
from the step's own per-layer kernel times: under EP4 a card holds about a
quarter of a layer's pairs (4.5 of 18 at three rows), and because rows of
different prompts share almost no experts, the held pairs are the distinct
experts; the per-layer spread across layers (held pairs vary, about 2 at p10
and 7 at p90) gives the marginal cost per held pair as 8.3 us for gate/up
(12.5 MB, 1.5 TB/s) and 3.9 for down (6.3 MB, 1.6 TB/s), the HBM read rate.
So a kernel that reads each distinct expert once reads what the direct
kernels already read, and what stands above the byte floor is a fixed term
per layer per card of about 21 us at every row count, 0.85 ms a step. The
microbenchmark (`dev/hero/bench_fewpair.py`: rows x 6 pairs launched, H held
on this card on distinct experts, the L2 defeated between calls, K split 2/2)
split it: the unheld pairs' empty programs, interleaved with the working ones
in dispatch order, cost about 14 us a layer (18 pairs launched and 1/2/4/6
held: 45.1/51.4/81.1/106.4 us against 25.4/37.7/67.4/92.5 with only the held
pairs launched); the in-kernel k-sum nothing; the rest is the two launches'
own ramp, chain and tail (gate/up about 4, down about 6). The compact grid
(`FP4_DIRECT_COMPACT`): the launch keeps its size, so a step graph captures
it, with the pairs on the slowest axis; each program scans the launch's pairs
in the slot map (two gathers of at most 64 entries) and takes the q-th held
pair's item, so the working programs dispatch first and the empty ones last;
the down kernel's token sum waits for the token's held count and the first
empty programs zero the rows of tokens with no held pair. Bit-identical to
the plain grid; 27.6/39.1/71.0/96.1 us at 1/2/4/6 held, a fixed term of
about 10.7 us a layer left. At the verify shapes (rows / held pairs / plain /
compact / GEMM form / floor, us a layer): 1 / 2 / 47.8 / 39.0 / - / 24.5;
2 / 3 / 62.1 / 53.7 / 111.4 / 36.7; 3 / 5 / 89.0 / 79.5 / 116.3 / 61.2;
5 / 8 / 131.8 / 123.2 / 145.3 / 98.0; 8 / 12 / 193.4 / 174.3 / 215.2 /
147.0. Measured and not taken: the gate/up programs prefetching the down
weights into the L2 (gate/up +5 to +19 us, down -1 to -4), an interleaved
compact order (the same), the compact grid under PDL (kernels longer), the
one-launch kernel (79/85/95 at 2/4/6 held), K split 4/4 (worse); K split 1/1
is better than 2/2 at 4 to 6 held (67.0/91.7 against 71.0/96.1) and worse at
2, a per-rows choice for the planner. The selection rule: under expert
parallelism the direct kernels run the compact grid up to `FP4_DIRECT_MAX`
pairs, since a card's pairs are distinct experts there; the GEMM form's
blocks of pairs sharing an expert pay for themselves only where rows share
experts (one prompt's rows, or the grouped widths above it). The floor in the
ledger is unchanged: the card's held distinct experts a layer times 18.8 MB
at the read peak, the record's distinct-experts statistic. What remains above
the floor after compaction, sized the same way: the two launches with nothing
held cost 8.1 us (the launch floor plus the dispatch of 8,352 exiting
programs; 11.1 on the plain grid), and against only the held pairs launched
the compact grid pays about 3.6 us at six held (96.1 against 92.5), which a
grid sized below the pair count would remove (programs looping over held
pairs q, q + G, ...; about 0.14 ms a step); the rest, about 7 us, is the two
launches' own ramp, chain and tail (one held pair alone: gate/up 13.3 us for
12.5 MB and down 12.0 for 6.3 at K split 2, 21.6 and 14.8 at K split 1, so
the split halves the chain; the k-sum about 1 us), which the per-pair
dependent launch could overlap (measured in the step ladder) and which goes
with the launches themselves in the resident executor. The step ladder (pod A, chain A37, rows of different prompts, the same tree,
`VXPU_FP4_DIRECT_COMPACT=1` against the control, 64 steps): one row 9.3 ->
9.2 ms, three rows 11.8 -> 11.6 (one prompt in three rows 11.6 -> 11.4), two,
four and five rows 10.5, 12.7 and 13.7 against the ladder of record's 10.7,
12.9 and 14.0; the tokens identical to the control at every width (the same
first-difference positions against the single-card reference). The 3-row
profile has card 0's expert kernels at 2.65 ms a step from 3.05 (gate/up
43.6 -> 40.0 us a launch, down 32.6 -> 26.2; the lightest layers' p10 24.2 ->
18.6 and 23.1 -> 14.8), the 0.4 ms the microbenchmark predicted, of which the
step shows 0.2: the routed experts run beside the layer's island branches,
so part of the kernel time saved was already overlapped. The compact grid is
on by default; with every pair held it costs the scan alone (93.9 against
92.9 us at 6 of 6 held, 171.0 against 173.1 at 12 of 12: inside the noise).

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

**Hero 15, the tensor-parallel step folded (Nov 2 to 3, measured on pods A and
B, four RTX PRO 6000 over PCIe, TP4 for the dense layers and EP4 for the
experts, one row unless said).** Hero 14 left the one-row step at 13.1 ms,
equal to the one-card control, with the executor's emulation of the same
graph at 8.2. The account, from a kernel profile of the sum-free step and
the step trace: 2,572 kernels a card a step (64 a layer) against the
emulation's 1,032 nodes, GPU busy 9.2 ms (fp8 GEMVs 291 x 11.5 us = 3.35,
the routed experts 1.9, the fronts' six kernels a layer 0.84, inductor
glue 0.62, the expert-parallel k-sum's adds 0.40, x_mxfp8 0.35, the rest
1.7), launch gaps 0.7, the 89 one-shot sums with their flag waits 2.2 (25
us a sum, 11 of it kernel), host dispatch 0.9 (inputs 0.10, thread start
0.16, copies 0.17, launch 0.18, join 0.31). Four folds and two switches
brought it to 9.4 ms with the tokens identical to the single-card
reference (the folds that change rounding keep the first 44 of 64 tokens):
(1) under expert parallelism the expert node sums its held pairs in the
kernel, not in a masked k-sum after it (12.9 -> 11.8 on A; about 160
launches); (2) the shared expert split across the ranks like a column-
and row-parallel pair and run through its fused front, its partial added to
the routed experts' before the one all_reduce (-> 10.6; fp8_linear calls in
a rank graph 170 -> 50; 35 MB a layer read once instead of four times); (3)
the attention's selection and output path placed under TP, attn_select and
a mix-free attn_out_partial whose partial the ranks sum in fp32 (-> 10.4;
kernels 2,572 -> 1,678, 42 a layer); (4) the ranks' sum mixes the stream in
its epilogue, tp_allreduce_hc, 80 of the 89 sums (-> 10.0; a rank graph's
aten mul 232 -> 72, sum 94 -> 14, copy_ 122 -> 42); then a K split for the
direct expert kernels, which at one or two experts a card ran about 200
latency-bound programs (gate/up 29.6 -> 21.8 us a layer; -> 9.5, tokens
identical for all 64), and one thread launching the four ranks' captured
graphs instead of a hand-off to four worker threads (-> 9.4; thread start
0.20 -> 0.02, join 0.24 -> about 0.1; on speculation's drafted step the
same change takes 0.12 ms off the drafter's call and leaves the verify
rows unchanged). The persistent rank threads were worth 0.6 ms over fresh
threads a step (10.6 against 11.3 at fold 2) and were cleared of the
"hangs" seen with them: every timed-out step but one was a first step
still building its AOTInductor packages past the 600 s step timeout (a
new tree, artifact or plan switch rebuilds about 293 a rank, serialized);
the first step now gets 7200 s. Knobs measured at one row: MXFP8_RECODE
-0.1 to -0.2 (290 launches fewer; +3 ms at 33 rows, so a per-rows choice),
SHARED_FFN_JOIN and FOLD_POST 0, FP4 one-launch and PDL 0 or worse, the
direct kernels' tiles (BN, BK, stages) 0 or worse, the reduce-scatter form
of the mixed sum 0 on pod A and -0.7 at 5 rows on pod B, whose cards sit
further apart on PCIe (B runs every multi-row shape 1 to 3 ms slower than A
and is kept for profiles).

The step at one row, with the sums, per kernel class (A, busy 8.80 of
the 9.5): the sums 1.80 (the mixing sum 80 x 13.4 us, of which about 5 us
is work and the rest the wait for the slowest peer's push; the push 89 x
8.2), fp8 GEMVs 211 x 10.1 = 2.13, experts 1.52, the fronts 0.83,
attention 0.47, gemv_pe 0.40, dual GEMV 0.38, x_mxfp8 0.31, row_post
0.22, glue about 0.5; host 0.3 with the launcher; gaps about 0.4. The sum
waits split into per-card expert imbalance 0.1 to 0.2 (the cards' expert
kernels 1.50 to 1.66 ms a step), launch skew 0.05 and the per-site
push-and-flag round trip about 0.8; of the forms that cut the last, the
fused push (the producing GEMV writes its partial into the peers' inboxes,
no push kernel) leaves about 13 us a site and applies to all 80 mixing
sites (0.6 ms a step; a separate workstream builds it), fewer sites per
layer applies to none (no two sums are adjacent without dependent compute
between them), and a lighter flag protocol saves 1 to 2 us a site.

The step by rows (A, rows of different prompts, which is how a drafted
verify's rows route): 9.4 / 10.7 / 11.8 / 12.9 / 14.0 at one to five rows
and 17.2 at eight, with the direct expert kernels up to 48 pairs
(VXPU_FP4_DIRECT_MAX=48; the GEMM form gives 12.0 / 12.6 / 13.7 / 15.1
at two to five). Per class, with the sums: the second row costs experts
+0.81 (gate/up 21.4 -> 33.2 us a layer, down 16.6 -> 24.7) and sums +0.42
(the mixing sum 13.4 -> 17.2 us a site, the push 8.2 -> 9.4); each further
row experts +0.7 and sums +0.3; the fronts +0.03 a row, GEMVs +0.01,
attention 0. The direct kernels' cost per held pair, read from the
per-layer spread of held pairs on one card (about 2 at p10 and 7 at p90
at three rows), is 8.3 us for gate/up (12.5 MB) and 3.9 us for down (6.3
MB): the HBM read rate, insensitive to BN, K split and stages (a first
reading divided a card's increment by all 18 pairs instead of the four to
five it holds and called it L2 traffic; the expert-kernel workstream
corrected it). What sits above the byte floor is a fixed term per layer
per card of about 21 us (gate/up about 8, down 13 to 15; an empty down
launch costs 4 us and the first held pair 13 more), the same at every row
count: 0.85 ms a step at one row as at five, the prize for a kernel without
it (the GEMM form's design, reading each distinct expert once, minus the
fixed costs that lose it 20 us a layer at one pair a block; its K split,
4ad7076, helps rows of one prompt, 11.7 -> 10.8 at three rows, and hurts
rows of different prompts, 12.6 -> 12.9).
Rows of one prompt (the same experts in every row): 10.4 / 11.7 / 12.0 /
12.4 / 13.1 at two to five and eight rows, GEMM form. reduce=skip is not a
step control above one row: with the sums skipped the tokens are wrong
from the first position and the routing is random (28 to 32 ms at five
to eight rows on B); the with-sums kernel profile works (the profiler
does not hang on the one-shot form) and replaces it.

Two folds closed by measurement rather than built. The GEMV pair that
shares the attention's input (attn_q's first GEMV and window_kv) already
runs as two parallel branches of the step graph: with no islands the
one-row step is 10.0 against 9.4, so the branches buy 0.6 ms, and a fused
launch of the pair could add at most the launch it saves, about 0.1 ms; the
shared expert's front as a third island (VXPU_ISLANDS=attn_q,window_kv,
shared_ffn_front) is 9.4 as well, tokens identical, so it stays off. The
fused push of the mixing sums (the producing GEMV stores its partial into
the peers' inboxes and raises their flags in its epilogue, no push kernel;
the second workstream's h15-fused-push, merged, VXPU_FUSED_PUSH=1) is
0.3 to 0.4 ms a step at every row count with the tokens identical: on pod
B, one row 9.4 -> 9.1, two rows 10.4, three 11.5, five 13.6 beside the
ladder of record; its 4-row run dumped core after the capture (the loader
race below, not the push). It went on by default on Nov 3 once its gates
passed: the drafted step's tokens identical with and without it on the
speculation workstream's pod C (42 of 82 accepted over 23 steps, the same
conditional acceptance 0.913 / 0.524 / 0.7 / 0.6 both ways, the drafted
step 16.09 -> 15.71 ms, each verify 0.2 to 0.4 faster), and twelve
four-row loads clean under MALLOC_CHECK_=3 on pod B. The ladder on the
merged head (pod B, Nov 3 13:00 UTC, different prompts, one thread
launching the graphs in turn): one to five rows 9.2 / 10.5 / 11.6 / 12.7 /
13.7 without it, 9.0 / 10.2 / 11.3 / 12.3 / 13.3 with it, rows identical
and the tokens identical to the control at every row count.

A regression to remember: at four rows, three of four first runs on the
rows15c-tp4-b4 artifact aborted in glibc ("double free or corruption
(fasttop)", "malloc(): unaligned tcache chunk detected") right after the
capture, with and without the fused push, inside torch's
AOTIModelPackageLoader (its metadata's JSON parse) as the four rank threads
each loaded their step host's expert nodes' packages at once; a rerun
loaded clean. The step host now constructs one package loader at a time in
the process (node_server.cpp, load_package), which costs load time only;
the four-row loads run under MALLOC_CHECK_=3 until a dozen pass clean.

A third of the kind "a module load or unload behind a spinning kernel":
the speculation workstream's sharded drafter at eight streams hung in the
first real step after its prime (pod C, Nov 3 evening): three ranks spun
on their flags while rank 2 sat in a compiled kernel's destructor
(CompiledKernel.__del__, the prime's solo programs' kernels collected as
the thread entered its step), whose module unload did not return while
the peers spun. The fix on their side: after the prime every rank runs a
collection and a device synchronize on its own thread with nothing in
flight; the capture already disables collection for the same reason, and
the Comm primes its kernels at setup. A first-step allowance of 7,200 s
hides such a hang for two hours; py-spy reads the stacks in a second.

A second regression from the same day: every load leaked its package
extractions. torch's package loader unpacks each .pt2 under TMPDIR
(aotinductor_XXXXXX/<key>.<pid>.tmp) and removes it only when the loader
object is destroyed, which an executor that exits with its loaders alive,
or is killed by a timeout, never does; a TP4 load leaves 1,172
directories, about 2 GB. Pod A's 200 GB tmpfs held 95,182 of them (177
GB) after a day, and every load then failed with "No space left on
device" (the 66- and 80-row runs of chain A38, then the soak). The engine
now reclaims the extraction directories of exited pids as its first act
(aoti.reclaim_extractions, before the lowering's cost measurements, which
compile with inductor and write under TMPDIR) and removes its own at
exit; it reads TMPDIR itself, since tempfile.gettempdir() falls back to
/tmp when the disk is full. The first load with it freed 177 GB.

The choices by rows belong to the plan, not the environment. The kernel
table now carries `knobs_by_rows` (a list of rows_max and knobs; the
engine applies the first entry whose rows_max covers its artifact's batch,
after the flat knobs; plan.knobs_for_rows, kernelplan.RowsKnobs), and the
planner writes the measured rule for Blackwell cards under expert
parallelism: through five rows the direct expert kernels up to 48 pairs
(FP4_DIRECT_MAX; 5 rows 15.1 -> 13.9 ms, 3 rows 12.6 -> 11.9, 2 rows 12.0
-> 10.7) with a K split of 2 (FP4_GU_KS, FP4_DN_KS; 10.0 -> 9.5 at one row,
4 and 3 give 9.6), the MXFP8 recode at one row only (9.3 -> 9.1; 0 at three
rows, +3 ms at 33); above five rows the GEMM form, which reads each
distinct expert once (8 rows 17.4 against 17.2 direct, and 13.1 against
18.4 when the rows share experts), with no split. The VXPU_ variables of
those knobs are A/B overrides of the table's values from here on
(apply_knobs leaves a knob the environment sets). The fused push joins the
rule the same way once its default is settled.

What the step is made of under tensor parallelism, for the planner: the
dense weights a quarter each, read once a step (2.13 ms of GEMVs at the
launch floor, 10 us each for 6.5 MB); the experts a card's held quarter,
read once a distinct expert (1.5 ms at one row, 2x the byte floor for
one or two experts a card); 89 sums over PCIe at 20 us a site; the fronts
and glue 1.5 ms; the host 0.3. A measurement that failed loudly: pod B was
killed by the container's memory limit mid-run (cause not established) and
pod A lost its node to a Spot preemption; both restaged from a pip freeze
and the PVCs, whose cold page cache makes the first load crawl (3 MB/s of
mmap faults) until a sequential read warms it.

**Hero 15, the many-row step (Nov 3, measured on pod A).** The plain TP4 +
EP4 step at 33, 40, 66 and 80 rows of different prompts (64 greedy steps,
median ms; different prompts / the same prompt in every row / one prompt;
pod D's 8-row point 17.6 everywhere): 33 rows 33.5 / 22.0 / 22.0; 40 rows
35.9 / 24.1 / 25.1; 66 rows 43.8 / 33.7 / 34.3; 80 rows 47.5 / 37.5 / 38.1
(988, 922, 744 and 684 tokens a second). The reduce-scatter sums
(VXPU_ALLREDUCE_SCATTER=1): 40 rows 34.0 / 22.1 / 23.1, 80 rows 43.7 /
33.6 / 34.2, so -1.9 and -3.8 ms, where they lose 0.5 at 5 and 8 rows
(15.7 against 15.1, 17.8 against 17.4). The scatter form reproduces the
one-shot form's tokens; against the one-row alone reference the GEMM
form's first differing token is 26 at 33 rows, none at 40, 4 at 66 and 80.

The account per card and step comes from with-sums profiles at 33 / 40 /
80 rows (card 0; the profiler's prompt set routes almost like one prompt,
22.7 / 25.1 / 38.2 ms before the profiler and 27.1 / 28.8 / 43.5 under
it, 1,928 kernels a step):
- experts (the GEMM form, 40 gate_up and 40 down): 3.9 / 4.1 / 7.4 at
  those prompts; with different prompts the ladder's same-against-all gap
  (11.5 / 11.8 / 10.0) is this term, 15 to 17 ms, every card reading its
  whole quarter each step from 33 rows up, at the whole-bank floor near
  15.6 ms;
- the pushes (89 sites at 54 / 64 / 121 us): 4.8 / 5.7 / 10.8 (the push
  kernel's form; the fused push, measured later on this ladder, is a loss
  here: 32.6 / 34.5 / 46.1 against 31.9 / 33.8 / 43.6 at 33 / 40 / 80 rows
  of different prompts, +0.7 / +0.7 / +2.5, the pushing GEMV's epilogue
  stores growing with the rows faster than the push kernel they replace,
  where it is -0.3 at one row, -0.2 at 8 and 0 at 16 (16.9 against 17.1
  and 21.8 against 21.8, tokens identical); so it is a choice by rows, on
  through 16, the plan's default following the artifact's batch); a card
  writes about 317 MB a step to its peers at 80 rows (bf16 partials at
  the attention sites, fp32 at the FFN sites, three peers each), 29 GB/s
  against a PCIe Gen5 peer-write floor near 50 GB/s;
- the step's first sum site: 4.6 / 4.6 / 5.0 ms of waiting on card 0 in
  these profiles (2.5 and 1.3 on cards 1 and 2, 0 on card 3), which is the
  profiler's, not the step's. Under the profiler the one launching
  thread's four cudaGraphLaunch calls take 1.3 ms each in sequence (5.4 ms
  for the four) and card 3's step starts that late; four threads launching
  at once overlap the calls (all start within 0.1 ms, each then lasts 3 ms)
  and the waits vanish, 30.0 -> 25.9 ms profiled, but the same two runs
  before the profiler attaches are 25.1 and 25.4 ms, and the timed ladders
  put the threads 0.3 ms behind the one thread at every row count. The
  call's host time is the profiler's per-kernel work (2.5 ms a call at one
  row, in a 9.2 ms step, 1,604 kernels), so without it the skew is below
  the noise and there is no launch term to cut; the account below closes
  without one. The other 88 sites wait 0 to 30 us each and the cards' busy
  time less their sums is equal (34.1 / 34.1 / 33.7 / 34.5 at 80 rows);
- the hc sums (80 at 30 / 34 / 41 us): 2.4 / 2.7 / 3.3, flag round trips
  mostly; the fronts (three kernels a layer): 2.7 / 3.0 / 4.9, of which
  front_dots 80 at 28 / 31 / 53 us (11 at one row, +0.5 us a row, a byte
  floor near 2 us); the dense GEMVs (211): 2.6 / 2.7 / 3.3; attention 0.8
  / 1.1 / 1.9; dual GEMVs, gemv_pe, the engram layers' cutlass GEMMs, ring
  writes and glue 3.3 / 3.5 / 5.5; the host 0.7 / 0.7 / 1.1 of span past
  busy and 0.83 between steps. At 80 rows of different prompts the terms
  sum to the step: experts 17, pushes 10.8, hc sums 3.3, fronts 4.9, dense
  GEMVs 3.3, attention 1.9, the rest 5.5 and the host 1, 47.7 against 47.5
  measured.

A token row past eight costs 0.64 ms from 8 to 33 rows and 0.30 from 33
to 80 on the ladder. The profiles' slope from 40 to 80 (0.36 a row
profiled) splits into the pushes 0.127 (floor 0.07 at 50 GB/s, 0.064
with the scatter form's bytes), the experts 0.08 until the bank saturates
near 33 rows, front_dots 0.045 (floor 0.003), attention 0.02, dense GEMVs
0.016, hc sums 0.015 and 0.05 of the rest; from 8 to 33 the experts'
saturation adds about 0.42 a row. The cuts sized from it, at 80 rows
(the launch skew is not one: the graphs launched from the rank threads at
once measured 33.9 / 36.2 / 47.9 against the one thread's 33.6 / 35.9 /
47.5 and 0.2 to 0.3 ms worse at one to five rows, so VXPU_LAUNCH_THREADS
stays 0): the scatter form from 16 rows (the planner's rule writes
ALLREDUCE_SCATTER_MIN_ROWS=16; measured on pod B at 16 rows 23.3 -> 22.9
with different prompts and 16.7 -> 16.3 with one, at 8 rows 17.8 -> 18.0,
so the crossover lies between 8 and 16 and 16 wins); front_dots with the rows in its grid (-2 / -2.3 / -4 estimated); the
FFN sites' partials pushed in bf16 instead of fp32 (-1.5 at 40 and -3 at
80 estimated at the measured rate, a numerics choice). The remaining
floor past those is the experts' whole bank plus the pushes' bytes:
about 22 ms at 80 rows against 47.5 today.

**Hero 15, the sums against B12X and the pull form (Nov 3, measured on pods
A and B).** The league fork's all-reduce (B12X, `comm/pcie`, a CuTe DSL
kernel: each rank's CTAs copy the input into the rank's own double-buffered
IPC slot, raise a sense-alternating system-scope flag in each peer, spin on
the peers' flags with one fence, then pull every rank's slot, sum in fp32
registers and store bf16; 256 threads times min(8, packs / 256) CTAs, 3 at
one row) was installed and measured on pod B by a parallel agent in 80-site
captured chains, bit-exact, us a site, bf16 [rows, 5120] at 1 / 5 / 8 / 16
/ 40 / 80 rows: 5.8-7.9 / 9.5 / 12-17 / 18.6 / 40 / 75, unchanged with a 21
MB GEMV feeding each site. Ours in the same kind of chain: the one-shot
plain sum 11.5 / 14.7 / 16.9 / 24.6 / 55.8 / 111.9, the scatter form 19.5 /
21.2 / 22.1 / 24.6 / 32.8 / 57.1 (ours ahead from 40 rows), and in the step
the fused push with the mix 13-15 at one row. Their MXFP8 dense GEMM is at
par with gemv_rows on our quarter-width shapes through 8 rows (10.7-10.9 us
both, about 975 GB/s, 1.6x a byte floor of 6.0-6.8 us a call, a correction
to the 2.6 this note used before), so no adoption; two planner facts from
it: a K split for narrow N at one row (5120 -> 1792: 112 programs under
188 SMs, about 1 us a call, 0.04 ms a step) and a 32-row tile from 33 rows
(1.3-1.5 us a call). Adopting B12X itself does not fit (a process per rank
under torch.distributed, the cutlass-dsl 4.7.1 dependency, no hc epilogue).

Our pull form built to their geometry (`allreduce.allreduce_pull_hc`,
`gemv_allreduce_pull_hc`: an export slot per card and site, the producing
GEMV writing it and raising the peers' flags from its epilogue, one kernel
of 3 to 8 programs waiting once then pulling every rank's slot in rank
order, the hc mix in its epilogue; bit-identical to the one-shot and fused
forms; the knob ALLREDUCE_PULL) measured in the fused push's harness, us a
site at 1 / 5 rows: attention site (bf16 wire) fused push 15.6 / 18.9,
fused pull 21.2 / 23.2 in the B12X geometry and 17.9 / 21.0 with 32
programs and one tile in flight; FFN site (fp32 wire) fused push 14.7 /
20.5, fused pull 54 / 166 and 32.6 / 46.0 (fp32 peer reads are
pathological from this kernel, bf16 reads are not, so the pull form carries
the bf16 wire class by construction and an fp32-wire variant costs 2x the
push); at 80 rows the bf16 pull 114 against the push 155, both behind the
scatter form's 57. With the spin removed (timing only) the push is 14.6
and the pull 20.2, so the barrier costs about 1 us in a chain and the rest
is the sum kernel's own work: about 8 us for local reads plus the hc mix
over 5,120 elements, 13 with PCIe reads. That is the gap to B12X's 6-8 us
(no hc epilogue, no output read back), and the next lever is the epilogue
itself: post and the 16 comb coefficients loaded once a row per program
instead of per element, programs laid along rows, sized at 3 to 4 us a
site, 0.3 ms a step at one row, for the push, scatter and pull kernels
alike. One piece of their spin landed on every form: the three flags
polled with one volatile vector load and the acquires taken once all are
up, instead of three system-scope atomics spinning in turn, -0.6 us a
site (fused push 15.6 -> 15.0, one-shot 20.8 -> 20.2), bits unchanged
(VXPU_ALLREDUCE_WAIT_VECTOR=0 the A/B back).

The sum comparison closed at par once the GEMV was taken out of our
number: B12X's all-reduce alone is 5.8-7.9 us a site at one row and 9.5
at five; our fused push's sum overhead beyond its 6.8 us GEMV is 14.7 -
6.8 = 7.9 at one row and 18.3 - 7.0 = 11.3 at five (the plain one-shot's
two launches 13.2 and 19). The mix epilogue with its coefficients loaded
once a block (VXPU_ALLREDUCE_ROWBLOCK=1) is -0.3 us a site and changes the
FMA contraction, so the stream differs by a last bit here and there; it
stays off. The sum kernel is not where the single-stream gap is.

The host's share, split from the same one-row profiles: the span beyond
the kernels' busy time is 1.25 ms a step profiled (about 0.8 without the
profiler), made of 1,600 to 1,700 gaps of under 1 us each between
dependent kernels inside the graph (about 1 us a boundary profiled, 0.3
to 0.5 without), plus 0.4 to 0.6 ms around a handful of copies: two
pageable host-to-device copies (0.23 and 0.12 ms), a device-to-pageable
copy (0.17) and device-to-device copies around the host nodes. So the
levers are kernels per step (x_mxfp8's 290 launches of 1.1 us into their
consumers' prologues, about 0.4 ms; the fronts' three kernels into one,
0.2; the glue's 300 small launches, 0.3). Pinned staging for the step's
host copies (the two step inputs built pinned, the argmax read back
through a pinned buffer; VXPU_PIN_HOST=0 the A/B) measured 0: 8.9 against
8.9 ms at one row and 13.6 / 13.3 / 13.3 against 13.5 / 13.3 / 13.3 at five
(pod B, one tree), so the copies' 0.5 ms in the profile overlapped the
graph or was the profiler's; it stays, harmless, and is not a term. The
x_mxfp8 fold's consumer-prologue form exists (MXFP8_RECODE) and loses
above one row, every GEMV program recoding its whole x tile at every K
step; the fold that pays is the producer's, the kernel that makes the
activation writing the codes and scales once, not built. Its shape from
the one-row profile: 290 launches a step in 251 runs (212 single, 39
pairs), consumed by gemv_rows (171), gemv_pe (40) and the dual SwiGLU
GEMV (40), produced by six kernels a layer that already write a bf16
activation on the fp8 grid: the row GEMV (41 runs), row_post (40), the
attention's combine (40), the norm's reduce (40), the SwiGLU rows kernel
(40), the front's last kernel (36), and act_quant (10). Each would store
the codes and scales beside its bf16 output (K + K / 32 bytes a row) and
the GEMVs would take them in place of calling x_mxfp8; about 0.4 ms a
step at one row for 290 launches of 1.1 us and their edges.

The fronts' fold as the existing one-pass pair of kernels (ffn_front,
two launches a layer, VXPU_FRONT_FORM=onepass) measured a loss on the
ladder (pod A, different prompts): 9.5 against 8.8 ms at one row and 14.1
/ 13.8 / 13.8 against 13.4 / 13.2 / 13.2 at five, with the greedy tokens
shifted (a rounding class); the three-kernel split form stays, and the
fold worth 0.2 ms is a one-kernel front with the split's parallelism, not
built. The split front's row tile is a table by rows (a program serves its rows
in turn, so a tall tile under-fills the card at few rows; the three
kernels a call on a quiet card, us, tiles of 1 / 2 / 4 / 8: 1 row 7.4 /
7.4 / 7.8 / 7.8, 5 rows 11.6 / 10.8 / 12.9 / 12.5, 8 rows 14.4 / 12.3 /
12.9 / 16.4, 16 rows 22.1 / 18.4 / 16.5 / 18.8, 33 rows 36.6 / 28.7 / 25.7
/ 25.5, 80 rows 69.0 / 55.0 / 48.1 / 44.8): 1 under 4 rows, 2 under 12, 4
under 24, 8 above. On the ladders the tile of 8 measured 33.2 / 35.4 /
46.4 against 33.8 / 36.3 / 48.1 at 33 / 40 / 80 rows of different prompts
(front_dots 56.9 -> 37.4 us a call at 80 rows, still 5x its floor). The
thin push (the attention partials' cast to fp32 dropped, 98bbc0b) on the
same ladder: 32.0 / 33.9 / 43.5 against 33.1 / 35.4 / 46.5 (same prompt
20.5 / 22.1 / 33.5 against 21.6 / 23.6 / 36.4), tokens identical, -1.1 /
-1.5 / -3.0 ms, at or above its sizing from the bytes; the many-row ladder
of record on the default tree is 32.0 / 33.9 / 43.5 at 33 / 40 / 80 rows.

The ladder with everything on by default (pod B, Nov 3 16:40 UTC, the
fused push, the thin push, the front tile table, the flag poll, the one
launching thread, pinned staging; different prompts): 8.9 / 10.2 / 11.3 /
12.4 / 13.3 ms at one to five rows, 9.2 at one row with the fused push off
in the same session; the one-row number of record is 8.9. An earlier
session on a tree without the flag poll and the tile table (15:50) read
9.2 / 10.7 / 11.6 / 12.9 / 13.7 with 9.3 off, and the 13:00 ladder on the
merged head 9.0 / 10.2 / 11.3 / 12.3 / 13.3 fused, 9.2 / 10.5 / 11.6 / 12.7
/ 13.7 control: the pod drifts by 0.1 to 0.4 ms between sessions, so a
record is a same-session pair.

For the drafted step's account at width (the speculation workstream, pod
C, Nov 3): the vocab-sharded Markov drafter at eight streams took the
drafter call from 3.51 to 2.44 ms a step at batch 8, the step 35.14 ->
33.93 ms on prose (639.6 -> 662.6 tok/s aggregate) and 37.31 -> 36.21 on
code (864.3 -> 890.5), the same tokens a row. And a regression to bisect:
the drafter call alone at one row reads 2.07 to 2.12 ms on the merged
head through 903b1bf against 1.71 on the head at 65b302b, class on or off
alike, the verify programs unchanged; the candidates between those heads
are the launch threads' default (6011555, on in that tree), the single
launcher (48eb1f6), the compact direct grid (10f7c28) and the thin push
(98bbc0b). The bisect by switch (pod B, chain B40, the same account on
tree13): base 1.985 ms a call, VXPU_LAUNCH_THREADS=0 1.706, which is
tree10's 1.71 and the current head's 1.70, VXPU_SINGLE_LAUNCHER=0 1.906;
so the cause was 6011555's default (the captured graphs launched from
four threads), reverted the same day in 97d663b: a fixed per-step hand-off
of about 0.3 ms that a small four-rank program pays in full, where the
single launcher itself is worth 0.08 the other way.
The whole one-stream drafted step on that head: 14.82 ms prose / 15.19
code, 204.0 / 305.4 tok/s against 193.1 / 288.9 on their tree12, the
five-row verify 13.39 against 13.95.

The push kernel measured alone (Nov 3 18:20, pod A, measure-push's
push-only form, chains of 80 sites, the FFN site's fp32 partial [rows,
5120] to three peers): 7.8 us at one row (60 KB), 20.3 at 8 (480 KB, 24
GB/s aggregate out of the card), 34.9 at 16 (960 KB, 27.5 GB/s), 91.3 at
40 (2.4 MB, 26 GB/s), about half the Gen5 floor; so a per-byte line of
3.4 GB/s read from a drafter trace was the site's waits, not the copy. A
program per (block, peer) or a 512-element block buys 5 to 10 percent
(18.5 / 31.8 at 8 / 16 rows against 20.3 / 34.9); the 512 block is now
the push's. What cost 3 us a site at one row and 3 to 5 at 8 to 16 was
the flag raise, three release.sys atomics; one fence.acq_rel.sys then
three relaxed flag stores (the same ordering for the readers' acquires)
gives push alone 7.2 -> 4.4 us at one row, the fused push 15.0 -> 12.4
(attention site) and 14.1 -> 11.5 (FFN), 25.1 -> 23.4 at 8 rows, in the
push kernel and the pushing GEMV's epilogue alike, on by default
(VXPU_ALLREDUCE_FENCE_ONCE=0 the A/B back). The fused push and the
one-shot form agree bit for bit at 7, 8, 9 and 16 rows on the step's path;
the harness's forced tile for the pushing GEMV is what differs at 8. On
the many-row ladder the fence is 0 (pod A, different prompts: 31.5 / 33.5
/ 43.3 against 31.6 / 33.5 / 43.3 at 33 / 40 / 80 rows, tokens identical),
as a per-site fixed cost should be where the bytes dominate. At one to
five rows (pod B, one tree, different prompts) the fence is 8.7 / 10.0 /
11.1 / 12.2 / 13.1 ms against 8.9 and 11.2 at one and three rows without
it, -0.2 at one row, the tokens matching the single-card reference at one
row as before; so 8.7 is the one-row number of record on the fence tree. The many-row ladder of record on the default
tree at the end of Nov 3 (the thin push, the front tiles, the fused push
off above 16 rows, the fence): 31.5 / 33.5 / 43.3 ms at 33 / 40 / 80 rows
of different prompts, from 33.5 / 35.9 / 47.5 in the morning's account.

The reference on our cards (main, Nov 4, the vLLM fork with DSpark on the
same four RTX PRO 6000; measured, two repeats identical): the plain
one-row prose step 7.18 ms (139.3 tok/s by their formula, the SSE path
included, so the engine step is at most that) against our 8.6 to 8.9;
plain 8 different streams 11.7 ms a step (85.3 tok/s a stream, no prefix
cache) against our 16.0 grouped / 16.4 direct / 17.0 GEMM at 8 rows of
different prompts. Their plain slope is 0.65 ms a row, which is the
no-sharing expert-byte floor (6 experts times 40 layers times 18.2 MB
over 4 cards at 1.6 TB/s is 0.68); their drafted 8-row step at about
10 ms gives 0.35 to 0.4 a row, the floor at the draft tree's 20 to 35
percent sharing. Their rows cost only bytes. Ours cost about 1.06 ms a
row on the plain step (8.6 to 16.0 over seven rows), 0.4 beyond the
expert bytes: the pushes, the sums, the fronts and the launch growth are
the whole plain slope difference, and the per-class split (the per-row
profiles on pod C: direct at 1, 3, 4 and 5 rows, grouped at 5) sizes
each. Their plain ladder on our cards (different streams, prose, their
formula): 7.19 / 8.18 / 8.88 / 9.34 / 10.18 / 11.86 ms at 1 / 2 / 3 / 4 /
5 / 8 rows, the first row +1.0 and about 0.6 a row after it; ours of
record 8.7 / 10.0 / 11.1 / 12.1 / 13.1 at 1 to 5 with the direct kernels
(9.9 / 13.0 at 2 / 5 and 16.0 at 8 with the grouped form). Their
fixed-depth drafted step (7 drafts verified every step, 8 rows) is 12.4
to 12.7 ms at 3.0 to 3.3 accepted, the direct pair for our 20.16 ms at
2.19 on the same shape.

Their plain one-row step profiled on our cards (main, rank 0, the median
of 38 steps, torch profiler): wall 7.38 ms, kernel union 7.15, idle 0.22
(3 percent); 1,780 launches a step in one graph launch; the busy sum 7.85
exceeds the union because the experts and the dense GEMMs run on side
streams inside the graph, about 0.7 ms overlapped. Per class, us a step:
dense GEMM with quant, MHC and norm 3,962 (pure dense GEMM about 2,240 in
248 launches at 5 to 10 us; MHC 670; MXFP8 row quant 122 x 1.3; grouped
RMS norm 85 x 2.5; bf16 small GEMV 46 x 5.1); experts 1,489 (40 x 25.3
W4A8 compact projection + 40 x 9.6, about 46 percent of the 0.68 ms byte
floor); attention 1,247 (MLA decode 38 x 13.4, top-k 41 x 3.7, rotary 98
x 1.5, the indexer); all-reduce 591 (81 one-shot B12X x 6.8 + 4 NCCL x
12); norm and elementwise 273; other 257; sampling 35. The cross-socket
rank reads all-reduce 558 against 591 with the same wall: no decode skew.
Two of their tricks for the backlog: the experts overlapped with the
dense GEMMs on a side stream inside one graph, and one graph launch a
step with no host hop.

Our one-row table in the same classes, taken on their node (pod C,
h0t7; pod A's artifact, direct form, the league prose prompt, the
median of the last eight profiled steps on rank 0). The un-profiled step
is 8.7 ms there (8.6 on pod A; the node is worth at most 0.1, settled by
running pod A's artifact on pod C: 8.7 against 9.5 for pod C's own
artifact, a stale export from Nov 2, fifteen minutes before the hc form
landed, whose graph has 89 plain sums with the mix as aten ops in place
of 80 hc sums and 9 plain ones, 1,763 launches a step against 1,443;
speculation's league artifacts carry the hc sites and are not affected). Profiled: span 11.60, kernel union 10.32, idle in span 1.28,
1,443 launches a step; the profiler adds 2.9 ms, nearly all of it in the
sum sites (38 us a site against 12.4 / 11.5 measured un-profiled), so the
sums below are the un-profiled per-site numbers and the other classes
the profile's kernel times. Ours against theirs, ms a step: dense GEMVs
2.90 (269 launches x 10.8 us) against 2.24 (248 x 5 to 10); quant 0.33
(300 x 1.1) against 0.16; fronts 0.85 (240 x 3.5) against their MHC 0.67;
norm and elementwise 0.14 against 0.48; experts 1.69 (160 x 10.6)
against 1.49; attention 0.95 (260 x 3.6) against 1.25; hc sums 0.60
measured (the stores, flags and sum kernel of 80 fused sites at 7.3 us an
FFN site and 5.6 an attention site, the fused site minus its GEMV, plus 9
pairs; the pushing GEMVs' bodies are in the dense class) against 0.59
(81 x 6.8 + 4 NCCL); misc 0.09 against 0.30; busy sum 7.55 against 7.85
(theirs overlaps 0.7 on side streams, and the profile shows ours
overlapping 0.47: busy 10.79 against a union of 10.32); idle plus host
hop, the step minus the busy sum, 1.15 against 0.22. The profile splits
the idle: 0.80 ms of gaps a step inside the span, of which 0.59 sits in
15 gaps around the step's memcpys (device to host, the sample, host to
device: the sampling hop) and about 0.2 in the 1,440 kernel edges at
0.15 us each, plus 0.09 outside the span. Of the 1.3 to 1.5 ms gap on
this node: the sampling hop 0.6 to 0.7, the dense GEMVs 0.66 per launch
with the same bytes, the small classes +0.2; the sums are at par, the
experts within 0.2 and attention 0.3 in our favour.

The one-row three-form account (pod C, pod A's artifact, the prose
prompt; un-profiled steps 8.7 direct, 10.1 the GEMM form forced on from
one pair, 10.8 the older tiled kernels; the GEMM and tiled forms are a
token class of their own, first difference at token 44 against the
reference). Kernel by kernel against direct, the sum sites excluded (the
profiler's skew lands there): the GEMM form's experts 240 launches x
10.1 us = 2.43 ms against 160 x 10.6 = 1.69 (+0.74: a schedule launch
and a SwiGLU-quant launch a layer more, the GEMM kernels about the direct
kernels' time), attention +0.05 (40 index gathers), norm and elementwise
+0.12 (120 launches: the gathers and scatters of rows to their experts),
misc +0.12 (40 reductions, the combine), 240 launches a step more in
all, about +1.1 of the +1.4 measured; the rest is the edges. Removable:
the schedule launch the way the grouped form's was (−0.2 to 0.3), the
SwiGLU quant into the down GEMM's prologue (−0.15), the gathers,
scatters and the combine into the GEMM kernels' epilogues (−0.3); the
fixed cost would then sit near 0.6 to 0.7, the bar at which the GEMM
form could be the one expert kernel from one row up; estimated, not
built. The tiled form's experts read 3.10 (+1.4 over direct) and it has
no claim.

The 8-row step on the league prose prompt, every row the same prompt
(pod C, pod A's artifact): direct 17.2, grouped 14.5, GEMM 12.8 ms,
against the reference's 11.86 on the same workload (their benchmark's
eight streams share the prompt and the text, so their experts sit under
the no-sharing byte floor). The rows rule alone picks the grouped
kernels there and loses 1.7 ms on exactly that workload, so the form
follows the sharing: the planner's table takes the measured distinct
experts a layer from the record (stats.distinct_experts) and chooses
the GEMM form from 6 pairs a distinct expert (8 rows of one prompt: 8
pairs, 12.8 against 14.5), the grouped kernels below it (5 rows of one
prompt: 5 pairs, 11.7 against 12.0; 8 rows of different prompts: 1.2
pairs, 16.0 against 16.8), one row direct and 16 rows and above GEMM
whatever the count; the by-rows table stays the default when no count
is recorded.

The sampling hop removed (Nov 4, VXPU_GRAPH_TOKEN, on by default; the
ladder below is its record): the captured step writes
its greedy choice into its own input_ids and advances its cache_position
at the end (`stepgraph.GRAPH_TOKEN`, in the capture after the choose
node; the host path overwrites both when it feeds a step, so either may
follow), each rank's graph is replayed on its own inputs with nothing
copied in (`StepGraph.relaunch`, `parallel.Decode.relaunch`), and the
tensor-parallel generate loop keeps one step queued ahead of the host:
rank 0's choice is copied to pinned memory behind each step with an
event, the host reads a step's tokens while the next one runs, the step
time is the GPU's event to event, and the last finish half runs at the
end (`Engine._tp_self_fed`). Two false starts before it measured: a
guard against host nodes (every step has 42, the expert and engram
nodes, whose inputs reach the step host through mailboxes captured in
the graph, never from the tensors the host passed, so the guard refused
every real step), and the rows loop (`_Rows`), which the tensor-parallel
path does not use. Measured, one row, direct form, tokens identical to
the reference: pod A 8.6 -> 8.1 ms (116.0 -> 125.2 tok/s), pod C 8.8 ->
8.2 (114.1 -> 123.3); the 0.5 to 0.6 the profile's gaps around the
step's memcpys had sized. The ladder (pod A, grouped form, one prompt /
same prompt / different prompts, off against on, tokens identical): five
rows 11.7 / 11.7 / 13.0 -> 11.8 / 11.3 / 12.5, eight rows 14.8 / 14.1 /
15.8 -> 14.9 / 13.6 / 15.3: -0.4 to -0.5 ms a step on the shared and the
different-prompt steps at every row count (the one-prompt arm, one live
row in a wider program, reads +0.1, within the pod's band). The serving path's native loop (C++, in-graph
choose) already had no Python between steps; the harness's TP loop did.
The drafted step is two graphs, the drafter's and the verify's, with the
host between them composing the draft into the verify's input and
choosing the verify program by draft length, and after the verify
reading the acceptance; its self-fed version needs the acceptance in
the graph (compare the verify's choices with the drafts, keep the
prefix, write the last kept token and the position into the drafter's
inputs) and either a fixed verify shape (the reference's: the 8-row
graph every step) or device-side choice of the program, which is the
resident executor's. Speculation built the fixed-shape form (cff4a11,
speculation.shape=fixed): at five verify rows it takes 0.68 ms a step
off with the text identical to the host path's (prose 178.0 -> 185.0
tok/s, code 296.7 -> 306.7 on pod B; under the confidence policy 216.8
prose and 306.9 code against 202.4 and 295.3 on the same pod and class).

The ranks' skew at the sum sites, measured without the profiler (Nov 4;
VXPU_SUM_TRACE=1: each hc sum's first program stamps the GPU's global
timer at arrival and when its peers' flags are up, `Comm.trace`; the four
GPUs' timers differ by constant offsets, so arrivals across ranks are
read against each rank's median and the quantity that costs, a rank's
wait, needs no calibration; hero-skew.py). Pod C, plain step, different
prompts, three runs each: the sum of a rank's waits a step is 0.6 to 0.9
ms at one row, 1.1 to 1.3 at three and 1.35 to 1.6 at five, about 0.2 ms
a row, nearly half of the 0.45 a row that is not bytes, and 0.7 ms of the
start term. 41 hot sites a step, the same every run: every other site,
the 40 FFN sums after the routed experts (the attention sums wait under
1 µs). The waits come in quanta of about 12 µs (one expert's 18.2 MB at
1.6 TB/s is 11.4), and which rank waits changes from site to site, so
the cause is the uneven routed pairs per card under expert parallelism,
not the launch order (the first site is not hot, rank 3 is not uniformly
last), the engram nodes or the cross-socket rank. The arithmetic: at one
row 6 pairs land on 4 cards and the step pays the fullest card, about 3
pairs against a mean of 1.5, 1.5 x 12 µs x 40 layers = 0.7 ms; at five
rows 30 pairs, about 10 against 7.5, 1.2 ms. The profiler's version of
this (ranks 0 to 2 waiting 0.86 ms a segment for rank 3) was its own
launch cost in launch order and is retracted. Sized fixes, estimated: a
placement that balances the routed load per layer from the measured
routing (hot experts replicated on two cards, routed to the lighter one)
for about half the spread, 0.3 ms at one row and 0.5 to 0.6 at five;
splitting each pair's expert over two cards halves the quantum at the
cost of a second sum a layer, which hero 14 priced above its gain.

The per-row table at matched sharing (Nov 4; the league prose prompt on
every row, the reference benchmark's shape; pod C; kernel times from the
profile, the sum sites and the waits apart because the profiler's cost
lands in them). Kernel ms a step, ours at 1 row (direct), 3, 5 and 8
rows (grouped) and 8 rows (GEMM form), then theirs at 1 and 8 rows:
dense GEMVs 2.90 / 3.02 / 3.18 / 2.75 / 2.79 against 2.24 / 7.12 (their
main GEMM goes 7.5 to 63.5 µs a launch from one row to eight; ours stays
at 10 to 12 µs for the same bytes); experts 1.73 / 2.51 / 3.11 / 4.09 /
3.57 against 1.49 / 4.90; attention 0.94 / 0.98 / 0.96 / 0.89 / 1.03
against 1.25 / 1.37; fronts, quant and norm 1.31 / 1.37 / 1.43 / 1.61 /
1.81; launches a step 1,443 / 1,483 / 1,483 / 1,599 / 1,699 against
1,780 / 1,876. The un-profiled steps on that prompt: ours 8.1 / 10.8 /
11.3 / 13.6 (grouped) and 12.8 (GEMM) at 1 / 3 / 5 / 8 rows, theirs 7.19
/ 8.88 / 10.18 / 11.86. At eight rows our compute kernels sum to 9.4 ms
against their 14.0 and the step is 13.6 against their 12.1 wall: what is
outside the compute kernels costs 4.2 ms here against 1.5 there (81
all-reduces at 13.9 µs and 0.24 of idle), the 80 sum sites at about 20
µs un-profiled and the waits, which with every row routed to the same
six experts are the one-row imbalance times eight (a card holding three
of the six does 24 pairs while one holding none does nothing), 2 to 2.5
ms a step. That is the benchmark's shape and the case the tensor-parallel
experts remove. From one to five rows the compute classes grow 0.3 ms a
row (the experts 0.35, the rest flat); the rest of the slope is the sums
and the waits.

The tensor-parallel experts (Nov 4, VXPU_EXPERT_TP=4, off by default
while the class is decided): every card holds a quarter-slice of every
routed expert instead of 96 whole ones (experts_gpu.Pool part: the gate
and up banks' rows of its intermediate range with their scales, the down
bank's packed columns of the same range; 72.2 GB a card either way), does
that share of every routed pair in one launch a layer, and its down
output is a K-partial that the FFN hc sum already adds (under EP4 the sum
added the cards' different pairs' partials the same way), so the per-card
load is the mean by construction and no collective is added. Measured on
pod C, EP4 against TP4 (the down kernel's K split 1 for the 576-wide
reduction), one prompt / same prompt / different prompts: one row 8.2 ->
7.6 ms (123.1 -> 133.1 tok/s); five rows 11.9 / 11.4 / 12.5 -> 10.5 /
9.8 / 11.4: the imbalance term removed as sized (0.7 and 1.5). The
tokens are a numerics class of their own: at one row the first 26 equal
the single-card reference and the text then takes another plausible
continuation (29 of 64 equal); at five rows the first differences sit at
tokens 4 and 26 where EP4's five-row class sits at 28. The cause is the
order of the routed pair's down accumulation (four K-quarters summed in
fp32 in the collective instead of one pass over K = 2304), the same
change the dense wo_b carries; the intermediate's block quantization is
unchanged (576 is 18 blocks of 32). The class is arbitrated as the bf16
wire's was: by the league's acceptance and text groups, not by the
one-row reference. At width (pod C, EP4 against TP4, one prompt / same
prompt / different prompts): 8 rows with the grouped form 15.0 / 13.7 /
15.3 -> 12.4 / 11.4 / 13.9 ms, the same-prompt step at 712 tok/s under
the reference's 11.86 on its own benchmark shape; 33 rows with the GEMM
form 20.2 / 19.6 / 31.5 -> 18.8 / 18.0 / 28.9, the plain 33-row record of
31.5 reading 28.9 (1,132 tok/s), the sized spread of 2.2 gone and 0.4
more. The ladder under TP4 experts on different prompts: 7.6 / 11.4 /
13.9 / 28.9 ms at 1 / 5 / 8 / 33 rows against 8.2 / 12.5 / 15.3 / 31.5
under EP4. On the reference benchmark's own shape (the league prose prompt on all
8 rows) the GEMM form under TP4 reads 10.9 ms, 745 tok/s, against EP4's
12.0 and the reference's 11.86 (682 tok/s in all), the text identical
across rows and to the one-row run; on 8 different prompts the GEMM and
grouped forms tie at 13.9 under TP4. The same-prompt ladder under TP4:
7.6 / 9.8 / 10.9 ms at 1 / 5 / 8 rows against their 7.19 / 10.18 /
11.86, ahead from five rows and 0.4 behind at one. By these numbers the
tensor-parallel experts are the default at every row count once the
class is accepted; the expert form's rules by rows and sharing stand
inside it, the slicing being orthogonal to the form. The class at
width (speculation, pod A, 16 streams, the five-token drafted programs,
budget policy, bf16 wire): prose 1,085.3 -> 1,125.8 tok/s (39.96 ->
36.98 ms a step), code 1,314.3 -> 1,355.6 (49.14 -> 46.70), acceptance
1.65 -> 1.59 prose and 3.36 -> 3.38 code, within the band; the texts
move at character 48 and 45, the class. One stream (pod A, the fixed
five-row self-fed shape, direct class): prose 209.9 -> 239.0 tok/s (the
step 14.09 -> 12.55 ms: the verify span 12.77 -> 11.34, the drafter 1.3
-> 1.2), code 319.3 -> 358.7 (14.21 -> 12.66); acceptance 1.99 -> 2.03
prose with the text forking at character 192 (the class), 3.59 both on
code with identical text. Eight streams (pod D): code 991.8 -> 1,069.9
and 1,067.5 (identical text over 300 characters, acceptance 3.55 both),
prose 676.3 -> 667.5 (the step 33.85 -> 32.51, but that prompt's text
under the class accepts 1.89 against 1.96, so the entry does not move).
The class holds everywhere by acceptance and text repeatability and wins
everywhere by step time; the one aggregate it loses is 8-stream prose,
by the prompt's text. The one-stream prose entry at 239.0 stands 11
short of the 250 bar. The verdict (Nov 4): the 132-prompt line with the
class on is in band, accepted 1.459 against 1.469 a step (−0.7 percent),
tokens a step 2.433 against 2.444, decode 194.7 against 173.2 tok/s
(+12.4 percent), so the class is accepted. The user's decision (Nov 4): the
tensor-parallel experts are a plan option the planner selects where a
workload's measurement chose it, not a default (`plan.EXPERT_TP_DEFAULT
= 0`; the league entries carry `expert_tp = 4` explicitly). The reason:
the placement gives up strict accuracy against the single-card
reference, and it is the first of many optimizations that will
(quantization among them); LLMs are approximation algorithms, so the
planner selecting an approximation is a good thing, but it is a
selection, not a default.

The attention front's two projections joined (Nov 4, VXPU_QKV_JOIN=1,
off: a recorded variant): attn_q and window_kv of a layer, which read
the same x, become one op (vxpu::attn_qkv, lower.join_qkv, the join
seeing through the alias nodes an earlier pass wraps x in) whose two
down-projections are one GEMV launch over the joined weight [1792,
5120], the per-row kernels reading their slices by a row stride. In a
captured graph the pair reads 10.5 µs as two launches and 7.7 as one
(2.8 µs a layer, 0.11 ms a step); in the step it loses 0.5 ms a step at
one row (8.1 -> 8.6) and at five (11.8 / 11.3 / 12.5 -> 12.2 / 11.7 /
13.0), tokens identical. The loss is a graph effect, not a kernel one:
window_kv ran on the step graph's parallel branch beside the q path
(hero 8's overlap) and the joined op puts both on one node. With the
branches off on both arms (one row): the pair 8.3, the joined form 8.5,
so 0.3 of the 0.5 is the branch the join undoes and 0.2 is the joined
GEMV itself, slower in the step than the two launches it replaces where
the captured microbenchmark had it 2.8 µs a layer faster. Closed as a
negative: the pair's gain was sized at 0.11 ms a step and a join at an op
boundary costs the graph more than the launch it saves.

The rule set at the end of the tensor-parallel work (Nov 4), in one
place (plan.go's Explain carries the same lines): the expert placement
is whole experts per card by default and the tensor-parallel experts
(`expert_tp = 4`) where a workload's measurement selected them (the
user's decision: an approximation the planner selects, not a default;
the class accepted on the 132-prompt line); the expert form is the direct kernels
at one row, the grouped kernels from two rows, the GEMM form from 16 rows
or from 6 pairs a distinct expert when the record carries the count; the
fused push is on through 16 token rows (batch times step_tokens) and off
under the bf16 wire class; the self-fed step is on; the bf16 wire is a
class per workload; the pull form and the joined attention front are
closed. The ladder of record under that set with TP4 on (pod C, one chain,
Nov 4), ms a step, ours against the reference on our cards: different
prompts 7.5 / 8.9 / 11.4 / 13.9 / 19.3 / 28.9 at 1 / 2 / 5 / 8 / 16 / 33
rows (134 / 230 / 449 / 589 / 858 / 1,132 tok/s) against their 7.19 /
8.18 / 10.18 / 11.86 at 1 / 2 / 5 / 8 (theirs at 16 and 33 unmeasured);
the same prompt on every row 8.1 / 9.9 / 11.4 / 13.0 / 18.0 at 2 / 5 / 8
/ 16 / 33 (the 8-row point 712 tok/s, 745 with the GEMM form); the
league prose prompt 7.5 at one row and 10.9 at 8 rows with the GEMM form
against their 7.19 and 11.86 on that shape. The 16-row point ran with the
fused push off on that tree and reads 14.1 / 12.9 / 19.2 with it on. The
drafted entries under the placement: one stream 239.0 prose and 358.7
code tok/s, eight streams 1,068.7 code (prose class off at 676.3),
sixteen streams 1,110.0 prose and 1,393.2 code (speculation). The remaining gap at 8
different rows is the sum sites (about 1.5 ms against 1.13) and the
small-kernel floors (about 1.9 against 1.0), the latter a compiler-level
fusion by size of work.

The dense GEMVs against their byte floor (Nov 4; gemv-floor.py: each
projection shape of a layer, one rank's shard, as gemv_rows in a captured
graph with distinct weight copies rotated past the card's 126 MB L2; a
first run that replayed one weight read it from L2 at up to 1.9 TB/s,
above the DRAM peak, and is discarded, as is the hero 11 figure of 87
percent for the GEMVs, which holds only with the weight in L2). µs a
launch at one row, GB/s, percent of the 1.6 TB/s floor, the narrow K
split: wq_a 1280x5120 8.1, 838, 52; wkv 512x5120 6.4, 423, 27; wq_b
8192x1280 11.0, 987, 62; the indexer's wq_b 1024x1280 4.1, 330, 21; wo_a
1024x8192 10.2, 845, 53; wo_b 5120x1024 8.1, 670, 42; weights_proj
64x5120 4.5, 75, 5 (a latency floor of about 4 µs); five rows within 0.6
µs of one. (The shared expert is fp4 like the routed experts and belongs
to the expert kernels; a first version of this table carried it as fp8
with double the bytes.) The sum is 52 µs a layer of GEMV time for 36 MB
whose floor is 22.5 µs: 1.2 ms a step over the floor at one row and the
same at five, row-invariant. Beside the reference's own DenseGemm
launches a step at one row (its profile, rank 0): 80 x 7.5 µs, 48 x 8.8,
40 x 9.8, 40 x 9.3, 40 x 5.1 and 46 x 5.1 of small-N GEMV, 2.22 ms, 56
µs a layer, their 6 to 11 MB launches at 56 to 69 percent of the floor:
per layer the two stacks are at par on the dense GEMVs, and the 1.2 ms
over the floor is a term both carry. The shapes under 60 percent have a
short K (wo_b, wq_b, wo_a: 1024 to 1280, few K steps a program) or a
narrow N (wkv, the indexer's wq_b, weights_proj: too few programs even
with the split); the tile sweep per shape (BN, BK, warps, stages, the K
split) in this bench is the cheap first step, the table carrying what it
finds; the honest size is 0.3 to 0.6 ms a step if the kernel reaches 75
to 80 percent where both stacks sit at 60. The sweep (gemv-sweep.py:
every tile the kernel accepts over BN, BK, warps, stages and the K
split, 142 to 360 a shape, rotated weights, one row): wo_b 8.1 -> 6.8 µs
(BN 128, BK 64, K split 4), the indexer's wq_b 4.1 -> 3.3 (BN 32, BK 64,
split 4), wq_b 11.0 -> 10.7; wq_a, wkv, wo_a and weights_proj unmoved,
their defaults the best of every tile. That is 2.4 µs a layer, 0.1 ms a
step at every row count, in the planner's blocks table; the kernel's
structure (a program owns 16 weight rows and streams K in 256-byte
steps) is what holds the dense GEMVs at 52 to 62 percent, and the path
to the floor is a kernel with more bytes in flight a program, hero 16
work beside the small-kernel fusion since both stacks sit at the same 60
percent and the term never separated them.

The ledger's floor claims against the L2 rule (a weight-streaming bench
must rotate distinct copies past the L2): the hero 11 "GEMVs at 87
percent" replayed one site's weight and is retired; the per-pair expert
figures come from the step's profiles with the routing changing the
experts every step and 18 MB a pair, and stand; the sum kernel's 3.7 µs
and the push's 24 to 35 GB/s are PCIe and latency, and stand; the B12X
at-par comparison was both forms in one harness on the same buffers,
and stands as a comparison, not as a floor; the fused push's per-site
totals include a GEMV body measured on one repeated weight (4.2 µs for
the FFN site's wo_b, 8.1 from DRAM), so the per-site absolutes understate
by about 4 µs while every step-level A/B on the fused push, the fence and
the rows rule stands; the fronts' tile table and the front fold were
step-level and stand; the qkv pair's 2.8 µs microbenchmark was one
repeated weight and is superseded by the step.

The five-row step under the tensor-parallel experts, profiled (pod C,
Nov 4, different prompts, the grouped form): the expert kernels 200
launches x 24.6 µs = 4.92 ms a step against a byte floor of 3.4 (30
pairs x 4.5 MB a card a layer, 135 MB, 84 µs; no sharing on these
prompts), where the whole-expert kernels under EP4 read at their floor
(3.1 against 3.3 ms for card 0's own pairs): the sliced kernels run at
about 70 percent of the bandwidth the whole experts reached, each pair's
program doing the same prologue, x and scale loads and down reduction
over a quarter of the bytes, 1.2 to 1.5 ms a step at five rows, which is
why the step gained 1.1 to 1.6 of the 1.5 ms imbalance it removed. The
rest: dense GEMVs 3.15, attention 1.02, fronts 0.95, hc sums 0.68 (6.8
µs a site and no waits: the imbalance is gone), 1,662 launches, idle in
span 0.70. The two knobs at five rows under TP4, off -> on (one prompt /
same prompt / different prompts): FOLD_POST 10.6 / 9.8 / 11.4 -> 10.6 /
9.8 / 11.3, MXFP8_RECODE 10.6 / 9.8 / 11.4 -> 10.5 / 9.7 / 11.4, tokens
identical, within the band. The direct expert kernels' tiles for 576-row
slices, one factor at a time on the five-row step under TP4 (one prompt
/ same prompt / different prompts; base 10.6 / 9.8 / 11.4): the down
kernel's BN 64 reads 10.2 / 9.5 / 11.3 and BN 128 10.2 / 9.4 / 11.4,
tokens identical, so the down kernel widens its N tile to 64 when its K
is a slice (I at or under 1024; the environment overrides); its stages 4
tie and warps 4 lose 0.5; the gate-up kernel's BN 64 loses 0.5 to 1.1,
BN 128 loses 6, BK 512 loses 0.8, and its K split of 1 (10.4 / 9.6 /
11.8) is a class of its own (its tokens are the GEMM form's); the GEMM
form under TP4 at five rows reads 10.7 / 9.6 / 11.8, so the form rule
stands. The slice's fixed cost on different prompts stays at 1.1 to 1.4
ms a step at five rows after the tile; what removes it is a kernel that
amortizes a pair's prologue over more of its bytes (two pairs a program,
or the GEMM form's amortization without its launches), hero 16 work
beside the dense GEMV.

The down kernel as a programmatic dependent launch (FP4_PDL 2: launched
while the gate-up grid drains, each pair's programs waiting on their own
arrival counter; a hero 11 knob left off then). Speculation measured it
on the drafted one-stream step under TP4 experts: prose 243.8 -> 253.1
tok/s (the step 12.30 -> 11.84 ms) and code 365.9 -> 379.7, texts and
acceptance identical. On the plain step under TP4 (pod C, one prompt /
same prompt / different prompts): five rows with the direct kernels 11.3
/ 10.3 / 11.3 -> 10.8 / 9.9 / 10.9, tokens identical; with the grouped
form 10.5 / 9.8 / 11.4 -> 10.5 / 9.8 / 11.3 (it takes no arrival
counters, so the knob degrades to the whole-grid wait there); one row
7.6 -> 7.7, within the band (six pairs leave nothing to overlap). It
reaches the direct two-launch path only: the GEMM form at width has no
dependent launch, so the knob is inert there, not unsafe. The planner's
2-to-8-row entries are now the direct kernels with FP4_PDL 2 (direct with
the dependent launch beats the grouped form by 0.5 on different prompts
and loses 0.1 to 0.3 on one prompt's rows), the grouped form the sharing
rule's middle band (2 to 6 pairs a distinct expert) and the GEMM form
above it; the follow-up is the grouped form's own per-pair counters, so
that the shared case takes both. The one-launch form stays off (a class
of its own, +0.05 ms).

The width terms for the code lines (Nov 4; 40 and 80 token rows under
the tensor-parallel experts with the GEMM form and the bf16 wire class,
the drafted code step's 48 to 96 rows bracketed; pod A, the EP4 pair on
pod C). Kernel ms a step on rank 0, profiled, TP4 against EP4: the expert
kernels 6.4 / 8.4 against 10.2 / 9.3 (gate-up 3.75 / 4.94, down 1.70 /
1.99, the shared expert 0.67 / 1.12; the EP4 card's own imbalance sits
inside its kernel time), growing 0.05 ms a row between 40 and 80 as the
distinct experts saturate; dense GEMVs 3.47 / 4.58; fronts 2.21 / 3.64
(the dots kernel 21 / 38 µs a launch, +0.045 ms a row); attention 1.77 /
2.68; quant, norm and misc 1.4 / 1.7; the sum sites 80 at about 50 and
90 µs un-profiled. The un-profiled steps 23.4 / 31.9 under TP4 against
24.8 / 34.5 under EP4 on the same prompts; the class is worth 1.2 to 1.5
ms at 40 rows and 2.8 at 80. The first cut, measured: the reduce-scatter
sums, which the planner's rule names from 16 rows but which no run took
without the table (the ladder of record at 16 and 33 rows included):
under TP4 with the class 29.4 -> 28.6 ms at 40 rows of different prompts
(18.2 -> 17.5 on one prompt's rows) and 38.5 -> 36.5 at 80 (27.3 ->
25.5), tokens identical, so the form is class-neutral; the knob's module
default is now the rule's 16 rows. Next on the list: the GEMM form's down
kernel over the 576-wide slice (the direct kernel's 64-wide K-step loss,
0.4 to 0.6 ms at width), then the fronts' row tile above 24 rows.

The start term, ranked (main's three questions, Nov 4):
1. The sampling hop, 0.6 to 0.7 ms a step at every row count. The plain
   step is one graph launch a rank a step already; what is not in the
   graph is the token: a device-to-host copy, the sample on the host and
   a host-to-device copy before the next launch (the 15 large gaps). The
   one-graph form at one stream needs no resident executor: a greedy
   argmax kernel inside the graph writing the next token into the
   graph's own input, and the hop is gone (0.5 to 0.6 estimated at one
   row and at five, a per-step term). The drafted path is two graphs,
   drafter and verify, with a host hop between them because the
   accepted length picks the next shape; one graph there needs a fixed
   verify shape or device-side control, which is the resident executor's
   territory.
2. The dense GEMVs, +0.66 against theirs with the same bytes. The
   honest comparison is dense plus sums: ours 2.90 + 0.60 = 3.50 against
   their 2.83, +0.67. The split: the sum kernel itself is cheaper than
   theirs (3.7 us a site measured against their 6.8 one-shot, -0.25 over
   80 sites); the peer stores and flags on the pushing GEMVs +0.2 to
   +0.3 (2 to 3.6 us a site, 80 sites; in the profile the 80 pushing
   GEMVs read 10.5 us and the 131 plain ones 12.1, so the stores are not
   the per-launch cost); the GEMV bodies and launch count +0.7 (269
   launches at 10.8 us against 248 at about 9: the narrow projections
   are at the launch floor, not the byte floor). The cut is fewer, wider
   launches: the 7 dense multiplications a layer fused by the planner
   at op boundaries (hero 8's templates), sized at 0.5 ms if the mean
   falls to theirs; estimated, not measured.
3. The side-stream overlap, their 0.7 ms. Our counterpart is the step
   graph's parallel branches (on by default since hero 8: the
   attention's two projections on a branch, 14.0 to 13.7 at one row),
   and the profile shows 0.47 ms overlapped today. What is blocked: the
   step is a chain at the node level; the shared expert cannot join a
   branch because its op takes the routed experts' sum in its epilogue,
   and the fronts, quant and row_post kernels (2.1 ms of launch-floor
   kernels) sit on the dependency chain. The sized gain is small at one
   row: the routed experts read at bandwidth (about 27 MB a card in 15
   us), so hiding the shared expert's 10 us a layer under them buys the
   tails, 0.1 to 0.2 ms, and the same at five rows; the larger gain is
   the launch-floor kernels moved off the chain by the planner's node
   cuts, 0.3 to 0.5 estimated and unmeasured. Ranked below the hop and
   the GEMV launches because both of those are per-step terms we can
   remove without a new executor.

A rule for every exchange, from a hang in speculation's embedding fetch
(Nov 4): a flag between cards is raised from inside the kernel that wrote
the data, a device store or atomic through the peer mapping after the
fence, never by a separate stream write to the peer's flag address. The
fetch raised its per-peer flags with the driver's write-value on the
peer's pointer (as the engine's ring-write-off branch does); after the
prime, rank 0's site-0 push was never seen by its peers and all four
ranks sat in the flag wait, first on hero7c and then again on pod C with
the same code that had just run 157 steps, so it is a race between the
stream's flag write and the kernel's data stores, not a pod's topology;
the ring path's in-kernel flag store has worked everywhere and the fetch
now raises its flags that way. The sums here comply: the push kernels, the pushing GEMV's
epilogue and the publish kernel raise their flags with system-scope
atomics inside the kernel after one fence (`_raise_flags`); the only
host-side fills are the priming tensors.

The sites per form, with the fence default, us at 1 / 8 / 16 / 40 rows
(pod A, measure-push). The FFN site (fp32 partial with the routed sum
added), fp32 wire: GEMV 4.2 / 4.9 / 5.2 / 8.2, push alone 5.0 / 15.8 /
28.6 / 66.4, the one-shot site 12.9 / 24.6 / 41.1 / 82.4, the fused push
11.5 / 23.3 / 39.4 / 87.7; so the sum kernel itself is 3.7 / 3.9 / 7.4 /
7.8 a site (a drafter trace's 37 to 112 us for it was the site's waits).
The same site with the bf16 wire class: push alone 4.4 / 9.2 / 15.7 /
34.7, the one-shot site 12.3 / 20.0 / 25.5 / 50.0 (-4.6 / -15.6 / -32.4 at
8 / 16 / 40 rows), the fused push 11.5 / 23.8 / 37.2 / 102.7 (its epilogue
pushes fp32 whatever the class), which is the fused push's loss at width
seen from inside. The attention site (bf16 partial): GEMV 6.8 / 7.0 / 8.7
/ 10.5, push alone 4.4 / 9.3 / 16.3 / 35.3 (1.2 MB a site in 35 us at 40
rows, 35 GB/s out of the card, near the practical Gen5 figure, so the
push's layout is closed), the fused push 12.4 / 21.1 / 32.5 / 67.3, the
fused pull 10.9 / 19.6 / 26.2 / 50.9 for its sum alone. What remains per
site at width over B12X's 40 us is the sum's 10 to 15 us (the wait, three
reads, the mix), 0.9 ms a step at 40 rows over 89 sites.

The grouped form's first build, split at one row (pod B, same prompt, per
launch): gate/up 20.2 us against the direct kernels' 19.3, the down
kernel 48.9 against 14.4 (its SwiGLU, clamp and quantization over the
whole 16-row tile at every K step), the hc sums 32.5 against 18.4 a site
(the peers waiting on the slowest card's down kernels): 12.0 against 8.9
ms. At sixteen rows the per-program scan took over. The revision does the
SwiGLU per member row (MT = 4) and reads the schedule table. Measured (A48,
pod A, one tree, one prompt / same prompt / different prompts, against
the direct kernels): one row 9.8 against 8.8; two rows 11.4 / 10.0 / 11.4
against 10.2 / 10.0 / 10.1; five rows 12.4 / 12.4 / 14.6 against 13.4 /
13.2 / 13.3 with a tile of 4 pairs, 12.2 / 12.2 / 13.7 with 2, 14.7 / 12.9
/ 18.4 with 8; tokens identical. So it wins 0.8 to 1.2 ms at five rows
where the rows share experts and loses 0.4 (tile 2) to 1.3 where they do
not, and costs 1.0 at one row where there is nothing to share: the tile
defaults to 2, one row never groups, and the form stays off (the knob
FP4_DIRECT_GROUP) until the drafted five-row verify, whose rows are one
sequence's draft tree, measures it with and without.

The producer-side MXFP8 fold, in stages. A step's memo holds an
activation's codes and scales under the tensor's identity (the entry
keeps it alive, so its address is not reused; cleared at every step's
start), x_mxfp8 consults it, and a producer that quantizes its output per
32-block stores the e4m3 codes and e8m0 exponents beside the bf16 values
and registers them. In: the GEMV's quant_out epilogue (wo_a's output for
wo_b, 40 launches a step) and the split front's x_in (for the shared
expert's dual GEMV, 40). The producer's scale and a recompute from the
stored bf16 values can sit a power of two apart for a block whose
quantized maximum fell under half the scale; both decode to the same
values and the consuming GEMV's result is bit for bit the pass's (GPU
tests). Three more producers followed the same pattern (row_post's
quantized rows for gemv_pe and wo_a, the dual GEMV's SwiGLU epilogue for
the down projection, the attention combine's finished heads for wo_a),
five of six, about 200 of the step's 290 launches; the norm's reduce is
an inductor kernel, not ours. VXPU_MX_MEMO=0 is the A/B back. A lesson
from its first ladder, which failed at compile three times: a runtime
cache keyed by storage has to be invisible to the lowering's traces (a
fake tensor's data_ptr raises in one torch and reads 0 in another, a
functional wrapper hides the fake one, and an entry registered under one
fake mode crosses into the next trace), the GPU unit tests do not pass
through those traces, and inductor's launcher for a user Triton kernel
miscounts the grid when a constexpr with a default is left out of a
launch; so the guard rejects trace tensors by class, meta flag and
pointer, every launch passes every constexpr, and a one-row compile check
opens every chain on a new tree. Then the ladder found a token drift: with
all five producers in, the plain one-row step no longer matched the
single-card reference (first difference at token 30, with the memo on or
off), where 8eccb7b, the stage-one tree (the memo and the GEMV producer)
and the tree with the front producer added all match with the memo off.
The three later producers' commit changes nothing outside its guarded
blocks, so the memo-off drift is a compile-level effect of those kernels'
new arguments in the segments inductor wraps; row_post, the SwiGLU rows
kernel and the attention combine are back to their 88162b8 form, and the
restored tree matches at one row with the memo on and off (8.6 ms both,
pod A) and reproduces the five-row tokens of record. The fold at this
stage (the GEMV and front producers, 80 launches a step) measures 0 within
the ladder's resolution at one row (8.6 both ways), at 33 rows (31.6 both)
and at 80 (43.4 against 43.3), tokens identical at every point. On pod B
the restored tree's 1-to-5 ladder (B42, different prompts: 8.7 / 10.0 /
11.1 / 12.1 / 13.1 ms) equals the fence tree's at every row in time and in
tokens, the one row against the reference and the 2-to-5-row first-diff
lists the same as the trees of record read on the same artifacts; memo
off reads the same at 1 and 3 rows. The three producers are dropped. The
lesson for the ledger: the un-profiled step shows no per-launch edge cost
of the size the profiles suggested, and a term sized from a launch count
is not a term until the ladder shows it; sizing from launch counts is
retired.

The grouped form with the single-pair direct body (3fb962f: inside the
launch a schedule block holding one pair runs the direct kernel's own
one-pair body, decided from the table in the kernel; blocks with two or
more take the member path) against direct, A58 on pod A, one-prompt /
same-prompt / different-prompts steps, tokens identical to direct at every
point: one row 8.6 / - / 8.6 against 8.6 / - / 8.6 (the form is off at
one row); two rows 10.3 / 9.5 / 10.3 against 10.0 / 9.7 / 9.9; five rows
12.1 / 12.0 / 13.3 against 13.3 / 13.1 / 13.1. The shared case wins 1.1
to 1.2 ms at five rows (A48's revision, with the member path on every
block, read +1.0 at one row and 1.2 at five); the no-share arms cost 0.2
to 0.4, which is the schedule pre-pass: a gather, a cast and a
one-program sort over 1,024 keys, three launches a layer on the path
between the routing and the gate/up kernel. At eight rows the form wins
every arm with the pre-pass still in place: 15.1 / 14.4 / 16.1 against
17.5 / 17.2 / 16.4.

The pre-pass folded (3f42fb8: the schedule kernel looks the slots up from
the routing indices and the slot map itself, its sort is the launch's
width, 64 keys at five rows, and the grouped gate/up kernel launches
behind it under PDL and waits for the table), A59 on pod A, the same
arms, tokens identical to direct throughout: one row 8.6 / - / 8.6 (the
compile check; the form is off at one row); two rows 9.9 / 9.2 / 9.9
against 9.9 / 9.7 / 9.9; five rows 11.8 / 11.7 / 13.0 against 13.3 /
13.1 / 13.1; eight rows 15.0 / 14.3 / 16.0 against 17.5 / 17.3 / 16.4.
Rows that share nothing now pay nothing (0 and −0.1 on different prompts
at two and five rows) and the shared case keeps its win (−0.5, −1.4 and
−3.0 on one prompt's rows at two, five and eight). The planner's table
gives rows two through eight to the grouped direct kernels, the direct
kernels to one row (`knobs_by_rows`); against the GEMM form at eight rows
on the same tree (A61: 13.0 / 12.6 / 16.8) the grouped form wins different
prompts by 0.8 and loses one prompt's rows by 1.7, a per-workload choice
the table records in its reason. At sixteen rows (96 pairs, the direct
limit raised to 128) the GEMM form wins every arm: 14.9 / 14.4 / 21.7
against the grouped form's 22.8 / 21.7 / 23.3 and the direct kernels'
27.7 / 26.9 / 24.3, so the table's boundary after eight rows stands; the
point between (12 rows) is unmeasured. The drafted step's verify rows are one
sequence's draft tree, and speculation's one-stream leg of 3fb962f (tree17,
without the schedule fold; pod D, against direct on the same tree, tokens
identical) reads a tie: prose 209.2 tok/s at 14.45 ms a step against
207.9 at 14.54, code 311.2 at 14.91 against 310.6 at 14.94; by verify
length one row 8.62 against 8.90, two 10.68 against 10.33, three 11.21
against 11.05, four 12.25 against 12.22, five 13.12 against 13.17. The
A48 revision's +1.0 ms a step on this path is gone, and the five-row
verify reads within 0.1 ms of direct where the plain step's five rows of
one prompt read −1.4: the draft tree's rows do not share experts the way
one prompt's repeated rows do. The same leg on the fold tree (tree18 = 3fb962f
plus 3f42fb8; pod A, direct control in the same chain, tokens identical)
is the verdict: five-token prose grouped 212.0 tok/s at 14.26 ms a step
against direct 207.9 at 14.54, code 315.2 at 14.72 against 310.7 at
14.93; by verify length prose one row 9.02 against 8.84, two 10.88
against 10.88, three 11.12 against 11.07, four 12.04 against 12.20, five
12.84 against 13.17; code three rows 11.16 against 11.25, five 13.07
against 13.31; the drafter call 1.66 against 1.67; the six-row code entry
348.2 tok/s at 15.70 with the six-row verify at 14.03. The fold turns the
tie into 0.24 to 0.33 ms at five rows and 0.21 to 0.28 ms a step, with
one row a shade behind (+0.18, one occurrence) as the planner's rule has
it: direct at one row, grouped from two. The grouped form stands for the
drafted step too. The win is small because the sharing is: speculation's
count says 26 to 35 percent of the rows' expert reads are shared at 3 to
5 verify rows on one sequence's draft tree, a ceiling of about 0.8 ms at
five rows, where one prompt's repeated rows share everything.

The grouped direct expert form's first build was a large loss on every
ladder (pod B, same prompt / one prompt / different prompts: 12.0 at one
row against 8.9 direct; 12.2 / 15.1 / 14.9 at two against 10.1 / 10.2 /
10.2; 15.6 / 18.3 / 23.3 at five against 13.4 / 13.6 / 13.4; 23.9 / 26.5 /
35.1 at eight against 17.2 / 17.5 / 16.4; pod A 82 / 85 / 103 at sixteen
against 27 / 28 / 24, the GEMM form 14.4 / 15.0 / 21.8), tokens identical
to the direct form's, and on the drafted step +3.0 ms at one row to +8.5
at five. The loss grew with the pairs, not with the shared experts: every
program ranked the distinct slots by a [pairs, pairs] scan, and the x tile
of 16 rows gathered inside the K loop was staged per pipeline stage. The
form is rewritten onto the GEMM form's schedule pre-pass (one small
captured launch a layer, blocks of MT = 4 pairs per slot) with the member
rows loaded one row each into registers, bit-identical again; its ladder
decides whether it stays a recorded negative.

The wire dtype is a numerics class apart from the form. The speculation
workstream's 132-prompt prose line with ALLREDUCE_PUSH_BF16: acceptance
1.436 against 1.459 a step (1.6 percent), the drafted step 14.66 against
14.96 ms, 166.0 against 164.2 tok/s; at eight streams prose -5.5 percent
and code +9 percent of acceptance; the one-row greedy tokens differ from
the reference from token 13. At 80 token rows (16 streams, the five-row verify
at batch 16, the speculation workstream's pod C) the verify went 46.06 ->
42.53 ms on prose and 48.68 -> 46.33 on code, the step 50.53 -> 47.29 and
54.04 -> 51.83, the aggregate 883.6 -> 944.1 and 1,147 -> 1,232 tok/s, with
acceptance unmoved (prose 2.797 tokens a row both ways, code 3.883 ->
4.0); so the class's gain per step grows with the rows (0.2 ms at one,
0.8 to 1.3 at 40, 2.4 to 3.5 at 80), as a wire-bytes term should, while
its acceptance cost shows only at one and eight streams. On the plain
many-row ladder (pod A, one tree, different prompts) the class is 30.9 /
32.3 / 41.4 ms against 32.0 / 33.9 / 43.6 at 33 / 40 / 80 rows (same
prompt 19.5 / 20.8 / 31.1 against 20.5 / 22.2 / 33.6), -1.1 / -1.6 / -2.2,
with the greedy tokens differing from the fp32 wire's as a numerics class
does. So the planner's rule: the knob off for prose entries at few streams, on for code and at
width; the pull form's wire is bf16 by construction and sits under the
same knob; the form by rows beside it, the push through 15 rows and
scatter from 16. Two more terms of the rule, from the speculation
workstream's drafted pairs on the fence head (pod D): with the class on
and the fused push left to its rows rule (on for the 8- and 16-row verify
programs) the tokens moved against the same tree with the fused push
forced off (4.04 against 4.255 accepted a row, 886 against 987 tok/s at 8
streams), because the plain push under the class rounds every partial to
bf16 before the fp32 sum while the fused push's epilogue keeps the fp32
wire, a third class across a step's sites; so the rule turns the fused
push off whenever the class is on, until the fused epilogue carries the
class itself. And without the class the 8- and 16-row verify programs
with the fused push on cost the drafted step 0.4 ms (686.4 against 696.0
tok/s prose, the same tokens) where the plain ladder read -0.2 and 0, so
the threshold is 6 rows: on for the one-stream verifies, off from 8. The
rule counts a program's token rows, batch times step_tokens, not the
binding's batch alone: the batch is the stream count, and speculation's
8-stream programs carry 1 to 5 token rows a stream (8 to 40 rows), so a
rule on the batch fused all of them and cost 0.4 to 0.55 ms a step on two
pods (680.7 against 693.5 tok/s prose at 8 streams, the same tokens). The
threshold is 8 token rows: the fused push wins at 8 rows in both shapes
measured, one stream's eight-token verify (16.54 against 16.81 ms a step,
medians over 126 steps, speculation) and eight streams of one row (the
plain ladder, −0.2 to 0), and loses from 16 (0) to 40 (+0.7), which is
where the eight-stream drafted step lost with the fused 40-row program.
The fused and plain pushes give identical text over 400 tokens at 8
rows (speculation's L8row pair), as the bit-for-bit check at 7, 8, 9 and
16 rows said; the numerics class that differs between row programs is
the row count's own kernel variants, not the push form. Under the
tensor-parallel experts (pod C, different prompts, one prompt / same
prompt / different prompts, the fused push off -> on): 5 rows 10.9 /
10.0 / 11.6 -> 10.6 / 9.8 / 11.4, 8 rows 12.7 / 11.6 / 14.1 -> 12.4 /
11.4 / 13.9, 16 rows 14.3 / 13.0 / 19.3 -> 14.1 / 12.9 / 19.2, tokens
identical; so the threshold is 16 token rows (the 16-row point read 0
under EP4, where the pair imbalance hid it), and the upper end is
measured too: at 33 rows under TP4 the fused push loses 1.0 ms a step
(28.9 -> 29.9 on different prompts, 17.9 -> 19.0 on one prompt's rows,
tokens identical), the pushing GEMV's epilogue stores being
bandwidth-bound there as the per-site table said. The pull form on the step
under TP4 is closed: with the fp32 wire 15.7 / 22.3 / 35.2 ms at 5 / 8 /
16 rows (its PCIe reads pathological, as the per-site table said), and
with the bf16 wire class 11.0 / 10.3 / 11.9, 12.9 / 11.9 / 14.5 and 14.3
/ 13.0 / 19.3, a tie with the fused-off step and 0.3 to 0.5 behind the
fused-on step at every width; the per-site harness's pull advantage does
not carry to the step. The bf16 class and the slicing compound only in
the roundings (a quarter-sum of every pair rounded on each card), not in
bytes.

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
