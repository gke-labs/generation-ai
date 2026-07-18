# Plan: paged-attention rewrite pass (saved for later)

Status: planned, not started. All prerequisite probes done and committed.

## Goal

Take the shipped, portable static-cache `decode.pt2` (reference
semantics) and, **executor-side at load time**, rewrite it into paged
attention over a shared page pool — then compile and serve. The
interchange artifact never changes; paging becomes an engine access-path
decision, like a database choosing storage layout for a logical plan.

## Evidence already gathered (all committed)

- `src/paged_probe.py`: paged flex attention == dense SDPA to 2.4e-07
  with pages physically out of order; torch.compile works on CUDA when
  page_size meets Triton block-size constraints (128 yes, 16 no; no
  flex lowering on macOS CPU at all); torch.export captures a correct
  paged graph in memory but `torch.export.save` fails on the flex HOP
  (`SerializeError`) — so the paged form cannot ship, and must be
  produced after load. Transform-on-arrival is therefore forced, and
  sufficient.
- `decode.pt2` recognizability (probed in-session, 2026-07-17):
  - 30 × `aten.scaled_dot_product_attention.default` — one single-op
    attention node per layer, `nn_module_stack` intact
    (`model.model.layers.N.self_attn`).
  - 60 × `aten.index_copy_` — the per-layer K and V cache writes,
    indexed by `cache_position`.
  - 30 × `aten.copy_` — cache counters.
  - `graph_signature.buffers_to_mutate` is EMPTY — state tensors are
    non-persistent buffers living in `ep.constants`; mutations are
    explicit in-place nodes in the graph. Find the cache tensors via
    the binding's `state` category, not the signature.
- Serving engine baseline (`src/engine.py`): per-session StaticCache =
  23.6 MB preallocated; demo sessions used ~18% of it. This is what
  the pool eliminates.

## Design

1. `src/paged_rewrite.py` — a graph pass over a loaded ExportedProgram:
   - Anchors: per layer, locate (index_copy_ K, index_copy_ V, sdpa)
     via nn_module_stack + binding `state` FQNs.
   - Substitute cache writes: static buffer -> shared pool tensor, with
     page-table address translation inserted before each index_copy_
     (`phys = table[pos // PAGE] * PAGE + pos % PAGE`).
   - Substitute reads: sdpa(q, k_buf, v_buf, mask) ->
     flex_attention(q, k_pool, v_pool, block_mask) using
     `torch.nn.attention.experimental._paged_attention.PagedAttention`
     mask conversion. PAGE=128 (Triton constraint, measured).
   - ~90 node edits out of 2,472. Same mechanism as
     `generator.strip_asserts` (edit graph, `recompile()`).
2. Engine integration (`src/engine.py`):
   - Engine owns: pool tensors per layer, one `PagedAttention`
     instance, free list. `new_session` = claim a page-table row
     (zero bytes of KV) instead of allocating 23.6 MB.
   - Binding gains state scope: engine-state (pool) vs session-state
     (page-table row, lengths). Document in README format notes.
3. Verification harness: run static and paged paths side by side,
   require logits parity (same tolerance discipline as
   `executor.py --verify`). Every rewritten model self-tests.

## Build order, with acceptance criteria

1. Re-export decode with dynamic batch Dim (one line in
   `export_cached.py`) — prerequisite for later continuous batching;
   harmless for single-session. Accept: export succeeds, generator
   still works.
2. Rewrite pass, single layer first, then all 30. Accept: paged
   decode logits == static decode logits (CPU eager; then L4 compiled
   with bf16 tolerance + argmax match).
3. Engine on pool: N sessions share one pool. Accept: 20 concurrent
   sessions, memory ~= pages-used (not 20 x 23.6 MB); session
   isolation demo still passes.
4. (Stretch) batched decode tick: one flex call steps all active
   sessions. Accept: tokens/sec scales with session count on L4.

## Hazards (ranked)

1. **Mask semantics**: substituting paged-causal is only valid if the
   source mask was causal-over-positions. Check manifest config
   (`sliding_window` etc. — Gemma differs!). Wrong guess = silent
   garbage; the parity harness is the guard.
2. **Batch dim**: do NOT shape-rewrite batch=1 -> N in the pass;
   re-export with dynamic Dim instead (step 1).
3. **In-place ordering**: 60 retargeted index_copy_ must preserve
   mutation order; compile-after-rewrite revalidates.
4. Flex lowering absent on macOS CPU: eager paged works locally, but
   compiled-paged testing needs the L4 pod (`dev/tasks/run-in-kube`).

## Open questions

- Where does the logical->physical block-mask conversion run per step:
  precomputed per session per tick (cheap tensors) vs inside the graph?
  Probe used precomputed; start there.
- Prefix sharing (refcounted pages, COW) — defer until pool works.
- When this stops being worth it vs adopting vLLM/HF continuous
  batching wholesale: revisit after step 3 numbers.
