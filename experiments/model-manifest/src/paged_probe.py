"""Probe: does PagedAttention survive graph capture?

torch ships an experimental paged KV abstraction built on FlexAttention
(a page table rewrites a logical BlockMask into physical page space).
Three staged questions, each may fail independently:

  1. eager: paged flex attention == dense SDPA?
  2. torch.compile: the intended consumption path
  3. torch.export: the interchange path — can a paged decode step become
     a portable artifact, with pool + page table as `state` tensors?
"""

import sys
import traceback

import torch
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.manual_seed(0)
H, D = 2, 16
PAGE, NPAGES = 16, 8
MAX_S = PAGE * NPAGES


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def section(title):
    print(f"\n=== {title} ===")


# --- setup: one sequence of 40 tokens scattered across pages ---------
paged = PagedAttention(NPAGES, PAGE, max_batch_size=1, device=DEVICE)
k_cache = torch.zeros(1, H, MAX_S, D, device=DEVICE)
v_cache = torch.zeros(1, H, MAX_S, D, device=DEVICE)

S = 40
q = torch.randn(1, H, S, D, device=DEVICE)
k = torch.randn(1, H, S, D, device=DEVICE)
v = torch.randn(1, H, S, D, device=DEVICE)
batch = torch.tensor([0], device=DEVICE)

paged.reserve(batch, torch.tensor([S], device=DEVICE))
paged.assign(batch, torch.arange(S, device=DEVICE).unsqueeze(0), k, v, k_cache, v_cache)
print(f"page table (logical->physical): "
      f"{paged.page_table[0][: (S + PAGE - 1) // PAGE].tolist()}")

logical = create_block_mask(causal, 1, 1, S, S, device=DEVICE, BLOCK_SIZE=PAGE)
physical = paged.convert_logical_block_mask(logical, batch_idx=batch)

# --- 1. eager correctness -------------------------------------------
section("1. eager: paged flex vs dense SDPA")
try:
    out = flex_attention(q, k_cache, v_cache, block_mask=physical)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"max |diff| = {(out - ref).abs().max().item():.2e}")
except Exception:
    print(traceback.format_exc().strip().splitlines()[-1])

# --- 2. torch.compile (intended path) -------------------------------
section("2. torch.compile of paged flex step")
try:
    compiled = torch.compile(flex_attention)
    out_c = compiled(q, k_cache, v_cache, block_mask=physical)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"compiles; max |diff| vs SDPA = {(out_c - ref).abs().max().item():.2e}")
except Exception:
    print(traceback.format_exc().strip().splitlines()[-1])


# --- 3. torch.export (interchange path) -----------------------------
class PagedDecode(torch.nn.Module):
    """One paged attention step. The pool and the block-mask component
    tensors are buffers — i.e. the binding's `state` category. The
    BlockMask object is rebuilt from those tensors inside forward."""

    def __init__(self, k_cache, v_cache, physical_mask):
        super().__init__()
        self.register_buffer("k_cache", k_cache)
        self.register_buffer("v_cache", v_cache)
        self.register_buffer("kv_num_blocks", physical_mask.kv_num_blocks)
        self.register_buffer("kv_indices", physical_mask.kv_indices)
        self.register_buffer(
            "full_kv_num_blocks", physical_mask.full_kv_num_blocks
        )
        self.register_buffer("full_kv_indices", physical_mask.full_kv_indices)
        self.mask_mod = physical_mask.mask_mod

    def forward(self, q):
        mask = BlockMask.from_kv_blocks(
            self.kv_num_blocks,
            self.kv_indices,
            self.full_kv_num_blocks,
            self.full_kv_indices,
            BLOCK_SIZE=(PAGE, PAGE),
            mask_mod=self.mask_mod,
            seq_lengths=(S, MAX_S),
        )
        return flex_attention(q, self.k_cache, self.v_cache, block_mask=mask)


section("3. torch.export of paged decode step")
try:
    module = PagedDecode(k_cache, v_cache, physical)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"module runs eagerly: "
          f"max |diff| = {(module(q) - ref).abs().max().item():.2e}")
    ep = torch.export.export(module, (q,))
    out_e = ep.module()(q)
    print(f"EXPORTS: {len(list(ep.graph.nodes))} nodes; "
          f"max |diff| = {(out_e - ref).abs().max().item():.2e}")
    torch.export.save(ep, "paged_decode.pt2")
    import os

    print(f"saved paged_decode.pt2 ({os.path.getsize('paged_decode.pt2') / 1e3:.0f} KB)")
except Exception:
    print(traceback.format_exc().strip().splitlines()[-1])
