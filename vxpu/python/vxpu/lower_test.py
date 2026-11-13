# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for graph lowering and kernel digest computation."""

import os
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from . import lower, triton_ops


class LowerTest(unittest.TestCase):

    def setUp(self):
        self.old_gemv = getattr(triton_ops, "GEMV_ROWS_MAX", None)
        self.old_fp4_direct_max = getattr(triton_ops, "FP4_DIRECT_MAX", None)
        self.old_allreduce_block = getattr(triton_ops, "ALLREDUCE_BLOCK", None)
        self.old_base_cache_dir = lower._base_cache_dir
        self.old_memo_digest = lower._memo_digest
        self.old_keyed_cache_dir = lower._keyed_cache_dir
        self.old_cache_env = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        lower._base_cache_dir = None
        lower._memo_digest = None
        lower._keyed_cache_dir = None

    def tearDown(self):
        lower._base_cache_dir = self.old_base_cache_dir
        lower._memo_digest = self.old_memo_digest
        lower._keyed_cache_dir = self.old_keyed_cache_dir
        if self.old_cache_env is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = self.old_cache_env

        if self.old_gemv is None:
            if hasattr(triton_ops, "GEMV_ROWS_MAX"):
                delattr(triton_ops, "GEMV_ROWS_MAX")
        else:
            triton_ops.GEMV_ROWS_MAX = self.old_gemv

        if self.old_fp4_direct_max is None:
            if hasattr(triton_ops, "FP4_DIRECT_MAX"):
                delattr(triton_ops, "FP4_DIRECT_MAX")
        else:
            triton_ops.FP4_DIRECT_MAX = self.old_fp4_direct_max

        if self.old_allreduce_block is None:
            if hasattr(triton_ops, "ALLREDUCE_BLOCK"):
                delattr(triton_ops, "ALLREDUCE_BLOCK")
        else:
            triton_ops.ALLREDUCE_BLOCK = self.old_allreduce_block

    def test_kernel_digest_and_key_cache_dir(self):
        triton_ops.GEMV_ROWS_MAX = 8
        d1 = lower.kernel_digest()
        self.assertEqual(len(d1), 16)
        cache_dir_1 = lower.key_cache_dir()
        self.assertTrue(cache_dir_1.endswith(f"vxpu-kernels-{d1}"))
        self.assertEqual(os.environ.get("TORCHINDUCTOR_CACHE_DIR"), cache_dir_1)

        # Recomputing with a new knob updates the cache directory and environment variable.
        triton_ops.GEMV_ROWS_MAX = 16
        d2 = lower.kernel_digest()
        self.assertEqual(len(d2), 16)
        self.assertNotEqual(d1, d2)

        cache_dir_2 = lower.key_cache_dir()
        self.assertTrue(cache_dir_2.endswith(f"vxpu-kernels-{d2}"))
        self.assertNotEqual(cache_dir_1, cache_dir_2)
        self.assertEqual(os.environ.get("TORCHINDUCTOR_CACHE_DIR"), cache_dir_2)
        # Verify base directory was not nested.
        self.assertNotIn("vxpu-kernels-", os.path.dirname(cache_dir_2))

        # Idempotent when digest does not change.
        self.assertEqual(lower.key_cache_dir(), cache_dir_2)

        # kernel_digest does not change when FP4_DIRECT_MAX changes.
        triton_ops.FP4_DIRECT_MAX = 10
        d3 = lower.kernel_digest()
        self.assertEqual(d3, d2)
        self.assertEqual(lower.key_cache_dir(), cache_dir_2)

        # kernel_digest does not change when ALLREDUCE_BLOCK changes.
        triton_ops.ALLREDUCE_BLOCK = 1024
        d4 = lower.kernel_digest()
        self.assertEqual(d4, d2)
        self.assertEqual(lower.key_cache_dir(), cache_dir_2)

        # An inherited keyed directory in TORCHINDUCTOR_CACHE_DIR strips to its parent.
        lower._base_cache_dir = None
        lower._memo_digest = None
        lower._keyed_cache_dir = None
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/ti/vxpu-kernels-0123456789abcdef"
        cache_dir_inherited = lower.key_cache_dir()
        self.assertEqual(os.path.dirname(cache_dir_inherited), "/tmp/ti")
        self.assertTrue(cache_dir_inherited.startswith("/tmp/ti/vxpu-kernels-"))

    def test_site_bytes(self):
        with FakeTensorMode():
            g = torch.fx.Graph()
            x = g.placeholder("x")
            x.meta["val"] = torch.empty((1, 16), dtype=torch.bfloat16)

            p_model_w = g.placeholder("p_model_w")
            p_model_w.meta["val"] = torch.empty((16, 8), dtype=torch.float8_e4m3fn)

            b_model_cache = g.placeholder("b_model_cache")
            b_model_cache.meta["val"] = torch.empty((64, 8), dtype=torch.bfloat16)

            perm = g.call_function(torch.ops.aten.permute.default, (p_model_w, [1, 0]))
            perm.meta["val"] = torch.empty((8, 16), dtype=torch.float8_e4m3fn)

            mm = g.call_function(torch.ops.aten.mm.default, (x, perm))
            mm.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            add = g.call_function(torch.ops.aten.add.Tensor, (b_model_cache, mm))
            add.meta["val"] = torch.empty((64, 8), dtype=torch.bfloat16)

            buf = g.get_attr("model_buf")

            buf_op = g.call_function(torch.ops.aten.add.Tensor, (mm, buf))
            buf_op.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            class Mod(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model_buf = torch.empty((1, 8), dtype=torch.bfloat16)

            gm = torch.fx.GraphModule(Mod(), g)

            # mm site reads: weights: 128 (16x8 fp8), activations: 48 (1x16 bf16 input [32] + 1x8 bf16 output [16]), state: 0
            sb_mm = lower.site_bytes(gm, mm)
            self.assertEqual(sb_mm.weights, 128)
            self.assertEqual(sb_mm.activations, 48)
            self.assertEqual(sb_mm.state, 0)

            # add site reads: weights: 0, activations: 1040 (1x8 bf16 input [16] + 64x8 bf16 output [1024]), state: 1024 (64x8 bf16)
            sb_add = lower.site_bytes(gm, add)
            self.assertEqual(sb_add.weights, 0)
            self.assertEqual(sb_add.activations, 1040)
            self.assertEqual(sb_add.state, 1024)

            # a get_attr buffer counts as a parameter (weights: 16 bytes for 1x8 bf16)
            sb_buf = lower.site_bytes(gm, buf_op)
            self.assertEqual(sb_buf.weights, 16)
            self.assertEqual(sb_buf.activations, 32)
            self.assertEqual(sb_buf.state, 0)

            # an intermediate call_function result counts as activation, never weights or state
            cond = g.placeholder("cond")
            cond.meta["val"] = torch.empty((1, 8), dtype=torch.bool)
            where_node = g.call_function(torch.ops.aten.where.self, (cond, mm, buf_op))
            where_node.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            consumer = g.call_function(torch.ops.aten.add.Tensor, (where_node, mm))
            consumer.meta["val"] = torch.empty((1, 8), dtype=torch.bfloat16)

            sb_consumer = lower.site_bytes(gm, consumer)
            self.assertEqual(sb_consumer.weights, 0)
            self.assertEqual(sb_consumer.activations, 48)
            self.assertEqual(sb_consumer.state, 0)


if __name__ == "__main__":
    unittest.main()
