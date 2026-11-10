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


if __name__ == "__main__":
    unittest.main()
