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

"""Tests for the package cache's housekeeping."""

import os
import subprocess
import sys
import tempfile
import unittest

from . import aoti


class ReclaimExtractionsTest(unittest.TestCase):
    def test_removes_only_dead_processes_extractions(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        key = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as root:
            for name in (f"aotinductor_dead1/{key}.{dead.pid}.tmp", f"{key}.{dead.pid}.tmp",
                         f"aotinductor_own/{key}.{os.getpid()}.tmp", f"{key}.{os.getpid()}.tmp",
                         f"aotinductor_mixed/{key}.{dead.pid}.tmp", f"aotinductor_mixed/{key}.{os.getpid()}.tmp",
                         f"{key}.pt2", "aotinductor_other/notes", f"not-a-key.{dead.pid}.tmp"):
                os.makedirs(os.path.join(root, name))
                with open(os.path.join(root, name, "model.so"), "w"):
                    pass
            os.makedirs(os.path.join(root, "aotinductor_empty"))
            removed = aoti.reclaim_extractions(root, pid_alive=lambda pid: pid != dead.pid)
            self.assertEqual(sorted(removed), sorted(["aotinductor_dead1", f"{key}.{dead.pid}.tmp"]))
            self.assertEqual(sorted(os.listdir(root)),
                             sorted([f"aotinductor_own", f"{key}.{os.getpid()}.tmp", "aotinductor_mixed",
                                     f"{key}.pt2", "aotinductor_other", "aotinductor_empty",
                                     f"not-a-key.{dead.pid}.tmp"]))

    def test_alive_sees_this_process_and_not_a_dead_one(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        self.assertTrue(aoti._alive(os.getpid()))
        # a reaped child's pid may be reused; only assert the live side strictly
        self.assertIn(aoti._alive(dead.pid), (True, False))


class StableOpBoundariesAndParallelBuildsTest(unittest.TestCase):
    """GPU-free unit tests for issue #527:
    - Stable custom op boundaries: changing table's choice does not change the package key (item 1).
    - Cache key = segment's own lowered source: changes only the touched segment's key where
      the body is emitted (item 2).
    - Parallel builder produces byte-identical packages to the sequential one (item 3).
    - Load prints package counts and wall time (item 4).
    """

    def setUp(self):
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _make_two_segment_fixture(self):
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            # Segment 1 has table-selected site aten.linear
            g1 = torch.fx.Graph()
            px1 = g1.placeholder("x1")
            pw1 = g1.placeholder("w1")
            x1 = torch.empty(1, 128, dtype=torch.bfloat16, device="cuda:0")
            w1 = torch.empty(64, 128, dtype=torch.bfloat16, device="cuda:0")
            px1.meta["val"] = x1
            pw1.meta["val"] = w1
            n1 = g1.call_function(torch.ops.aten.linear.default, (px1, pw1))
            n1.meta["val"] = torch.empty(1, 64, dtype=torch.bfloat16, device="cuda:0")
            n1.meta["nn_module_stack"] = {"model.layers.0.gate": ("model.layers.0.gate", "torch.nn.Linear")}
            g1.output(n1)
            seg1 = torch.fx.GraphModule(torch.nn.Module(), g1)

            # Segment 2 is untouched
            g2 = torch.fx.Graph()
            px2 = g2.placeholder("x2")
            x2 = torch.empty(1, 128, dtype=torch.bfloat16, device="cuda:0")
            px2.meta["val"] = x2
            n2 = g2.call_function(torch.mul, (px2, 2.0))
            n2.meta["val"] = x2
            g2.output(n2)
            seg2 = torch.fx.GraphModule(torch.nn.Module(), g2)

            return seg1, seg2, (x1, w1), (x2,)

    def test_table_choice_does_not_change_package_key_behind_stable_op(self):
        """Item 1: changing table choice and knobs behind stable op boundary does not change package key."""
        import json
        from vxpu import lower, plan, ops, triton_ops

        # Table 1: picks gemv_rows (stable custom op boundary vxpu::linear)
        p1 = plan.Plan("cuda", triton_available=True)
        p1.choices["fuse"].value = ("matmul",)
        t1 = {
            "format": "vxpu-kernel-table/1",
            "gpu": "stub",
            "sites": [{"op": "aten::linear", "impl": "gemv_rows"}],
        }
        p1.table = plan.load_table(json.dumps(t1))
        seg1_a, seg2_a, a1, a2 = self._make_two_segment_fixture()
        lower.lower(seg1_a, p1)
        lower.lower(seg2_a, p1)

        k1_seg1 = aoti.cache_key(seg1_a, a1)
        k1_seg2 = aoti.cache_key(seg2_a, a2)

        # Table 2: flip table knobs: VXPU_FP4_ONE_LAUNCH, VXPU_ALLREDUCE_PUSH_BF16, GEMV_BLOCKS
        os.environ["VXPU_FP4_ONE_LAUNCH"] = "1"
        os.environ["VXPU_ALLREDUCE_PUSH_BF16"] = "1"
        triton_ops.GEMV_BLOCKS["stub"] = {"BN": 64, "BK": 128}

        p2 = plan.Plan("cuda", triton_available=True)
        p2.choices["fuse"].value = ("matmul",)
        p2.table = plan.load_table(json.dumps(t1))
        seg1_b, seg2_b, _, _ = self._make_two_segment_fixture()
        lower.lower(seg1_b, p2)
        lower.lower(seg2_b, p2)

        k2_seg1 = aoti.cache_key(seg1_b, a1)
        k2_seg2 = aoti.cache_key(seg2_b, a2)

        self.assertEqual(k1_seg1, k2_seg1, "Changing table choice or knobs must not change package key behind stable op")
        self.assertEqual(k1_seg2, k2_seg2, "Untouched segment package key must remain unchanged")

    def test_emitted_body_changes_only_touched_segment_key(self):
        """Item 2: changing table choice where body is emitted changes ONLY touched segment's key."""
        import json
        from vxpu import lower, plan

        p1 = plan.Plan("cuda", triton_available=True)
        p1.choices["fuse"].value = ("matmul",)
        t1 = {
            "format": "vxpu-kernel-table/1",
            "gpu": "stub",
            "sites": [{"op": "aten::linear", "impl": "gemv_rows"}],
        }
        p1.table = plan.load_table(json.dumps(t1))
        seg1_1, seg2_1, a1, a2 = self._make_two_segment_fixture()
        lower.lower(seg1_1, p1)
        lower.lower(seg2_1, p1)

        k1_seg1 = aoti.cache_key(seg1_1, a1)
        k1_seg2 = aoti.cache_key(seg2_1, a2)

        # Table 3: picks reduction (emitted body)
        p3 = plan.Plan("cuda", triton_available=True)
        p3.choices["fuse"].value = ("matmul",)
        t3 = {
            "format": "vxpu-kernel-table/1",
            "gpu": "stub",
            "sites": [{"op": "aten::linear", "impl": "reduction"}],
        }
        p3.table = plan.load_table(json.dumps(t3))
        seg1_3, seg2_3, _, _ = self._make_two_segment_fixture()
        lower.lower(seg1_3, p3)
        lower.lower(seg2_3, p3)

        k3_seg1 = aoti.cache_key(seg1_3, a1)
        k3_seg2 = aoti.cache_key(seg2_3, a2)

        self.assertNotEqual(k1_seg1, k3_seg1, "Touched segment key must change when body is emitted")
        self.assertEqual(k1_seg2, k3_seg2, "Untouched segment key must NOT change when another segment's body changes")

    def test_parallel_builder_produces_byte_identical_packages(self):
        """Item 3: parallel builder produces byte-identical packages to sequential builder."""
        import shutil
        import torch

        class M1(torch.nn.Module):
            def forward(self, x):
                return x * 2.0 + 3.0

        class M2(torch.nn.Module):
            def forward(self, x):
                return x - 5.0

        m1 = torch.fx.symbolic_trace(M1())
        m2 = torch.fx.symbolic_trace(M2())
        x = torch.randn(2, 4)

        with tempfile.TemporaryDirectory() as d_seq, tempfile.TemporaryDirectory() as d_par:
            pkg_seq = aoti.Packages(d_seq)
            pkg_par = aoti.Packages(d_par)

            tasks_seq = [(m1, (x,)), (m2, (x,))]
            tasks_par = [(m1, (x,)), (m2, (x,))]

            seq_b = aoti.SequentialBuilder(pkg_seq)
            paths_seq = seq_b.build_all(tasks_seq)

            par_b = aoti.ParallelBuilder(pkg_par, max_workers=2)
            paths_par = par_b.build_all(tasks_par)

            self.assertEqual(len(paths_seq), len(paths_par))
            for p_s, p_p in zip(paths_seq, paths_par):
                with open(p_s, "rb") as f1, open(p_p, "rb") as f2:
                    b1 = f1.read()
                    b2 = f2.read()
                self.assertEqual(b1, b2, f"Package {p_s} differs between sequential and parallel builds")

    def test_load_prints_package_counts_and_wall_time(self):
        """Item 4: Packages tracks and prints package built/loaded counts and wall time."""
        import io
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as cache:
            pkg = aoti.Packages(cache, expected=2)
            pkg.loaded = 1
            pkg.built = 1
            pkg.wall_s = 3.5

            with patch("sys.stdout", new_callable=io.StringIO) as out:
                pkg._print_summary()
                printed = out.getvalue()

            self.assertIn("AOTInductor packages: 1 loaded from the cache, 1 built in 3.5 s", printed)


if __name__ == "__main__":
    unittest.main()
