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

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from . import diagnostics
from .localblobs import external_files_dir, tensor_key


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.m = {
            "source": {"repo": "test/model"},
            "files": {
                "s1": {"size": 4, "name": "f1"},
                "s2": {"size": 4, "name": "f2"},
            },
            "tensors": {
                "w": {"file_sha256": "s1", "offset": 0, "length": 4},
                "s": {"file_sha256": "s1", "offset": 4, "length": 4},
            },
        }
        self.b = {
            "external": {
                "experts": {"0": {"w1": {}}},
                "engram": {"0": {"weight": "w", "scale": "s"}},
            }
        }
        self.max_map_path = os.path.join(self.root, "max_map_count")
        self._write(self.max_map_path, b"1048576\n")
        self.core_pattern_path = os.path.join(self.root, "core_pattern")
        self._write(self.core_pattern_path, b"/dev/null\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, path, data=b"x" * 4096):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def test_blob_dir(self):
        status, msg = diagnostics.check_blob_dir(self.m, {})
        self.assertEqual(status, "FAIL")
        self.assertIn("f1", msg)

        bdir = os.path.join(self.root, "blobs")
        self._write(os.path.join(bdir, "s1"), b"1234")
        status, msg = diagnostics.check_blob_dir(self.m, {"VXPU_BLOB_DIR": bdir})
        self.assertEqual(status, "FAIL")

        self._write(os.path.join(bdir, "s2"), b"5678")
        status, msg = diagnostics.check_blob_dir(self.m, {"VXPU_BLOB_DIR": bdir})
        self.assertEqual(status, "PASS")

    def test_external_dir(self):
        status, _ = diagnostics.check_external_dir(self.m, self.b, {})
        self.assertEqual(status, "FAIL")

        edir = os.path.join(self.root, "ext")
        status, msg = diagnostics.check_external_dir(self.m, self.b, {"VXPU_EXTERNAL_DIR": edir})
        self.assertEqual(status, "FAIL")
        self.assertIn("missing", msg)

        mm = external_files_dir(edir, self.m)
        for p in diagnostics._external_paths(self.m, self.b, env={}):
            self._write(os.path.join(mm, p))
        status, _ = diagnostics.check_external_dir(self.m, self.b, {"VXPU_EXTERNAL_DIR": edir})
        self.assertEqual(status, "PASS")

    def test_external_dir_experts_in_ram(self):
        # With VXPU_EXPERTS_IN_RAM=1, expert bank files are not expected.
        b_experts_only = {"external": {"experts": {"0": {"w1": {}}}}}
        status, msg = diagnostics.check_external_dir(
            self.m, b_experts_only, {"VXPU_EXPERTS_IN_RAM": "1"}
        )
        self.assertEqual(status, "PASS")
        self.assertIn("experts held in RAM", msg)

        # With engram tables and experts in RAM, only engram tables are expected.
        edir = os.path.join(self.root, "ext")
        mm = external_files_dir(edir, self.m)
        for p in diagnostics._external_paths(self.m, self.b, env={"VXPU_EXPERTS_IN_RAM": "1"}):
            self._write(os.path.join(mm, p))
        status, msg = diagnostics.check_external_dir(
            self.m, self.b, {"VXPU_EXTERNAL_DIR": edir, "VXPU_EXPERTS_IN_RAM": "1"}
        )
        self.assertEqual(status, "PASS")
        self.assertIn("experts held in RAM", msg)

    def test_external_dir_layer_range(self):
        # With layer range 1:2, layer 0 is not expected on disk for either banks or tables.
        m = dict(self.m)
        m["tensors"] = {
            "w0": {"file_sha256": "s1", "offset": 0, "length": 4},
            "s0": {"file_sha256": "s1", "offset": 4, "length": 4},
            "w1": {"file_sha256": "s2", "offset": 0, "length": 4},
            "s1": {"file_sha256": "s2", "offset": 4, "length": 4},
        }
        b_multi_layer = {
            "external": {
                "experts": {"0": {"w1": {}}, "1": {"w1": {}}},
                "engram": {
                    "0": {"weight": "w0", "scale": "s0"},
                    "1": {"weight": "w1", "scale": "s1"},
                },
            }
        }
        edir = os.path.join(self.root, "ext")
        mm = external_files_dir(edir, m)
        self._write(os.path.join(mm, "experts.1.w1"))
        self._write(os.path.join(mm, tensor_key(m["tensors"]["w1"])))
        self._write(os.path.join(mm, tensor_key(m["tensors"]["s1"])))
        status, _ = diagnostics.check_external_dir(
            m, b_multi_layer, {"VXPU_EXTERNAL_DIR": edir, "VXPU_LAYER_RANGE": "1:2"}
        )
        self.assertEqual(status, "PASS")

    def test_tokenizer(self):
        self.assertEqual(diagnostics.check_tokenizer(self.m, {})[0], "FAIL")
        hdir = os.path.join(self.root, "hf")
        self.assertEqual(diagnostics.check_tokenizer(self.m, {"HF_HOME": hdir})[0], "FAIL")
        self._write(os.path.join(hdir, "models--test--model", "snapshots", "rev", "tokenizer.json"), b"{}")
        self.assertEqual(diagnostics.check_tokenizer(self.m, {"HF_HOME": hdir})[0], "PASS")

    def test_environment_and_caches(self):
        env = {
            "HF_HUB_OFFLINE": "0",
            "VXPU_SCRATCH_DIR": self.root,
        }
        self.assertEqual(
            diagnostics.check_environment(
                env,
                max_map_count_path=self.max_map_path,
                core_pattern_path=self.core_pattern_path,
            )[0],
            "FAIL",
        )

        cache_keys = ("TMPDIR", "VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        # HF_HUB_OFFLINE accepts 1, TRUE, on, yes case-insensitively
        for truthy in ("1", "TRUE", "on", "yes"):
            env["HF_HUB_OFFLINE"] = truthy
            env.update({k: "/tmp" for k in cache_keys})
            # Sibling prefix like /mnt/scratch2 should not match /mnt/scratch
            self.assertEqual(
                diagnostics.check_environment(
                    env,
                    max_map_count_path=self.max_map_path,
                    core_pattern_path=self.core_pattern_path,
                )[0],
                "FAIL",
            )

        # Common path under scratch directory
        env.update({k: os.path.join(self.root, k) for k in cache_keys})
        self.assertEqual(
            diagnostics.check_environment(
                env,
                max_map_count_path=self.max_map_path,
                core_pattern_path=self.core_pattern_path,
            )[0],
            "PASS",
        )

        # Caches
        self.assertEqual(diagnostics.check_caches(env)[0], "FAIL")
        for k in ("VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            os.makedirs(env[k], exist_ok=True)
        self.assertEqual(diagnostics.check_caches(env)[0], "PASS")

    def test_max_map_count_and_core_dumps(self):
        cache_vars = ("TMPDIR", "VXPU_AOTI_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        base_env = {
            "HF_HUB_OFFLINE": "1",
            "VXPU_SCRATCH_DIR": self.root,
            **{k: os.path.join(self.root, k) for k in cache_vars},
        }

        # vm.max_map_count too low
        bad_map_path = os.path.join(self.root, "bad_map")
        self._write(bad_map_path, b"65530\n")
        status, msg = diagnostics.check_environment(
            base_env,
            max_map_count_path=bad_map_path,
            core_pattern_path=self.core_pattern_path,
        )
        self.assertEqual(status, "FAIL")
        self.assertIn("vm.max_map_count is 65530", msg)

        # core dumps enabled: soft limit > 0 and core_pattern not /dev/null
        bad_core_path = os.path.join(self.root, "bad_core")
        self._write(bad_core_path, b"/core.%e\n")
        with patch("resource.getrlimit", return_value=(1024, 2048)):
            status, msg = diagnostics.check_environment(
                base_env,
                max_map_count_path=self.max_map_path,
                core_pattern_path=bad_core_path,
            )
            self.assertEqual(status, "FAIL")
            self.assertIn("core dumps enabled", msg)

        # core dumps disabled via RLIMIT_CORE soft limit 0
        with patch("resource.getrlimit", return_value=(0, 2048)):
            status, _ = diagnostics.check_environment(
                base_env,
                max_map_count_path=self.max_map_path,
                core_pattern_path=bad_core_path,
            )
            self.assertEqual(status, "PASS")

        # core dumps disabled via core_pattern /dev/null
        with patch("resource.getrlimit", return_value=(1024, 2048)):
            status, _ = diagnostics.check_environment(
                base_env,
                max_map_count_path=self.max_map_path,
                core_pattern_path=self.core_pattern_path,
            )
            self.assertEqual(status, "PASS")

    def test_scratch_and_page_cache(self):
        sdir = os.path.join(self.root, "scratch")
        edir = os.path.join(self.root, "ext")
        bdir = os.path.join(self.root, "blobs")
        adir = os.path.join(self.root, "aoti")
        # Temporary files from 4 operations: compilation, extraction, blob copy, AOTI
        self._write(os.path.join(sdir, "vxpu-compile-test", "f"), b"x" * 100)
        self._write(os.path.join(edir, "keyed", "experts.0.w1.partial.123"), b"x" * 200)
        self._write(os.path.join(bdir, ".incoming", "123-x", "f"), b"x" * 300)
        self._write(os.path.join(adir, "package.123.tmp.pt2"), b"x" * 400)

        env = {
            "VXPU_SCRATCH_DIR": sdir,
            "TMPDIR": sdir,
            "VXPU_EXTERNAL_DIR": edir,
            "VXPU_BLOB_DIR": bdir,
            "VXPU_AOTI_CACHE_DIR": adir,
        }
        self.assertEqual(diagnostics._scan_strays(env), 1000)

        status, msg = diagnostics.check_scratch(env)
        self.assertEqual(status, "PASS")
        self.assertIn("0.0 GB temporaries", msg)

        # Write under keyed external directory and assert page cache PASS with percentage
        mm = external_files_dir(edir, self.m)
        self._write(os.path.join(mm, "experts.0.w1"), b"x" * 8192)
        b_experts = {"external": {"experts": {"0": {"w1": {}}}}}
        status, msg = diagnostics.check_page_cache(self.m, b_experts, {"VXPU_EXTERNAL_DIR": edir})
        self.assertEqual(status, "PASS")
        self.assertIn("100.0% resident", msg)

    def test_gpus_and_run_diagnostics(self):
        self._write(os.path.join(self.root, "locks", "LOCK-test"), b"1")

        # Torch import failure reports FAIL
        with patch.dict(sys.modules, {"torch": None}):
            status, msg = diagnostics.check_gpus({"VXPU_LOCK_PATTERN": "/nonexistent/*"})
            self.assertEqual(status, "FAIL")
            self.assertIn("GPU check failed", msg)

        # Mock torch cleanly without poisoning later tests
        fake = MagicMock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 2
        fake.cuda.mem_get_info.return_value = (8e10, 9e10)
        with patch.dict(sys.modules, {"torch": fake}):
            status, msg = diagnostics.check_gpus({"VXPU_LOCK_PATTERN": os.path.join(self.root, "locks", "LOCK-*")})
            self.assertEqual(status, "WARN")
            self.assertIn("held by", msg)

            status, msg = diagnostics.check_gpus({"VXPU_LOCK_PATTERN": "/nonexistent/*"})
            self.assertEqual(status, "PASS")
            self.assertIn("2 GPUs", msg)

        art = os.path.join(self.root, "art")
        self._write(os.path.join(art, "manifest.json"), json.dumps(self.m).encode())
        self.assertEqual(diagnostics.run_diagnostics(art, {}, log=lambda _: None), 1)

        # Test artifact discovery from VXPU_ARTIFACT_DIR when artifact_dir is None
        self.assertEqual(diagnostics.run_diagnostics(None, {"VXPU_ARTIFACT_DIR": art}, log=lambda _: None), 1)

        # Malformed VXPU_LAYER_RANGE is reported as FAIL (exit 1)
        self.assertEqual(diagnostics.run_diagnostics(art, {"VXPU_LAYER_RANGE": "abc"}, log=lambda _: None), 1)


if __name__ == "__main__":
    unittest.main()
