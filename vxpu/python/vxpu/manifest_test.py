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

"""Unit tests for offline manifest building and Hub cache resolution."""

import os
import unittest
from unittest import mock

from vxpu import fetch_test_models, manifest, rehydrate


class ManifestOfflineTest(unittest.TestCase):
    """Verifies that manifests build offline from cache and fail loudly when uncached."""

    def test_is_hub_url(self) -> None:
        """Verifies distinction between Hub endpoints and local test loopback servers."""
        self.assertTrue(manifest._is_hub_url("https://huggingface.co/org/model/resolve/main/m.safetensors"))
        self.assertTrue(manifest._is_hub_url("https://us.aws.cdn.hf.co/org/model/resolve/main/m.safetensors"))
        self.assertFalse(manifest._is_hub_url("http://127.0.0.1:8080/chunk"))
        self.assertFalse(manifest._is_hub_url("http://localhost:9000/chunk"))

    def test_resolve_local_hub_file(self) -> None:
        """Verifies resolution of Hugging Face resolve URLs to cached local files."""
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
        resolved = manifest._resolve_local_hub_file(url)
        self.assertIsNotNone(resolved)
        if resolved is not None:
            self.assertTrue(os.path.exists(resolved))
            self.assertTrue(resolved.endswith("config.json"))

    def test_find_cached_snapshot(self) -> None:
        """Verifies cached snapshot directory discovery for pre-fetched test models."""
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        snapshot = manifest._find_cached_snapshot(model_id)
        self.assertIsNotNone(snapshot)
        if snapshot is not None:
            snapshot_dir, commit_sha = snapshot
            self.assertTrue(os.path.isdir(snapshot_dir))
            self.assertGreater(len(commit_sha), 0)

    def test_build_manifest_offline(self) -> None:
        """Verifies that build_manifest succeeds from local cache when offline mode is active."""
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_ENDPOINT": "http://127.0.0.1:9"}):
            doc = manifest.build_manifest(model_id)
            self.assertEqual(doc["format"], manifest.MANIFEST_FORMAT)
            self.assertEqual(doc["source"]["repo"], model_id)
            self.assertGreater(len(doc["files"]), 0)
            self.assertGreater(len(doc["tensors"]), 0)

    def test_build_manifest_uncached_fails_offline(self) -> None:
        """Verifies that an uncached model fails loudly when offline mode is active."""
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                manifest.build_manifest("test-org/uncached-model")
            self.assertIn("HF_HUB_OFFLINE is enabled but model test-org/uncached-model is not in local cache", str(ctx.exception))

    def test_fetch_range_uncached_fails_offline(self) -> None:
        """Verifies that uncached remote URLs fail loudly when offline mode is active."""
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                manifest._fetch_range("https://huggingface.co/uncached/resolve/main/blob", 0, 8)
            self.assertIn("HF_HUB_OFFLINE is enabled", str(ctx.exception))

    def test_rehydrate_read_range_uncached_fails_offline(self) -> None:
        """Verifies that rehydrate._read_range fails loudly on uncached URLs in offline mode."""
        buf = bytearray(8)
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaises(RuntimeError) as ctx:
                rehydrate._read_range("https://huggingface.co/uncached/resolve/main/blob", 0, 8, memoryview(buf))
            self.assertIn("HF_HUB_OFFLINE is enabled", str(ctx.exception))

    def test_fetch_model_retries_transient_failure(self) -> None:
        """Verifies retry mechanism in fetch_test_models upon encountering transient errors."""
        attempts = 0

        def flaky_download(*args: object, **kwargs: object) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionResetError("Connection reset by peer")
            return "/tmp/fake-snapshot"

        with mock.patch("vxpu.fetch_test_models.snapshot_download", side_effect=flaky_download):
            with mock.patch("vxpu.fetch_test_models.AutoTokenizer.from_pretrained"):
                with mock.patch("vxpu.fetch_test_models.AutoModelForCausalLM.from_pretrained"):
                    path = fetch_test_models.fetch_model("test-org/test-model", max_retries=5, backoff_base=0.01)
                    self.assertEqual(path, "/tmp/fake-snapshot")
                    self.assertEqual(attempts, 3)


if __name__ == "__main__":
    unittest.main()
