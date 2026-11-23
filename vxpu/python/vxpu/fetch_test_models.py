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

"""Pre-fetches test models from Hugging Face Hub once with retries.

Tests run with the Hub offline (HF_HUB_OFFLINE=1) so that transient
Hub or CDN outages do not flake checks. This setup script downloads
every model required by tests into the local cache directory.
"""

import sys
import time
from typing import Sequence

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_MODELS: tuple[str, ...] = (
    "trl-internal-testing/tiny-random-LlamaForCausalLM",
)


def fetch_model(repo_id: str, max_retries: int = 5, backoff_base: float = 2.0) -> str:
    """Fetches a model snapshot from Hugging Face Hub, retrying on transient errors."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            path = snapshot_download(repo_id, max_workers=4)
            # Validate that the tokenizer and model load cleanly from the downloaded files.
            AutoTokenizer.from_pretrained(repo_id)
            AutoModelForCausalLM.from_pretrained(repo_id)
            print(f"[vxpu] pre-fetched test model {repo_id} into {path}", flush=True)
            return path
        except Exception as err:
            last_err = err
            if attempt == max_retries:
                break
            wait = backoff_base ** attempt
            print(
                f"[vxpu] fetch {repo_id} failed (attempt {attempt}/{max_retries}): {err}; retrying in {wait:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {repo_id} after {max_retries} attempts: {last_err}") from last_err


def fetch_test_models(models: Sequence[str] = TEST_MODELS) -> None:
    """Pre-fetches all test models from Hugging Face Hub."""
    for repo_id in models:
        fetch_model(repo_id)


def main() -> None:
    fetch_test_models()


if __name__ == "__main__":
    main()
