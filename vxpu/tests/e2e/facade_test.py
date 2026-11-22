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

"""E2E test verifying vxpu.AutoModelForCausalLM facade against transformers.

This test connects to a running vxpu-router (either directly via --router or
through automatic kubectl port-forwarding), loads the model, and asserts
token-for-token equality between vxpu.AutoModelForCausalLM.generate and
transformers.AutoModelForCausalLM.generate with greedy decoding.
"""

from __future__ import annotations

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM as TransformersModel
from transformers import AutoTokenizer
from vxpu import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify vxpu facade generate matches transformers greedy output",
    )
    parser.add_argument(
        "--model",
        default="trl-internal-testing/tiny-random-LlamaForCausalLM",
        help="Model repository ID to test",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional path to existing exported artifact directory",
    )
    parser.add_argument(
        "--router",
        default=None,
        help="Router host:port (if omitted, uses kubectl port-forward to pod/vxpu-router)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id: str = args.model

    print(f"Loading reference transformers model: {model_id}...", flush=True)
    ref_model = TransformersModel.from_pretrained(model_id, dtype="auto").eval()

    print(f"Loading vxpu facade model: {model_id}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        router=args.router,
        artifact_dir=args.artifact_dir,
    )
    try:
        max_new_tokens = 10

        # 1. Verify with raw tensor input_ids
        raw_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        raw_attn = torch.ones_like(raw_ids)
        print(f"Testing raw input tensor: {raw_ids.tolist()}...", flush=True)
        vxpu_raw = model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        ref_raw = ref_model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        print(f"vxpu raw output:         {vxpu_raw[0].tolist()}", flush=True)
        print(f"transformers raw output: {ref_raw[0].tolist()}", flush=True)
        if not torch.equal(vxpu_raw, ref_raw):
            print(
                f"ERROR: Raw token mismatch: vxpu={vxpu_raw.tolist()} vs ref={ref_raw.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 2. Verify with tokenized prompt
        tok = AutoTokenizer.from_pretrained(model_id)
        prompt_text = "Is the sky blue?"
        inputs = tok(prompt_text, return_tensors="pt")
        prompt_len = inputs.input_ids.shape[1]
        print(f"Testing tokenized prompt {prompt_text!r} ({prompt_len} tokens)...", flush=True)
        vxpu_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        ref_out = ref_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        print(f"vxpu tokenized output:         {vxpu_out[0].tolist()}", flush=True)
        print(f"transformers tokenized output: {ref_out[0].tolist()}", flush=True)
        if not torch.equal(vxpu_out, ref_out):
            print(
                f"ERROR: Tokenized mismatch: vxpu={vxpu_out.tolist()} vs ref={ref_out.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        model.close()

    print("Facade and transformers greedy outputs are token-for-token identical!", flush=True)


if __name__ == "__main__":
    main()
