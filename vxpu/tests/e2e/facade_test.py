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
through automatic kubectl port-forwarding), loads the model, and asserts:
1. Token-for-token equality between facade generate and transformers generate.
2. Suffix-only prefill on a grown conversation in a second generate.
3. Seeded sampling repeatability and stop token handling.
4. Prompt client-side cancellation and executor session freeing.
"""

from __future__ import annotations

import argparse
import sys
import time
import torch
from transformers import AutoModelForCausalLM as TransformersModel
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from vxpu import AutoModelForCausalLM


class CancelOnTokenStreamer(BaseStreamer):
    """Raises GeneratorExit after receiving target number of tokens to cancel generate."""

    def __init__(self, stop_after_tokens: int = 2) -> None:
        self.count = 0
        self.stop_after_tokens = stop_after_tokens

    def put(self, value: torch.Tensor) -> None:
        if value.numel() == 1:
            self.count += 1
            if self.count >= self.stop_after_tokens:
                raise GeneratorExit("client cancelled generation")

    def end(self) -> None:
        pass


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

        # 3. Verify second generate on grown conversation prefills only new suffix
        suffix_text = " Why?"
        suffix_ids = tok(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids
        suffix_len = suffix_ids.shape[1]
        grown_ids = torch.cat([vxpu_out, suffix_ids], dim=1)
        grown_attn = torch.ones_like(grown_ids)
        print(
            f"Testing grown conversation ({grown_ids.shape[1]} tokens, suffix {suffix_len} tokens)...",
            flush=True,
        )
        vxpu_grown = model.generate(
            grown_ids,
            attention_mask=grown_attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        ref_grown = ref_model.generate(
            grown_ids,
            attention_mask=grown_attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        print(f"vxpu grown output:         {vxpu_grown[0].tolist()}", flush=True)
        print(f"transformers grown output: {ref_grown[0].tolist()}", flush=True)
        if not torch.equal(vxpu_grown, ref_grown):
            print(
                f"ERROR: Grown conversation mismatch: vxpu={vxpu_grown.tolist()} vs ref={ref_grown.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)
        stats = model.last
        print(
            f"Grown prefill stats: prompt_tokens={stats.prompt_tokens}, prefilled_tokens={stats.prefilled_tokens}",
            flush=True,
        )
        if stats.prompt_tokens != grown_ids.shape[1]:
            print(
                f"ERROR: Expected prompt_tokens {grown_ids.shape[1]}, got {stats.prompt_tokens}",
                file=sys.stderr,
            )
            sys.exit(1)
        if stats.prefilled_tokens != suffix_len:
            print(
                f"ERROR: Expected prefilled_tokens {suffix_len}, got {stats.prefilled_tokens}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 4. Verify seeded sampling is repeatable
        print("Testing seeded sampling repeatability...", flush=True)
        model.new_session()
        sample1 = model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            seed=42,
        )
        model.new_session()
        sample2 = model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            seed=42,
        )
        print(f"vxpu sample 1 (seed=42): {sample1[0].tolist()}", flush=True)
        print(f"vxpu sample 2 (seed=42): {sample2[0].tolist()}", flush=True)
        if not torch.equal(sample1, sample2):
            print(
                f"ERROR: Seeded sampling mismatch: sample1={sample1.tolist()} vs sample2={sample2.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 5. Verify stop token ends generation
        # With raw_ids [1, 2, 3, 4], the greedy tokens are [25190, 6074, 6566, ...]
        stop_token = 6566
        print(f"Testing stop token {stop_token} ends generation...", flush=True)
        model.new_session()
        vxpu_stop = model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=stop_token,
        )
        ref_stop = ref_model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=20,
            do_sample=False,
            eos_token_id=stop_token,
        )
        print(f"vxpu stop output:         {vxpu_stop[0].tolist()}", flush=True)
        print(f"transformers stop output: {ref_stop[0].tolist()}", flush=True)
        if not torch.equal(vxpu_stop, ref_stop):
            print(
                f"ERROR: Stop token mismatch: vxpu={vxpu_stop.tolist()} vs ref={ref_stop.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)
        if vxpu_stop[0, -1].item() != stop_token:
            print(
                f"ERROR: Expected last token to be stop token {stop_token}, got {vxpu_stop[0, -1].item()}",
                file=sys.stderr,
            )
            sys.exit(1)
        if model.last.finish_reason != "eos":
            print(
                f"ERROR: Expected finish_reason 'eos', got {model.last.finish_reason!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        if model.last.generated != 3:
            print(
                f"ERROR: Expected 3 generated tokens before stop, got {model.last.generated}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 6. Verify cancelled generate stops promptly and frees session on executor
        print("Testing cancelled generate stops promptly and frees session...", flush=True)
        streamer = CancelOnTokenStreamer(stop_after_tokens=2)
        start_cancel = time.perf_counter()
        cancelled = False
        try:
            model.generate(
                raw_ids,
                attention_mask=raw_attn,
                max_new_tokens=50,
                do_sample=False,
                streamer=streamer,
            )
        except GeneratorExit:
            cancelled = True
        cancel_duration = time.perf_counter() - start_cancel
        print(f"Cancelled in {cancel_duration * 1000:.1f}ms", flush=True)
        if not cancelled:
            print("ERROR: generate did not raise GeneratorExit on cancel", file=sys.stderr)
            sys.exit(1)

        # Immediate follow-up generate on the same session must succeed promptly without hang
        vxpu_followup = model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=5,
            do_sample=False,
        )
        ref_followup = ref_model.generate(
            raw_ids,
            attention_mask=raw_attn,
            max_new_tokens=5,
            do_sample=False,
        )
        print(f"vxpu followup output:         {vxpu_followup[0].tolist()}", flush=True)
        print(f"transformers followup output: {ref_followup[0].tolist()}", flush=True)
        if not torch.equal(vxpu_followup, ref_followup):
            print(
                f"ERROR: Followup after cancel mismatch: vxpu={vxpu_followup.tolist()} vs ref={ref_followup.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        model.close()

    print(
        "Facade tests passed: greedy equivalence, suffix prefill, sampling, stop tokens, and cancellation verified!",
        flush=True,
    )


if __name__ == "__main__":
    main()
