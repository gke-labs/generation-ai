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

"""Tests for chat tokenization and vocabulary export."""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from typing import Any, cast
import unittest

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from . import engine
from . import vxpu_pb2
from .server import ExecutorServicer

TokenizeChatRequest: Any = getattr(vxpu_pb2, "TokenizeChatRequest")
ChatMessage: Any = getattr(vxpu_pb2, "ChatMessage")


def verify_vocab_decoding_equality(
    tok: PreTrainedTokenizerBase,
    pieces: list[bytes],
    num_random_sequences: int = 3000,
) -> None:
    """Verifies that concatenating pieces equals decoding after a prompt."""
    prompt_ids_raw = cast(list[int], tok.encode("User: Hello\nAssistant:", add_special_tokens=False))
    prompt_ids: list[int] = [int(x) for x in prompt_ids_raw]
    prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)

    vocab_size = len(tok)
    rng = random.Random(42)
    for _ in range(num_random_sequences):
        seq_len = rng.randint(1, 10)
        seq = [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
        concat_bytes = b"".join(pieces[t] for t in seq if t < len(pieces))
        reconstructed = concat_bytes.decode("utf-8", errors="replace")
        full = tok.decode(prompt_ids + seq, skip_special_tokens=True)
        from_prompt = full[len(prompt_text):]
        if reconstructed != from_prompt:
            raise AssertionError(
                f"Mismatch on sequence {seq}:\n"
                f"  concatenated: {reconstructed!r}\n"
                f"  from prompt:  {from_prompt!r}"
            )


class VocabTest(unittest.TestCase):

    def test_tokenize_chat_options(self) -> None:
        """Tests that chat template kwargs preserve boolean and string values."""
        try:
            tok = AutoTokenizer.from_pretrained(
                "trl-internal-testing/tiny-random-LlamaForCausalLM", local_files_only=False)
        except Exception as e:
            self.skipTest(f"tiny-random-LlamaForCausalLM unavailable: {e}")
            return
        tok.chat_template = (
            "{% if enable_thinking %}YES_THINK{% else %}NO_THINK{% endif %}: "
            "{{ style }}: {{ messages[0]['content'] }}"
        )

        td = tempfile.mkdtemp(prefix="vxpu-vocab-test-")
        self.addCleanup(shutil.rmtree, td, True)
        servicer = ExecutorServicer(work_dir=td, device="cpu")

        class MockEngine:
            def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
                self.tokenizer: PreTrainedTokenizerBase = tokenizer

            def tokenize_chat(self, messages: list[dict[str, str]], add_generation_prompt: bool = True, **kwargs: Any) -> list[int]:
                res: Any = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=add_generation_prompt,
                    tokenize=True, return_tensors=None, **kwargs)
                if hasattr(res, "get") and "input_ids" in res:
                    res = res["input_ids"]
                elif hasattr(res, "__getitem__") and "input_ids" in res:
                    res = res["input_ids"]
                if hasattr(res, "tolist"):
                    res = res.tolist()
                return [int(x) for x in res]

        servicer.engine = cast(Any, MockEngine(tok))

        class MockContext:
            code: Any = None
            details: str = ""
            def set_code(self, code: Any) -> None: self.code = code
            def set_details(self, details: str) -> None: self.details = details

        # 1. Test enable_thinking=False (boolean must arrive as False, not string "false")
        req_false = TokenizeChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            chat_template_kwargs_json=json.dumps({"enable_thinking": False, "style": "brief"}),
        )
        resp_false = servicer.TokenizeChat(req_false, MockContext())
        decoded_false = tok.decode(list(resp_false.input_ids))
        self.assertIn("NO_THINK: brief: Hi", decoded_false)
        self.assertNotIn("YES_THINK: brief: Hi", decoded_false)

        # 2. Test enable_thinking=True
        req_true = TokenizeChatRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            chat_template_kwargs_json=json.dumps({"enable_thinking": True, "style": "detailed"}),
        )
        resp_true = servicer.TokenizeChat(req_true, MockContext())
        decoded_true = tok.decode(list(resp_true.input_ids))
        self.assertIn("YES_THINK: detailed: Hi", decoded_true)

    def test_vocab_export_equality(self) -> None:
        """Tests vocabulary piece concatenation against tokenizer decoding without building an Engine."""
        try:
            tok = AutoTokenizer.from_pretrained(
                "trl-internal-testing/tiny-random-LlamaForCausalLM", local_files_only=False)
        except Exception as e:
            self.skipTest(f"tiny-random-LlamaForCausalLM unavailable: {e}")
            return

        pieces, specials = engine.export_vocab_pieces(tok)
        self.assertGreater(len(pieces), 0)
        eos_id = getattr(tok, "eos_token_id", 2) or 2
        self.assertIn(eos_id, specials)

        # Sequence with multi-byte characters split across tokens, first token of reply, and special tokens:
        # [10994 ('Hello' without leading space), 3186 (' world'), 29991 ('!'), 29871, 243, 162, 143, 162 (' 🌟'), 2 (EOS)]
        test_seq = [10994, 3186, 29991, 29871, 243, 162, 143, 162, 2]
        decoded_pieces = b"".join(pieces[t] for t in test_seq).decode("utf-8")
        prompt_ids = [1060]  # 'X'
        prompt_text = tok.decode(prompt_ids, skip_special_tokens=True)
        from_prompt = tok.decode(prompt_ids + test_seq, skip_special_tokens=True)[len(prompt_text):]
        self.assertEqual(decoded_pieces, from_prompt)
        self.assertEqual(decoded_pieces, "Hello world! 🌟")

        # Verify on 3,000 random sequences under prompt+reply definition.
        verify_vocab_decoding_equality(tok, pieces, num_random_sequences=3000)

    def test_released_model_tokenizer(self) -> None:
        """Tests vocabulary export against released model tokenizer files if provided."""
        path = os.environ.get("VXPU_TOKENIZER_PATH")
        if not path or not os.path.exists(path):
            self.skipTest(
                "local tokenizer files for DeepSeek-V4.1-Flash not found "
                "(set VXPU_TOKENIZER_PATH=/path/to/tokenizer to test against released files)")
            return
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        pieces, _ = engine.export_vocab_pieces(tok)
        self.assertGreater(len(pieces), 0)
        verify_vocab_decoding_equality(tok, pieces, num_random_sequences=3000)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify vocabulary piece concatenation against released model tokenizer files on disk")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to directory containing DeepSeek-V4.1-Flash tokenizer files (tokenizer.json, tokenizer_config.json)",
    )
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.tokenizer):
        sys.exit(f"Tokenizer path does not exist: {args.tokenizer}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    pieces, specials = engine.export_vocab_pieces(tok)
    print(f"Loaded {args.tokenizer}: vocab_size={len(pieces)}, specials={len(specials)}")
    verify_vocab_decoding_equality(tok, pieces, num_random_sequences=3000)
    print(f"PASS: Vocabulary decoding equality verified for 3,000 random sequences on {args.tokenizer}!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        main()
    else:
        unittest.main()
