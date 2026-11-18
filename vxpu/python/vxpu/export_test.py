# pyright: strict
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

"""Tests for exporting models with and without a drafter, program execution, and stated facts."""

import json
import os
import re
import tempfile
import unittest
from typing import Any, cast

import torch
import safetensors.torch
import transformers

from . import deepseek_v41 as ds
from .engine import Engine, SamplingParams
from .export import export_artifact
from . import rehydrate


class ExportDrafterTest(unittest.TestCase):
    """Verifies that an exported model carries the drafter and verify programs and stated facts."""

    def test_export_tiny_deepseek_with_drafter(self) -> None:
        """Verifies that exporting tiny DeepSeek with drafter creates draft.pt2, verify.pt2,
        and records step_tokens and draft.block in binding.json."""
        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e

        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir: str = tiny_deepseek_v41.write(os.path.join(td, "model"), drafter=True)
            except (ImportError, ModuleNotFoundError) as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 fork unavailable: {e}") from e

            art_dir: str = os.path.join(td, "artifact")
            _manifest: dict[str, Any]
            _binding: dict[str, Any]
            _manifest, _binding = export_artifact(model_dir, art_dir, draft=True)

            self.assertTrue(os.path.exists(os.path.join(art_dir, "prefill.pt2")))
            self.assertTrue(os.path.exists(os.path.join(art_dir, "decode.pt2")))
            self.assertTrue(os.path.exists(os.path.join(art_dir, "verify.pt2")))
            self.assertTrue(os.path.exists(os.path.join(art_dir, "draft.pt2")))

            with open(os.path.join(art_dir, "binding.json")) as f:
                binding = cast(dict[str, Any], json.load(f))

            self.assertEqual(binding.get("step_tokens"), 4)
            self.assertEqual(binding.get("min_prefill_len"), 4)
            draft_meta = cast(dict[str, Any], binding.get("draft", {}))
            self.assertEqual(draft_meta.get("block"), 3)

    def test_drafter_and_verify_programs_direct(self) -> None:
        """Run directly outside the engine: drafter proposals equal eager spec_draft proposals,
        and multi-token verify choices equal 1-token decode choices one at a time."""
        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e

        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir: str = tiny_deepseek_v41.write(os.path.join(td, "model"), drafter=True)
            except (ImportError, ModuleNotFoundError) as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 fork unavailable: {e}") from e

            art_dir: str = os.path.join(td, "artifact")
            _manifest: dict[str, Any]
            _binding: dict[str, Any]
            _manifest, _binding = export_artifact(model_dir, art_dir, draft=True)

            torch.set_default_dtype(torch.bfloat16)
            ds_fork: Any = getattr(ds, "_fork")()
            args: Any = getattr(ds, "model_args")(model_dir, 64, step_tokens=4)
            tok_fn: Any = getattr(transformers.AutoTokenizer, "from_pretrained")
            tokenizer: Any = tok_fn(model_dir)
            eager: Any = getattr(ds_fork, "Transformer")(args, tokenizer).eval()

            load_file_fn: Any = getattr(safetensors.torch, "load_file")
            st_dict = cast(dict[str, torch.Tensor], load_file_fn(os.path.join(model_dir, "model.safetensors")))
            state: dict[str, torch.Tensor] = dict(st_dict)
            bank = re.compile(
                r"^(?P<prefix>(?:layers|mtp)\.\d+\.ffn)\.experts\.(?P<i>\d+)\.(?P<proj>w[123])\.(?P<kind>weight|scale)$")
            groups: dict[str, list[tuple[int, torch.Tensor]]] = {}
            for name in list(state):
                m = bank.match(name)
                if m:
                    prefix = m.group("prefix")
                    proj = m.group("proj")
                    kind = m.group("kind")
                    key = f"{prefix}.{proj}" + ("" if kind == "weight" else "_scale")
                    groups.setdefault(key, []).append((int(m.group("i")), state.pop(name)))
            for key, parts in groups.items():
                state[key] = torch.stack([t for _, t in sorted(parts, key=lambda p: p[0])], dim=0)
            _ = getattr(eager, "load_state_dict")(state, strict=False)

            engine = Engine(art_dir, device="cpu")
            state_specs: dict[str, tuple[torch.Size, torch.dtype]] = getattr(engine, "state_specs")
            scratch: dict[str, torch.Tensor] = {
                fqn: torch.zeros(shape, dtype=dtype, device="cpu")
                for fqn, (shape, dtype) in state_specs.items()}
            shared_tensors: dict[str, torch.Tensor] = getattr(engine, "shared")
            tensors: dict[str, torch.Tensor] = {**shared_tensors, **scratch}
            aliases: dict[str, str] = getattr(engine, "aliases")
            tensors.update({alias: tensors[canonical]
                            for alias, canonical in aliases.items()
                            if canonical in tensors})

            load_prog_fn: Any = getattr(rehydrate, "load_program")
            prefill_ep: Any = load_prog_fn(os.path.join(art_dir, "prefill.pt2"), tensors, "cpu")
            verify_ep: Any = load_prog_fn(os.path.join(art_dir, "verify.pt2"), tensors, "cpu")
            draft_ep: Any = load_prog_fn(os.path.join(art_dir, "draft.pt2"), tensors, "cpu")

            prefill_mod: Any = prefill_ep.module()
            verify_mod: Any = verify_ep.module()
            draft_mod: Any = draft_ep.module()

            binding = cast(dict[str, Any], getattr(engine, "binding"))
            state_fqns = set(cast(list[str], binding["state"]))
            share_st_fn: Any = getattr(rehydrate, "share_state")
            _ = share_st_fn(prefill_mod, verify_mod, state_fqns)
            _ = share_st_fn(prefill_mod, draft_mod, state_fqns)

            prompt = torch.tensor([[5, 12, 19, 26]])
            with engine.external_scope():
                _, _, eager_hidden = eager(prompt, 0)
                getattr(eager, "spec_observe")(eager_hidden, 0)

                _ = prefill_mod(input_ids=prompt, cache_position=torch.arange(4))

                last_token = prompt[:, -1:]
                pos = torch.tensor([4])
                eager_draft_tokens: Any
                eager_draft_tokens, _, _ = getattr(eager, "spec_draft")(last_token.view(-1), pos)
                eager_proposals = cast(torch.Tensor, eager_draft_tokens[:, 1:])

                draft_proposals = cast(torch.Tensor, draft_mod(input_ids=last_token, cache_position=pos))

                self.assertTrue(torch.equal(draft_proposals, eager_proposals))

                verify_row = torch.cat([last_token, draft_proposals], dim=1)

                eager_single: Any = getattr(ds_fork, "Transformer")(args, tokenizer).eval()
                _ = getattr(eager_single, "load_state_dict")(getattr(eager, "state_dict")(), strict=False)
                _ = eager_single(prompt, 0)

                single_choices: list[torch.Tensor] = []
                for t in range(4):
                    step_logits: Any
                    _, step_logits, _ = eager_single(verify_row[:, t:t+1], torch.tensor([4 + t]))
                    single_choices.append(cast(torch.Tensor, step_logits.argmax(-1).squeeze()))
                single_choices_tensor = torch.stack(single_choices, dim=-1).unsqueeze(0)

                verify_logits = cast(torch.Tensor, verify_mod(input_ids=verify_row, cache_position=pos))
                verify_choices = verify_logits.argmax(-1)

                self.assertTrue(torch.equal(verify_choices, single_choices_tensor))

    def test_export_without_drafter_unchanged(self) -> None:
        """Verifies that exporting without drafter produces no draft or verify program,
        omits draft and step_tokens facts, produces the same content, and engine generation
        tokens match."""
        try:
            from .fixtures import tiny_deepseek_v41
        except ImportError as e:
            raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e

        with tempfile.TemporaryDirectory() as td:
            try:
                model_dir: str = tiny_deepseek_v41.write(os.path.join(td, "model"), drafter=False)
            except (ImportError, ModuleNotFoundError) as e:
                raise unittest.SkipTest(f"tiny_deepseek_v41 fork unavailable: {e}") from e

            art1: str = os.path.join(td, "art1")
            art2: str = os.path.join(td, "art2")
            _m1: dict[str, Any]
            _b1: dict[str, Any]
            _m2: dict[str, Any]
            _b2: dict[str, Any]
            _m1, _b1 = export_artifact(model_dir, art1, draft=False)
            _m2, _b2 = export_artifact(model_dir, art2, draft=False)

            self.assertFalse(os.path.exists(os.path.join(art1, "draft.pt2")))
            self.assertFalse(os.path.exists(os.path.join(art1, "verify.pt2")))

            with open(os.path.join(art1, "binding.json")) as f:
                b1 = cast(dict[str, Any], json.load(f))
            with open(os.path.join(art2, "binding.json")) as f:
                b2 = cast(dict[str, Any], json.load(f))

            self.assertNotIn("draft", b1)
            self.assertNotIn("step_tokens", b1)
            self.assertEqual(b1.get("min_prefill_len"), 2)

            self.assertEqual(set(cast(list[str], b1["unused"])), set(cast(list[str], b2["unused"])))
            b1_clean = {k: v for k, v in b1.items() if k != "unused"}
            b2_clean = {k: v for k, v in b2.items() if k != "unused"}
            self.assertEqual(b1_clean, b2_clean)

            with open(os.path.join(art1, "manifest.json")) as f:
                m1 = cast(dict[str, Any], json.load(f))
            with open(os.path.join(art2, "manifest.json")) as f:
                m2 = cast(dict[str, Any], json.load(f))
            self.assertEqual(m1, m2)

            eng1 = Engine(art1, device="cpu")
            s1: str = eng1.new_session()
            t1: list[int] = []
            gen1: Any = getattr(eng1, "generate")(s1, [5, 12, 19], 4, SamplingParams())
            for chunk in cast(list[dict[str, Any]], list(gen1)):
                t1.extend(cast(list[int], chunk.get("token_ids", [])))

            eng2 = Engine(art2, device="cpu")
            s2: str = eng2.new_session()
            t2: list[int] = []
            gen2: Any = getattr(eng2, "generate")(s2, [5, 12, 19], 4, SamplingParams())
            for chunk in cast(list[dict[str, Any]], list(gen2)):
                t2.extend(cast(list[int], chunk.get("token_ids", [])))

            self.assertEqual(t1, t2)
