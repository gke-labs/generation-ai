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
import zipfile

import safetensors.torch
import torch
import transformers

from . import deepseek_v41 as ds
from . import export
from . import plan
from . import rehydrate
from .engine import Engine, SamplingParams
from .export import export_artifact


def _write_tiny_model(out_dir: str, *, drafter: bool) -> str:
    """Write a tiny DeepSeek checkpoint, skipping when the fork is absent unless CI requires it."""
    try:
        from .fixtures import tiny_deepseek_v41
        return tiny_deepseek_v41.write(out_dir, drafter=drafter)
    except (ImportError, ModuleNotFoundError) as e:
        if plan.require_model_source():
            raise
        raise unittest.SkipTest(f"tiny_deepseek_v41 unavailable: {e}") from e


def _generate_tokens(engine: Engine, session_id: str, prompt: list[int], max_new_tokens: int) -> list[int]:
    """Run greedy generation on an engine session and return the generated token IDs."""
    tokens: list[int] = []
    gen: Any = getattr(engine, "generate")(session_id, prompt, max_new_tokens, SamplingParams())
    for chunk in cast(list[dict[str, Any]], list(gen)):
        tokens.extend(cast(list[int], chunk.get("token_ids", [])))
    return tokens


class ExportDrafterTest(unittest.TestCase):
    """Verifies that an exported model carries the drafter and verify programs and stated facts."""

    _tmp: tempfile.TemporaryDirectory[str]
    _drafter_model_dir: str
    _drafter_art_dir: str
    _drafter_nodraft_art_dir: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        try:
            cls._drafter_model_dir = _write_tiny_model(
                os.path.join(cls._tmp.name, "model_drafter"), drafter=True
            )
            cls._drafter_art_dir = os.path.join(cls._tmp.name, "art_drafter")
            export_artifact(cls._drafter_model_dir, cls._drafter_art_dir, max_cache_len=64, draft=True)
            cls._drafter_nodraft_art_dir = os.path.join(cls._tmp.name, "art_drafter_nodraft")
            export_artifact(cls._drafter_model_dir, cls._drafter_nodraft_art_dir, max_cache_len=64, draft=False)
        except BaseException:
            cls._tmp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_export_tiny_deepseek_with_drafter(self) -> None:
        """Exporting tiny DeepSeek with --draft creates draft.pt2 and verify.pt2 and records facts."""
        art_dir = self._drafter_art_dir

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
        """Outside the engine, draft proposals match eager spec_draft and verify matches 1-token decode."""
        model_dir = self._drafter_model_dir
        art_dir = self._drafter_art_dir

        with ds.bf16_default():
            ds_fork: Any = getattr(ds, "_fork")()
            args: Any = getattr(ds, "model_args")(model_dir, 64, step_tokens=4)
            tok_fn: Any = getattr(transformers.AutoTokenizer, "from_pretrained")
            tokenizer: Any = tok_fn(model_dir)
            eager: Any = getattr(ds_fork, "Transformer")(args, tokenizer).eval()

            load_file_fn: Any = getattr(safetensors.torch, "load_file")
            st_dict = cast(dict[str, torch.Tensor], load_file_fn(os.path.join(model_dir, "model.safetensors")))
            state: dict[str, torch.Tensor] = dict(st_dict)
            bank = re.compile(
                r"^(?P<prefix>(?:layers|mtp)\.\d+\.ffn)\.experts\.(?P<i>\d+)\.(?P<proj>w[123])\.(?P<kind>weight|scale)$"
            )
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
                for fqn, (shape, dtype) in state_specs.items()
            }
            shared_tensors: dict[str, torch.Tensor] = getattr(engine, "shared")
            tensors: dict[str, torch.Tensor] = {**shared_tensors, **scratch}
            aliases: dict[str, str] = getattr(engine, "aliases")
            tensors.update({
                alias: tensors[canonical]
                for alias, canonical in aliases.items()
                if canonical in tensors
            })

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
                    _, step_logits, _ = eager_single(verify_row[:, t : t + 1], torch.tensor([4 + t]))
                    single_choices.append(cast(torch.Tensor, step_logits.argmax(-1).squeeze()))
                single_choices_tensor = torch.stack(single_choices, dim=-1).unsqueeze(0)

                verify_logits = cast(torch.Tensor, verify_mod(input_ids=verify_row, cache_position=pos))
                verify_choices = verify_logits.argmax(-1)
                self.assertTrue(torch.equal(verify_choices, single_choices_tensor))

    def test_engine_generates_on_drafter_artifact_with_short_prompts(self) -> None:
        """Engine.generate on a drafter artifact matches the non-drafter export for prompts and
        follow-up turns shorter than min_prefill_len (such as 2 or 3 tokens) as well as longer prompts."""
        eng_draft = Engine(self._drafter_art_dir, device="cpu")
        eng_nodraft = Engine(self._drafter_nodraft_art_dir, device="cpu")

        for prompt in ([5], [5, 12], [5, 12, 19], [5, 12, 19, 26]):
            t_draft = _generate_tokens(eng_draft, eng_draft.new_session(), prompt, 6)
            t_nodraft = _generate_tokens(eng_nodraft, eng_nodraft.new_session(), prompt, 6)
            self.assertEqual(t_draft, t_nodraft, f"mismatch for prompt {prompt}")

        # Multi-turn follow-ups of 2 and 3 new tokens (shorter than min_prefill_len=4).
        turn1 = [5, 12, 19, 26]
        for extra in ([33, 40], [33, 40, 47]):
            s_draft = eng_draft.new_session()
            s_nodraft = eng_nodraft.new_session()
            gen1_draft = _generate_tokens(eng_draft, s_draft, turn1, 2)
            gen1_nodraft = _generate_tokens(eng_nodraft, s_nodraft, turn1, 2)
            self.assertEqual(gen1_draft, gen1_nodraft)

            turn2 = turn1 + gen1_draft + extra
            t_draft = _generate_tokens(eng_draft, s_draft, turn2, 6)
            t_nodraft = _generate_tokens(eng_nodraft, s_nodraft, turn2, 6)
            self.assertEqual(t_draft, t_nodraft, f"mismatch for follow-up {extra}")

    def test_export_without_drafter_omits_draft_programs_and_facts(self) -> None:
        """Exporting without --draft omits draft.pt2, verify.pt2, draft, and step_tokens,
        keeps min_prefill_len at 2, and is deterministic across repeated exports."""
        # A checkpoint with a drafter exported without --draft omits draft programs and facts.
        self.assertFalse(os.path.exists(os.path.join(self._drafter_nodraft_art_dir, "draft.pt2")))
        self.assertFalse(os.path.exists(os.path.join(self._drafter_nodraft_art_dir, "verify.pt2")))
        with open(os.path.join(self._drafter_nodraft_art_dir, "binding.json")) as f:
            b_nodraft = cast(dict[str, Any], json.load(f))
        self.assertNotIn("draft", b_nodraft)
        self.assertNotIn("step_tokens", b_nodraft)
        self.assertEqual(b_nodraft.get("min_prefill_len"), 2)

        # A checkpoint without a drafter also omits draft programs and facts and exports deterministically.
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_tiny_model(os.path.join(td, "model"), drafter=False)
            art1 = os.path.join(td, "art1")
            art2 = os.path.join(td, "art2")
            _ = export_artifact(model_dir, art1, max_cache_len=64, draft=False)
            _ = export_artifact(model_dir, art2, max_cache_len=64, draft=False)

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
            eng2 = Engine(art2, device="cpu")
            t1 = _generate_tokens(eng1, eng1.new_session(), [5, 12, 19], 4)
            t2 = _generate_tokens(eng2, eng2.new_session(), [5, 12, 19], 4)
            self.assertEqual(t1, t2)

    def test_tiny_deepseek_rotary_tables_deduplicated(self) -> None:
        """Exported tiny DeepSeek aliases layers 2 and 3 freqs_cis to layer 1 and keeps decode.pt2 compact."""
        art = self._drafter_nodraft_art_dir
        with open(os.path.join(art, "binding.json")) as f:
            binding = cast(dict[str, Any], json.load(f))
        aliases = cast(dict[str, str], binding.get("aliases", {}))
        constants = set(cast(list[str], binding.get("constant", [])))
        for layer in (2, 3):
            fqn = f"model.layers.{layer}.attn.freqs_cis"
            self.assertEqual(aliases.get(fqn), "model.layers.1.attn.freqs_cis")
            self.assertNotIn(fqn, constants)
        self.assertIn("model.layers.1.attn.freqs_cis", constants)
        self.assertIn("model.layers.0.attn.freqs_cis", constants)
        with zipfile.ZipFile(os.path.join(art, "decode.pt2")) as z:
            blobs = [f for f in z.namelist() if "constants/tensor_" in f]
            self.assertLessEqual(len(blobs), 10)
        assert_no_dups: Any = getattr(export, "assert_no_duplicate_constants")
        assert_no_dups(os.path.join(art, "decode.pt2"))
        assert_no_dups(os.path.join(art, "prefill.pt2"))


class DeduplicateConstantsTest(unittest.TestCase):
    """Verifies export.deduplicate_constants on synthetic programs without building an Engine."""

    def test_conflicting_names_produce_no_aliases(self) -> None:
        """Identical buffer names holding different values across programs produce no aliases."""
        dedup_fn: Any = getattr(export, "deduplicate_constants")

        class MConf(torch.nn.Module):
            c0: torch.Tensor
            c1: torch.Tensor

            def __init__(self, c0: torch.Tensor, c1: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("c0", c0, persistent=False)
                self.register_buffer("c1", c1, persistent=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.c0 + self.c1

        x, y, z = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
        p_pref = torch.export.export(MConf(x, z), (torch.zeros(1),), strict=False)
        p_dec = torch.export.export(MConf(y, x), (torch.zeros(1),), strict=False)
        aliases = dedup_fn({}, p_pref, p_dec)
        self.assertEqual(aliases, {})
        self.assertIn("c0", p_pref.constants)
        self.assertIn("c1", p_pref.constants)
        self.assertIn("c0", p_dec.constants)
        self.assertIn("c1", p_dec.constants)

    def test_canonical_constant_in_other_program(self) -> None:
        """Duplicate buffers within one program alias to a canonical buffer present in that program."""
        dedup_fn: Any = getattr(export, "deduplicate_constants")
        assert_no_dups: Any = getattr(export, "assert_no_duplicate_constants")

        class M1(torch.nn.Module):
            c0: torch.Tensor

            def __init__(self, t: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("c0", t, persistent=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.c0

        class M2(torch.nn.Module):
            c1: torch.Tensor
            c2: torch.Tensor

            def __init__(self, t: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("c1", t, persistent=False)
                self.register_buffer("c2", t, persistent=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.c1 + self.c2

        t = torch.tensor([42.0])
        p1 = torch.export.export(M1(t), (torch.zeros(1),), strict=False)
        p2 = torch.export.export(M2(t), (torch.zeros(1),), strict=False)
        aliases = dedup_fn({}, p1, p2)
        self.assertEqual(aliases, {"c2": "c1"})
        self.assertEqual(list(p2.constants.keys()), ["c1"])
        self.assertEqual(list(p1.constants.keys()), ["c0"])
        assert_no_dups(p1)
        assert_no_dups(p2)
