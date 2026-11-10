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

"""Test vectors: what a model says, recorded by any producer, compared.

A vectors file is the interchange format between engines. Eager
transformers records the reference; the vXPU engine (in-process, or
through a router) and vLLM record candidates. ``compare`` diffs two
files: where the candidate's greedy tokens leave the reference's, the
reference's logits at that position say whether the flip was a tie or
a divergence, and the top-k logits of every position up to there must
agree within a tolerance.

    python -m vxpu.vectors_test record --producer transformers --model REPO -o ref.json
    python -m vxpu.vectors_test record --producer vxpu-engine --model REPO --artifact DIR -o got.json
    python -m vxpu.vectors_test record --producer vxpu-router --model REPO --artifact DIR [--router host:port] -o got.json
    python -m vxpu.vectors_test record --producer vllm --model REPO -o vllm.json  # or: python vectors_test.py ...
    python -m vxpu.vectors_test compare ref.json got.json
    python -m vxpu.vectors_test selftest  # the CI tier: tiny models, in-process

Format (vxpu-vectors/1):

    {"format": "vxpu-vectors/1",
     "model": {"repo": ..., "revision": ..., "commit_sha": ...},
     "producer": {"kind": "transformers" | "vxpu-engine" | "vxpu-router" | "vllm",
                  "torch": ..., "transformers": ..., "device": ..., ...},
     "settings": {"new_tokens": 32, "top_k": 32, "logit_positions": 32,
                  "greedy": true},   # logit_positions defaults to new_tokens
     "prompts": [{"name": ..., "text": ... | "ids": [...],
                  "input_ids": [...], "generated_ids": [...],
                  "finish": "eos" | "length",
                  "top_logits": [{"ids": [...], "values": [...]}, ...],
                  "prefill_ms": ..., "ms_per_token": ...}]}

Logits are stored as the top-k (id, value) pairs of each generated
position (``logit_positions``, every generated position unless set
lower; a few tens of KB per prompt): they are what lets a token flip be judged as a tie or a
divergence, and they stay useful under any later tolerance.
"""

import argparse
import collections
import json
import os
import sys
import tempfile
import time

from . import vxpu_pb2
from .export import assert_no_duplicate_constants

PROMPTS = [
    {"name": "sky", "text": "Is the sky blue?"},
    {"name": "kubernetes", "text": "Write one sentence about Kubernetes."},
    {"name": "long", "text": (
        "Kubernetes schedules containers onto nodes, restarts them when "
        "they fail, and scales them with demand. ") * 24
        + "Summarize the above in one sentence."},
]

# Prompts as token ids, for models without a tokenizer (test fixtures).
ID_PROMPTS = [
    {"name": "ids-short", "ids": [5, 17, 42, 99, 7, 300, 11, 128, 64, 2, 250, 33, 8]},
    {"name": "ids-long", "ids": [(i * 37 + 11) % 500 + 3 for i in range(100)]},
]

DEFAULTS = {"new_tokens": 32, "top_k": 32, "logit_positions": None, "greedy": True}
FORMAT = "vxpu-vectors/1"


def _versions(device):
    import torch
    import transformers
    out = {"torch": torch.__version__, "transformers": transformers.__version__,
           "device": str(device)}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        # The kernel path follows from the card, so name the one used.
        out["gpu"] = torch.cuda.get_device_name(torch.device(str(device)))
    return out


def _r(value):
    return None if value is None else round(float(value), 1)


def _top(logits, k):
    import torch
    values, ids = torch.topk(logits.detach().float().cpu(), k)
    return {"ids": ids.tolist(), "values": [round(v, 5) for v in values.tolist()]}


def _encode(tokenizer, prompt):
    """input_ids for a prompt: text through the chat template, or raw ids."""
    if "ids" in prompt:
        return list(prompt["ids"])
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            add_generation_prompt=True, return_tensors="pt",
            return_dict=True)["input_ids"][0].tolist()
    return tokenizer(prompt["text"])["input_ids"]  # base model: plain text


def _tokenizer(model, revision, prompts="auto"):
    if prompts == "ids":
        return None
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model, revision=revision)
    except Exception:  # no tokenizer files, or a backend missing: ids prompts
        return None


def _progress(prompt, new, ms_per_token):
    """One line per finished prompt on stderr, so a long run (the real
    DeepSeek reference takes hours on a CPU) shows where it is."""
    rate = f"{ms_per_token:.0f} ms/token" if ms_per_token else "no timing"
    print(f"[vectors] {prompt['name']}: {len(new)} tokens, {rate}", file=sys.stderr, flush=True)


def _prompt_set(tokenizer, prompts):
    if prompts == "ids" or (prompts == "auto" and tokenizer is None):
        return ID_PROMPTS
    return PROMPTS


def record_transformers(model, revision, prompts, settings, device="cpu", dtype="auto",
                        dequantize=False):
    """Reference: eager transformers, greedy, logits from generate().

    ``dtype="auto"`` runs the model in the dtype its config declares,
    which is what the exporter captures; a different reference dtype
    would compare two different computations. ``dequantize`` asks
    transformers to expand an fp8 checkpoint to bf16 at load (the same
    weight values, materialized once), which is far faster on a CPU
    than dequantizing per step.
    """
    import torch
    from transformers import AutoModelForCausalLM
    tok = _tokenizer(model, revision, prompts)
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    kwargs = {"dtype": torch_dtype, "revision": revision}
    if dequantize:
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    if device != "cpu":
        kwargs["device_map"] = device
    lm = AutoModelForCausalLM.from_pretrained(model, **kwargs).eval()
    eos = lm.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([eos] if eos is not None else []))
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        inputs = torch.tensor([ids], device=lm.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = lm.generate(inputs, max_new_tokens=settings["new_tokens"],
                                 do_sample=False, output_logits=True,
                                 return_dict_in_generate=True)
        elapsed = time.perf_counter() - t0
        new = result.sequences[0, len(ids):].tolist()
        top = [_top(step[0], settings["top_k"])
               for step in result.logits[:settings["logit_positions"]]]
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": "eos" if new and new[-1] in eos_ids else "length",
                    "top_logits": top,
                    "ms_per_token": round(1000 * elapsed / max(1, len(new)), 1)})
        _progress(prompt, new, 1000 * elapsed / max(1, len(new)))
    producer = {"kind": "transformers", **_versions(lm.device), "dtype": str(lm.dtype),
                "dequantize": dequantize,
                "experts_implementation": getattr(lm.config, "_experts_implementation", None)}
    return _document(model, revision, lm.config, producer, settings, out)


def _download(url, dest, size=None):
    """Stream url to dest (resumes nothing; a complete file of the
    expected size is kept). Sends the Hub token when one is configured."""
    import requests
    from huggingface_hub.utils import build_hf_headers
    if size is not None and os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    with requests.get(url, headers=build_hf_headers(), stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    os.replace(part, dest)
    return dest


def record_deepseek_eager(model, revision, prompts, settings):
    """Reference for a deepseek_v41 checkpoint: the model maker's module
    from the fork, run eagerly with the plain-aten kernels on the CPU (no
    transformers implementation exists; the plain-aten kernels are CPU
    code, so there is no device to choose). Greedy; logits at every step."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from . import deepseek_v41 as ds
    from .manifest import build_manifest
    manifest = build_manifest(model, revision)
    tok = _tokenizer(model, revision, prompts)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    fork = ds._fork()
    args = ds.model_args(model, settings.get("max_cache_len", 1024))
    with ds.bf16_default():
        lm = fork.Transformer(args, tokenizer).eval()
    # Load shard by shard into the module: the checkpoint never coexists
    # with the parameters in memory, so a 510 GB model needs ~510 GB, not
    # twice that. Remote shards are streamed a few at a time ahead of the
    # loader and deleted once loaded (hf_hub_download refuses files over
    # 50 GB without its xet client, and the engram shards are ~100 GB).
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_url
    cache = os.path.join(os.environ.get("VXPU_SHARD_CACHE", os.path.join(tempfile.gettempdir(), "vxpu-shards")),
                         model.replace("/", "--"))

    def fetch(ref):
        if ref["source"].startswith("file://"):
            return ref["source"][len("file://"):], False
        return _download(hf_hub_url(model, ref["name"], revision=revision),
                         os.path.join(cache, ref["name"]), ref.get("size")), True

    # Every tensor the module declares must come from some shard, else the
    # reference would run on random initialisation and the comparator would
    # blame the engine. A key is missing only if every shard load reported
    # it so; expert banks are assembled from per-expert checkpoint tensors
    # (the module's _load_from_state_dict), so they are checked by count.
    import re
    expert_re = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
    missing_all, experts_seen = None, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, ref) for ref in sorted(manifest["files"].values(), key=lambda r: r["name"])]
        for fut in futures:
            path, fetched = fut.result()
            shard = load_file(path)
            for key in shard:
                m = expert_re.match(key)
                if m:
                    experts_seen.setdefault((int(m.group(1)), m.group(3), m.group(4)), set()).add(int(m.group(2)))
            missing, unexpected = lm.load_state_dict(shard, strict=False)
            missing_all = set(missing) if missing_all is None else missing_all & set(missing)
            unexpected = [u for u in unexpected if not u.startswith(("mtp.", "vision.", "aligner.", "image_"))]
            if unexpected:
                raise RuntimeError(f"checkpoint tensors the module did not take: {unexpected[:5]}")
            del shard
            if fetched:
                os.remove(path)
    bank_re = re.compile(r"^layers\.(\d+)\.ffn\.(w[123])(_scale)?$")
    problems = sorted(k for k in (missing_all or set()) if not bank_re.match(k))
    for key in sorted(k for k in (missing_all or set()) if bank_re.match(k)):
        layer, proj, scale = bank_re.match(key).groups()
        seen = experts_seen.get((int(layer), proj, "scale" if scale else "weight"), set())
        if len(seen) != args.n_routed_experts:
            problems.append(f"{key}: {len(seen)} of {args.n_routed_experts} experts in the checkpoint")
    if problems:
        raise RuntimeError(f"module tensors the checkpoint did not provide: {problems[:5]}")
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        t0 = time.perf_counter()
        new, top = [], []
        # The exporter sets the expert routing per graph (grouped for the
        # prefill, batched for the decode step); the module's default is
        # grouped, which on a decode step dequantizes every expert of
        # every layer (384 x 40 here) for one token's six. Same math,
        # ~40x the work: nine hours instead of fifteen minutes.
        with torch.no_grad(), ds.bf16_default():
            ds.set_routing(lm, "grouped")
            logits = lm(torch.tensor([ids]), 0)[1]
            ds.set_routing(lm, "batched")
            for step in range(settings["new_tokens"]):
                if step < settings["logit_positions"]:
                    top.append(_top(logits[0], settings["top_k"]))
                nxt = int(logits[0].argmax())
                new.append(nxt)
                if nxt in eos_ids:
                    break
                logits = lm(torch.tensor([[nxt]]), len(ids) + step)[1]
        elapsed = time.perf_counter() - t0
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": "eos" if new and new[-1] in eos_ids else "length",
                    "top_logits": top,
                    "ms_per_token": round(1000 * elapsed / max(1, len(new)), 1)})
        _progress(prompt, new, 1000 * elapsed / max(1, len(new)))
    producer = {"kind": "deepseek-eager", **_versions("cpu"), "kernels": "aten"}
    return _document(model, revision, type("Config", (), {"model_type": ds.MODEL_TYPE})(), producer, settings, out)


def record_vxpu_engine(model, revision, prompts, settings, artifact, device="cpu",
                       compile_decode=False, telemetry=vxpu_pb2.TELEMETRY_NONE, keep=None, prepare=None, external_dir=None):
    """Candidate: the executor's engine in this process (no cluster).
    With ``telemetry=vxpu_pb2.TELEMETRY_SUMMARY``, each prompt carries its execution
    telemetry. With ``telemetry=vxpu_pb2.TELEMETRY_TRACE``, it also records ordered
    expert traces. keep, a list, receives the engine (for checks of its
    state afterwards)."""
    import torch
    from .engine import Engine, SamplingParams
    tok = _tokenizer(model, revision, prompts)
    engine = Engine(artifact, device=device, compile_decode=compile_decode,
                    external_dir=external_dir)
    if prepare is not None:  # a check's change to the loaded engine, before it generates
        prepare(engine)
    if keep is not None:
        keep.append(engine)
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        sid = engine.new_session()
        top = []

        def sink(position, logits, top=top):
            if position < settings["logit_positions"]:
                top.append(_top(logits, settings["top_k"]))

        new, stats = [], None
        for event in engine.generate(sid, ids, settings["new_tokens"],
                                     SamplingParams(do_sample=False),
                                     logits_sink=sink, telemetry=telemetry):
            new.extend(event.get("token_ids", []))
            if event.get("done"):
                stats = event
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": stats.get("finish_reason") if stats else None,
                    "top_logits": top,
                    "prefill_ms": stats.get("prefill_ms") if stats else None,
                    "ms_per_token": _r(stats.get("ms_per_token")) if stats else None,
                    **({"telemetry": stats.get("telemetry")} if telemetry not in (None, vxpu_pb2.TELEMETRY_NONE) and stats else {})})
        _progress(prompt, new, stats.get("ms_per_token") if stats else None)
    with open(os.path.join(artifact, "binding.json")) as f:
        binding = json.load(f)
    from .plan import record as plan_record
    producer = {"kind": "vxpu-engine" if not external_dir else "vxpu-engine (external_dir)",
                **_versions(device),
                "compile": compile_decode,
                "binding": {k: len(binding.get(k, [])) for k in
                            ("bound", "derived", "state", "stacked", "aliases")},
                "quantization": binding.get("quantization"),
                "plan": plan_record(engine.explain())}
    return _document(model, revision, engine.config, producer, settings, out)



def record_vxpu_router(model, revision, prompts, settings, artifact,
                       router=None, router_admin=None, telemetry=vxpu_pb2.TELEMETRY_NONE):
    """Candidate: an executor reached through a router (tokens only;
    the wire carries no logits). With ``telemetry=vxpu_pb2.TELEMETRY_SUMMARY``, each prompt
    carries its execution telemetry, through an internal Execute call on the router's admin port.
    With ``telemetry=vxpu_pb2.TELEMETRY_TRACE``, it also records ordered expert traces."""
    from .modeling import VxpuModelForCausalLM
    tok = _tokenizer(model, revision, prompts)
    admin_forward = None
    if telemetry not in (None, vxpu_pb2.TELEMETRY_NONE) and not router_admin and not os.environ.get("VXPU_ROUTER_ADMIN"):
        from .client import DEFAULT_ROUTER_TARGET, PortForward
        admin_forward = PortForward(DEFAULT_ROUTER_TARGET, remote_port=50052)
        try:
            router_admin = admin_forward.start()
        except RuntimeError:
            admin_forward = None
    lm = VxpuModelForCausalLM.from_pretrained(
        model, router=router, router_admin=router_admin,
        artifact_dir=artifact, revision=revision,
        progress=lambda *a: None)
    lm.collect_telemetry = telemetry
    out = []
    try:
        for prompt in _prompt_set(tok, prompts):
            ids = _encode(tok, prompt)
            lm.new_session()
            import torch
            result = lm.generate(torch.tensor([ids]),
                                 max_new_tokens=settings["new_tokens"],
                                 do_sample=False)
            new = result[0, len(ids):].tolist()
            stats = lm.last
            out.append({**prompt, "input_ids": ids, "generated_ids": new,
                        "finish": getattr(stats, "finish_reason", None),
                        "top_logits": [],
                        "prefill_ms": getattr(stats, "prefill_ms", None),
                        "ms_per_token": _r(getattr(stats, "ms_per_token", None)),
                        **({"telemetry": getattr(stats, "telemetry", None)} if telemetry not in (None, vxpu_pb2.TELEMETRY_NONE) else {})})
            _progress(prompt, new, getattr(stats, "ms_per_token", None))
    finally:
        lm.close()
        if admin_forward is not None:
            admin_forward.stop()
    producer = {"kind": "vxpu-router", **_versions("remote"),
                "router": lm.client.address}
    return _document(model, revision, lm.config, producer, settings, out)


def record_vllm(model, revision, prompts, settings, dtype="auto",
                max_model_len=None, enforce_eager=False):
    """Candidate: vLLM serving the same repo/revision, greedy, raw logits.

    vLLM is given the prompt as the same token ids the other producers
    use (the chat template is applied here, not by vLLM), and asked for
    the top-k *raw logits* of each generated position (``logprobs_mode=
    "raw_logits"``: the pre-softmax values before any processor), so the
    recording compares with the others at the logit level. Sampling
    defaults come from vLLM, not the model's generation_config, and
    everything that matters is set explicitly: temperature 0, no
    penalties. Expect larger logit differences than between two
    transformers runs: vLLM's fp8 path quantizes activations to fp8
    too (W8A8), where transformers and the vXPU graph dequantize the
    weights and multiply in bf16 (W8A16); the greedy tokens are the
    test, the logits say how close the two engines' views are.

    vLLM places the model itself (there is no device argument); the
    recording is labelled with the device and card it ran on.

    Runs with the ``vllm`` package only (plus transformers for the
    tokenizer): copy this file into a vLLM image and run it directly,
    ``python vectors_test.py record --producer vllm ...``.
    """
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = _tokenizer(model, revision, prompts)
    kwargs = {"model": model, "revision": revision, "dtype": dtype,
              "logprobs_mode": "raw_logits", "max_logprobs": settings["top_k"],
              "generation_config": "vllm", "enforce_eager": enforce_eager, "seed": 0}
    if max_model_len:
        kwargs["max_model_len"] = max_model_len
    llm = LLM(**kwargs)
    config = llm.llm_engine.model_config.hf_config
    params = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                            max_tokens=settings["new_tokens"],
                            logprobs=settings["top_k"], skip_special_tokens=False)
    out = []
    for prompt in _prompt_set(tok, prompts):
        ids = _encode(tok, prompt)
        t0 = time.perf_counter()
        result = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - t0
        completion = result.outputs[0]
        new = list(completion.token_ids)
        top = []
        for step in (completion.logprobs or [])[:settings["logit_positions"]]:
            # {token_id: Logprob(logprob=<raw logit here>, rank, ...)}; the
            # sampled token is included even when outside the top-k, so
            # keep the k best by value, like torch.topk does.
            pairs = sorted(((lp.logprob, i) for i, lp in step.items()), reverse=True)
            pairs = pairs[:settings["top_k"]]
            top.append({"ids": [i for _, i in pairs],
                        "values": [round(float(v), 5) for v, _ in pairs]})
        out.append({**prompt, "input_ids": ids, "generated_ids": new,
                    "finish": {"stop": "eos", "length": "length"}.get(
                        completion.finish_reason, completion.finish_reason),
                    "top_logits": top,
                    "ms_per_token": round(1000 * elapsed / max(1, len(new)), 1)})
        _progress(prompt, new, 1000 * elapsed / max(1, len(new)))
    mc = llm.llm_engine.model_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    producer = {"kind": "vllm", "vllm": vllm.__version__, **_versions(device),
                "dtype": str(mc.dtype), "quantization": mc.quantization,
                "enforce_eager": enforce_eager, "logprobs_mode": "raw_logits"}
    return _document(model, revision, config, producer, settings, out)


def _document(model, revision, config, producer, settings, prompts):
    return {"format": FORMAT,
            "model": {"repo": model, "revision": revision,
                      "model_type": getattr(config, "model_type", None)},
            "producer": producer, "settings": settings, "prompts": prompts}


def _scale(values):
    """The scale a relative tolerance applies to at one position: the
    largest logit magnitude in the recorded top-k, at least 1.0. Used
    for logit diffs and for the tie set alike, so one tolerance means
    one thing."""
    return max(1.0, max(abs(v) for v in values))


def tie_set(top, logit_tolerance):
    """The ids a greedy decoder could defensibly pick at a position: every
    token whose logit is within the tolerance of the top one, on the
    same scale ``compare`` uses for logit diffs."""
    if not top or not top["values"]:
        return set()
    best = top["values"][0]
    allowed = logit_tolerance * _scale(top["values"])
    return {i for i, v in zip(top["ids"], top["values"]) if best - v <= allowed}


def _first_flip(r, p, logit_tolerance):
    """Where the candidate's greedy tokens leave the reference's, and
    whether the reference's logits there excuse it. None when the
    streams are equal; else a dict with the position, the two tokens,
    and a kind: ``tie`` (the candidate's token is in the reference's
    tie set), ``outside`` (it is not: the reference had a clear winner
    the candidate missed), ``unjudged`` (no reference logits at that
    position to say), or ``length`` (one side stopped where the other
    went on)."""
    pos = next((i for i, (x, y) in enumerate(zip(r["generated_ids"], p["generated_ids"]))
                if x != y), None)
    if pos is None:
        if len(r["generated_ids"]) == len(p["generated_ids"]):
            return None
        pos = min(len(r["generated_ids"]), len(p["generated_ids"]))  # one stopped early
    flip = {"position": pos,
            "ref_token": r["generated_ids"][pos] if pos < len(r["generated_ids"]) else None,
            "token": p["generated_ids"][pos] if pos < len(p["generated_ids"]) else None}
    if flip["token"] is None or flip["ref_token"] is None:
        flip["kind"] = "length"  # one side stopped (eos) where the other went on
        return flip
    tops = r.get("top_logits") or []
    if pos >= len(tops) or not tops[pos] or not tops[pos]["values"]:
        flip["kind"] = "unjudged"  # no reference logits recorded there
        return flip
    top = tops[pos]
    values = dict(zip(top["ids"], top["values"]))
    best = top["values"][0]
    flip["gap"] = round(best - values[flip["token"]], 6) if flip["token"] in values else None
    flip["kind"] = "tie" if flip["token"] in tie_set(top, logit_tolerance) else "outside"
    return flip


def compare(reference, candidate, logit_tolerance=0.025, strict_tokens=False):
    """Per prompt: do the greedy tokens hold up, and how far apart are the
    top-k logits. Returns (rows, ok).

    Greedy tokens are judged against the reference's *tie set*: at the
    first position where the candidate's token differs, the reference's
    own logits say whether that token was within the tolerance of its
    top choice. A flip inside the tie set is a coin toss two correct
    engines can call differently (eager transformers on a CPU and on an
    L4 do, at a gap of one bf16 ulp) and passes; a flip outside it is a
    divergence and fails; a flip at a position with no reference logits
    cannot be judged and fails, which is the reason to record logits
    for every generated position. ``strict_tokens=True`` (the CI
    selftest: the same computation on two paths) also fails a flip
    inside the tie set unless the logit error measured at that position
    covers the gap the flip had to close: two paths that differ by
    rounding may call an exact tie differently, but a flip across a gap
    larger than their disagreement is a bug, not rounding.

    ``logit_tolerance`` is relative to the largest logit at that
    position (at least 1.0): softmax is shift-invariant, so what matters
    is error against the scale of the distribution, and two bf16
    computations of the same model legitimately differ by a few units
    in the last place of that scale (a logit of 30 carries 0.25 per ulp
    in bf16). 2.5% covers that; a real divergence is tens of percent.

    The two top-k sets need not hold the same ids. An id in one set and
    not the other still has a known bound on the other side: its logit
    there is no larger than that side's k-th value, so the diff is at
    least the gap down to it. That gap is what is charged, in both
    directions; a diff of exactly the k-th-largest cut is the smallest
    the truth could be, never an acquittal.

    Logits are compared only up to and including the first position
    where the generated tokens differ: a later position is conditioned
    on a different prefix on each side, and its logits are two
    different questions, not two answers to one.
    """
    rows, ok = [], True
    ref = {p["name"]: p for p in reference["prompts"]}
    for p in candidate["prompts"]:
        r = ref.get(p["name"])
        if r is None:
            rows.append({"prompt": p["name"], "status": "no reference"})
            continue
        same_prompt = r["input_ids"] == p["input_ids"]
        flip = _first_flip(r, p, logit_tolerance)
        tokens_equal = flip is None
        max_diff, max_rel, at_flip = None, None, None
        comparable = (len(r["generated_ids"]) if flip is None
                      else min(flip["position"] + 1, len(r["generated_ids"])))
        if r.get("top_logits") and p.get("top_logits"):
            diffs, rels = [], []
            for pos, (a, b) in enumerate(list(zip(r["top_logits"], p["top_logits"]))[:comparable]):
                av, bv = dict(zip(a["ids"], a["values"])), dict(zip(b["ids"], b["values"]))
                a_floor, b_floor = min(a["values"]), min(b["values"])
                scale = _scale(a["values"])
                here = []
                for i, v in av.items():
                    here.append(abs(v - bv[i]) if i in bv else max(0.0, v - b_floor))
                for i, v in bv.items():
                    if i not in av:
                        here.append(max(0.0, v - a_floor))
                diffs.extend(here)
                rels.extend(d / scale for d in here)
                if flip is not None and pos == flip["position"] and here:
                    at_flip = max(here)
            max_diff = max(diffs) if diffs else None
            max_rel = max(rels) if rels else None
        tokens_ok = tokens_equal or flip["kind"] == "tie"
        if strict_tokens and not tokens_equal and tokens_ok:
            # the measured error at that position must cover the gap
            flip["explained"] = (flip.get("gap") is not None and at_flip is not None
                                 and flip["gap"] <= at_flip + 1e-6)
            tokens_ok = flip["explained"]
        row_ok = same_prompt and tokens_ok and (max_rel is None or max_rel <= logit_tolerance)
        ok &= row_ok
        rows.append({"prompt": p["name"], "status": "ok" if row_ok else "DIFF",
                     "same_prompt": same_prompt, "tokens_equal": tokens_equal,
                     "tokens_ok": tokens_ok, "flip": flip,
                     "compared": f"{comparable}/{len(r['generated_ids'])}",
                     "max_logit_diff": None if max_diff is None else round(max_diff, 4),
                     "max_logit_rel": None if max_rel is None else round(max_rel, 4),
                     "ref_ms_per_token": r.get("ms_per_token"),
                     "ms_per_token": p.get("ms_per_token")})
    return rows, ok


def table(reference, candidate, rows):
    head = (f"{candidate['model']['repo']} @ {candidate['model']['revision']}: "
            f"{reference['producer']['kind']} vs {candidate['producer']['kind']}")
    lines = [head, "| prompt | tokens (compared) | top-k logits max diff (abs / rel) | ms/token (ref -> candidate) |",
             "|---|---|---|---|"]
    for r in rows:
        if r["status"] == "no reference":
            lines.append(f"| {r['prompt']} | no reference | | |")
            continue
        logit = ("n/a" if r["max_logit_diff"] is None
                 else f"{r['max_logit_diff']} / {100 * r['max_logit_rel']:.1f}%")
        lines.append(f"| {r['prompt']} | {_describe_tokens(r)} ({r['compared']}) | "
                     f"{logit} | {r['ref_ms_per_token']} -> {r['ms_per_token']} |")
    return "\n".join(lines)


def _describe_tokens(row):
    f = row["flip"]
    if f is None:
        return "equal"
    where = f"@{f['position']}"
    if f["kind"] == "tie":
        return f"tie flip {where}, gap {f['gap']}"
    if f["kind"] == "outside":
        return f"DIFFER {where}, gap {f['gap'] if f['gap'] is not None else 'beyond top-k'}"
    if f["kind"] == "length":
        return f"DIFFER {where}, {'candidate' if f['token'] is None else 'reference'} stopped"
    return f"DIFFER {where}, no logits to judge"


def _load(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise SystemExit(f"{path}: not a {FORMAT} file")
    return doc


def _settings(args):
    # Logits for every generated position unless asked for fewer: a flip
    # at a position without them cannot be judged.
    positions = args.logit_positions if args.logit_positions is not None else args.new_tokens
    return {"new_tokens": args.new_tokens, "top_k": args.top_k,
            "logit_positions": positions, "greedy": True}


def cmd_record(args):
    settings = _settings(args)
    telemetry = (vxpu_pb2.Telemetry.Value("TELEMETRY_" + args.telemetry.upper())
                 if args.telemetry else vxpu_pb2.TELEMETRY_NONE)
    if args.producer == "transformers":
        doc = record_transformers(args.model, args.revision, args.prompts, settings,
                                  device=args.device, dtype=args.dtype,
                                  dequantize=args.dequantize)
    elif args.producer == "vxpu-engine":
        doc = record_vxpu_engine(args.model, args.revision, args.prompts, settings,
                                 args.artifact, device=args.device,
                                 compile_decode=args.compile, telemetry=telemetry)
    elif args.producer == "deepseek-eager":
        doc = record_deepseek_eager(args.model, args.revision, args.prompts, settings)
    elif args.producer == "vllm":
        doc = record_vllm(args.model, args.revision, args.prompts, settings,
                          dtype=args.dtype,
                          max_model_len=args.max_model_len,
                          enforce_eager=args.enforce_eager)
    else:
        doc = record_vxpu_router(args.model, args.revision, args.prompts, settings,
                                 args.artifact, router=args.router,
                                 router_admin=args.router_admin,
                                 telemetry=telemetry)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"{args.out}: {len(doc['prompts'])} prompts by {doc['producer']['kind']}")


def cmd_compare(args):
    reference, candidate = _load(args.reference), _load(args.candidate)
    rows, ok = compare(reference, candidate, args.logit_tolerance)
    print(table(reference, candidate, rows))
    if not ok:
        raise SystemExit("vectors differ")


def _check_compare():
    """compare's rules on synthetic vectors: an exact tie the candidate
    calls the other way passes strict mode when the measured logit
    error covers the gap, and fails it when the gap is wider than the
    paths' disagreement."""
    def doc(kind, ids, tops):
        return {"producer": {"kind": kind},
                "prompts": [{"name": "p", "input_ids": [1, 2], "generated_ids": ids,
                             "top_logits": [{"ids": list(t), "values": list(v)} for t, v in tops]}]}
    ref = doc("a", [7, 3], [((7, 3, 5), (10.0, 10.0, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    # same logits to within 0.001 at the tie, other token picked: rounding
    got = doc("b", [3, 3], [((3, 7, 5), (10.0005, 10.0, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    _, ok = compare(ref, got, strict_tokens=True)
    assert ok, "a flip at an exact tie within the measured error must pass strict mode"
    # the gap (0.2) is wider than any logit disagreement (0.001): not rounding
    ref2 = doc("a", [7, 3], [((7, 3, 5), (10.0, 9.8, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    got2 = doc("b", [3, 3], [((7, 3, 5), (10.001, 9.8, 8.0)), ((3, 7, 5), (9.0, 8.0, 7.0))])
    rows, ok = compare(ref2, got2, strict_tokens=True)
    assert not ok and rows[0]["flip"]["kind"] == "tie" and rows[0]["flip"]["explained"] is False
    _, ok = compare(ref2, got2)  # the cluster tier: within the tie tolerance, passes
    assert ok


def _check_compile_plan():
    """Verify configure_inductor and Plan's compiler choices for prologue_fusion."""
    from . import engine, plan

    # CPU: prologue_fusion is False, marked CUDA only
    p_cpu = plan.Plan("cpu", compile_decode=False)
    assert p_cpu["prologue_fusion"] is False
    assert p_cpu.choices["prologue_fusion"].source == "CUDA only"

    # CUDA without compile_decode: requires compile_decode
    p_cuda_no_compile = plan.Plan("cuda", compile_decode=False)
    assert p_cuda_no_compile["prologue_fusion"] is False
    assert p_cuda_no_compile.choices["prologue_fusion"].source == "requires compile_decode"

    # CUDA with compile_decode: True by default
    p_cuda_compile = plan.Plan("cuda", compile_decode=True)
    assert p_cuda_compile["prologue_fusion"] is True
    assert p_cuda_compile.choices["prologue_fusion"].source == "default"

    # Recorded in explain choices
    explained = p_cuda_compile.explain()
    assert "prologue_fusion" in explained["choices"]
    assert explained["choices"]["prologue_fusion"]["value"] is True

    # configure_inductor sets torch._inductor.config.prologue_fusion when enabled by the plan
    try:
        import torch._inductor.config as inductor_config
        orig = getattr(inductor_config, "prologue_fusion", None)
        try:
            inductor_config.prologue_fusion = False
            engine.configure_inductor(p_cpu)
            assert inductor_config.prologue_fusion is False  # untouched on CPU
            engine.configure_inductor(p_cuda_compile)
            assert inductor_config.prologue_fusion is True
            engine.configure_inductor(p_cpu)
            assert inductor_config.prologue_fusion is True  # left untouched
        finally:
            if orig is not None:
                inductor_config.prologue_fusion = orig
    except ImportError:
        pass


def _check_expert_sim():
    """expert_sim's policies on synthetic traffic whose hot set moves: four
    requests use experts 0-3, then eight use 4-7, a budget of four. The
    static placement (from the first four) serves none of the later ones;
    holding the last request's experts (decay 0) misses only the first;
    cumulative LFU (decay 1) switches once the new set's counts pass the
    old; the prefill (here the same experts as the decode) and the oracle
    serve them all."""
    from . import expert_sim
    import collections
    def req(experts):
        c = collections.Counter({(0, e): 10 for e in experts})
        return c, 10, c
    reqs = [req(range(4))] * 4 + [req(range(4, 8))] * 8
    def share(policy, **kw):
        got = expert_sim.simulate(reqs, 4, 4, policy, **kw)
        return sum(s for s, _ in got) / len(got), sum(m for _, m in got)
    assert share("static") == (0.0, 0), share("static")
    assert share("adaptive", decay=0.0) == (7 / 8, 4), share("adaptive", decay=0.0)
    assert share("adaptive", decay=1.0) == (3 / 8, 4), share("adaptive", decay=1.0)
    assert share("prefill", prefill_weight=0.75) == (1.0, 4), share("prefill", prefill_weight=0.75)
    assert share("oracle") == (1.0, 4), share("oracle")


def _check_continuation(artifact, ids, new_tokens=4):
    """A session extended by a later turn must see what a fresh session
    given the whole transcript sees. For a model whose prefill graph
    starts at position 0 the engine re-prefills from the start, so the
    two are bit-identical; this is the multi-turn case of chat."""
    import torch
    from .engine import Engine, SamplingParams
    engine = Engine(artifact)

    def run(sid, ids):
        first = {}

        def sink(position, logits):
            if position == 0:
                first["logits"] = logits.detach().float().clone()
        gen = []
        for ev in engine.generate(sid, ids, new_tokens, SamplingParams(do_sample=False), logits_sink=sink):
            gen.extend(ev.get("token_ids", []))
        return gen, first["logits"]

    sid = engine.new_session()
    gen1, _ = run(sid, ids)
    turn2 = ids + gen1 + ids[:7]  # the transcript so far plus a new turn
    gen2, extended = run(sid, turn2)
    gen3, fresh = run(engine.new_session(), turn2)
    if gen2 != gen3 or not torch.equal(extended, fresh):
        raise SystemExit(f"a continued session differs from a fresh one: {gen2} vs {gen3}, "
                         f"max logit diff {(extended - fresh).abs().max().item():.4g}")
    print(f"  continuation ok: {len(turn2)}-token second turn matches a fresh session")


def _check_telemetry(model, prompts, settings, artifact, plain, workdir):
    """Check the engine's execution telemetry end to end. Recording it
    changes no token. Each prompt carries a vxpu-telemetry/1 record
    with its prefill token count and its decode expert counts. With
    ``telemetry=vxpu_pb2.TELEMETRY_TRACE``, the record also lists each decode step's
    expert selections, which add up to the decode counts. Plain telemetry
    has no trace. expert_stats can read the file."""
    import collections
    from . import expert_stats
    from .engine import SamplingParams
    keep = []
    doc_plain = record_vxpu_engine(model, "main", prompts, settings, artifact, telemetry=vxpu_pb2.TELEMETRY_SUMMARY, keep=keep)
    doc_trace = record_vxpu_engine(model, "main", prompts, settings, artifact, telemetry=vxpu_pb2.TELEMETRY_TRACE)
    path = os.path.join(workdir, "tiny-deepseek-v41-telemetry.json")
    with open(path, "w") as f:
        json.dump(doc_trace, f, indent=1)
    problems = []

    # Check that invalid telemetry modes are rejected.
    engine = keep[0]
    sid = engine.new_session()
    ids = list(plain["prompts"][0]["input_ids"])
    for bad_mode in ("invalid", "Trace", "summary", "trace", None, -1, 99, True, False):
        try:
            for _ in engine.generate(sid, ids, 1, SamplingParams(do_sample=False), telemetry=bad_mode):
                pass
            problems.append(f"engine.generate: accepted invalid telemetry={bad_mode!r}")
        except ValueError:
            pass

    # Check TELEMETRY_SUMMARY adds nothing to expert_trace
    for a, b in zip(plain["prompts"], doc_plain["prompts"]):
        tel = b.get("telemetry") or {}
        if a["generated_ids"] != b["generated_ids"]:
            problems.append(f"{b['name']}: tokens changed with telemetry on")
        if "expert_trace" in tel:
            problems.append(f"{b['name']}: TELEMETRY_SUMMARY should not include expert_trace")

    # Check TELEMETRY_TRACE
    for a, b in zip(plain["prompts"], doc_trace["prompts"]):
        tel = b.get("telemetry") or {}
        if a["generated_ids"] != b["generated_ids"]:
            problems.append(f"{b['name']}: tokens changed with TELEMETRY_TRACE")
        if tel.get("format") != "vxpu-telemetry/1":
            problems.append(f"{b['name']}: no telemetry record")
            continue
        if tel.get("prefill_tokens") != len(b["input_ids"]):  # a fresh session prefills the whole prompt
            problems.append(f"{b['name']}: prefill_tokens {tel.get('prefill_tokens')} for {len(b['input_ids'])} prompt tokens")
        steps = tel["decode_steps"]
        decode_counts = tel.get("experts", {}).get("decode", {})
        totals = {sum(c.values()) for c in decode_counts.values()}
        if not steps or len(totals) != 1 or next(iter(totals)) % steps:
            problems.append(f"{b['name']}: decode expert counts {sorted(totals)} for {steps} decode steps")
        trace = tel.get("expert_trace")
        if not trace:
            problems.append(f"{b['name']}: no expert_trace in TELEMETRY_TRACE")
        else:
            for l, steps_list in trace.items():
                if len(steps_list) != steps:
                    problems.append(f"{b['name']}: layer {l} trace has {len(steps_list)} steps for {steps} decode steps")
                trace_counts = collections.Counter(e for step in steps_list for e in step)
                expected_counts = {int(k): v for k, v in decode_counts.get(l, {}).items()}
                if dict(trace_counts) != expected_counts:
                    problems.append(f"{b['name']}: layer {l} trace counts do not match decode expert counts")
    import math
    from . import expert_sim
    stats_out, text = expert_stats.report([path], artifact=artifact)
    print(text)
    per_expert, experts = expert_sim.resolve_expert_size(artifact, None)
    sim_out, sim_text = expert_sim.report_trace(
        [path], per_expert=per_expert, experts=experts, train=0.0
    )
    print(sim_text)

    # Check that the oracle row matches expert_stats across layer expert fractions (#447).
    docs = expert_stats.load([path])
    layer_steps, _ = expert_sim.traces(docs)
    for f_str, expected_rate in stats_out["hit_rate_overall"].items():
        k = math.ceil(float(f_str) * experts)
        hits, misses, total = expert_sim.simulate_trace_policy(layer_steps, k, train_steps=0, policy="oracle")
        actual_rate = round(hits / total, 4) if total else 0.0
        if actual_rate != expected_rate:
            problems.append(f"expert_sim: oracle hit rate for {f_str} is {actual_rate}, expected {expected_rate}")

    # Check that LRU at 100% capacity misses only each expert's first selection (#447).
    distinct = len({(int(l), e) for l, steps in layer_steps.items() for step in steps for e in step})
    hits, lru_misses, _ = expert_sim.simulate_trace_policy(layer_steps, experts, train_steps=0, policy="lru")
    if lru_misses != distinct:
        problems.append(
            f"expert_sim: LRU at 100% capacity has {lru_misses} misses, expected {distinct} "
            f"(one per distinct expert)"
        )

    for p in problems:
        print("telemetry check:", p)
    return not problems


def _aoti_complex_proxy_fallbacks(cache_dir, prior=None):
    """Any proxy-executor fallback on complex ops in the cached AOTInductor packages."""
    import zipfile
    found = collections.defaultdict(list)
    if not os.path.exists(cache_dir):
        return found
    prior = prior or set()
    for fname in os.listdir(cache_dir):
        if not fname.endswith(".pt2") or fname.endswith(".tmp.pt2"):
            continue
        if prior and fname in prior:
            continue
        path = os.path.join(cache_dir, fname)
        try:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.endswith(".wrapper.json"):
                        data = json.loads(z.read(name).decode("utf-8"))
                        for entry in data.get("nodes", []):
                            target = entry.get("node", {}).get("target", "")
                            if any(k in target for k in ("conj", "complex", "clone")):
                                found[fname].append(target)
        except zipfile.BadZipFile:
            continue
    return dict(found)


def _check_rotary():
    """Verify vxpu::rotary: Triton and reference implementations with
    unit-modulus RoPE angles at DeepSeek's shape ([1, 512, 64, 64] and
    [1, 512, 64]), confirming the rounding numerics class bounded by 1 bf16 ulp
    plus 8 fp32 ulps of the operand scale |x_re·c| + |x_im·s| across 3D and 4D
    shapes, with and without inverse rotation, and refusing mismatched angles."""
    import torch
    from . import ops
    from . import triton_ops

    def operand_bound(x, f, y_ref):
        c, s = f.real, f.imag
        if x.ndim == 3:
            c = c.view(1, x.size(1), -1)
            s = s.view(1, x.size(1), -1)
        else:
            c = c.view(1, x.size(1), 1, -1)
            s = s.view(1, x.size(1), 1, -1)
        xf = x.float().unflatten(-1, (-1, 2))
        x_re, x_im = xf[..., 0], xf[..., 1]
        op_scale = torch.stack([x_re.abs() * c.abs() + x_im.abs() * s.abs(),
                                x_re.abs() * s.abs() + x_im.abs() * c.abs()],
                               dim=-1).flatten(-2)
        exp_bf16 = torch.frexp(y_ref.float().abs())[1]
        ulp_bf16 = torch.ldexp(torch.ones_like(op_scale), torch.clamp(exp_bf16 - 8, min=-133))
        exp_fp32 = torch.frexp(op_scale)[1]
        ulp_fp32 = torch.ldexp(torch.ones_like(op_scale), torch.clamp(exp_fp32 - 24, min=-149))
        return ulp_bf16 + 8 * ulp_fp32

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for dev in devices:
        torch.manual_seed(42)
        dim = 64
        base = 10000.0
        seqlen = 512
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=dev, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(seqlen, device=dev, dtype=torch.float32), freqs)
        f = torch.polar(torch.ones_like(angles), angles)

        # 3D: [b, s, d] (e.g. kv cache projection)
        x3 = torch.randn(1, seqlen, dim, dtype=torch.bfloat16, device=dev)
        # 4D: [b, s, h, d] (DeepSeek-V4.1: 64 query heads, rope_head_dim=64)
        x4 = torch.randn(1, seqlen, 64, dim, dtype=torch.bfloat16, device=dev)

        for x in (x3, x4):
            # Mismatched rank and angle shapes must be refused
            if dev == "cuda" and triton_ops.available():
                for bad_x in (torch.randn(seqlen, dim, device=dev), torch.randn(1, seqlen, 2, 2, dim, device=dev)):
                    try:
                        triton_ops.rotary(bad_x, f.real, f.imag)
                        raise AssertionError(f"triton_ops.rotary accepted unsupported rank {bad_x.ndim}")
                    except ValueError:
                        pass
                bad_cos = torch.randn(seqlen + 1, dim // 2, device=dev)
                bad_sin = torch.randn(seqlen + 1, dim // 2, device=dev)
                try:
                    triton_ops.rotary(x, bad_cos, bad_sin)
                    raise AssertionError("triton_ops.rotary accepted mismatched angle shape")
                except ValueError:
                    pass

            for inv in (False, True):
                y_ref = ops._rotary_reference(x, f, inv)
                y_op = torch.ops.vxpu.rotary(x, f, inv)
                if dev == "cuda":
                    # On CUDA: compare Triton kernel against reference and lowered
                    diff = (y_op.float() - y_ref.float()).abs()
                    bound = operand_bound(x, f, y_ref)
                    exceeds = diff > bound
                    if exceeds.any():
                        bad = exceeds.sum().item()
                        max_ratio = (diff / bound).max().item()
                        raise AssertionError(
                            f"vxpu::rotary on {dev} (inv={inv}) diverges from reference beyond operand bound "
                            f"(1 bf16 ulp + 8 fp32 ulps of |x_re·c| + |x_im·s|): "
                            f"{bad}/{x.numel()} elements exceed bound, max ratio={max_ratio:.2f}")
                    # Lowered call runs the exact same Triton kernel, so it must be bit-identical
                    if triton_ops.available():
                        y_lowered = triton_ops.lowered.rotary(x, f.real, -f.imag if inv else f.imag)
                        if not torch.equal(y_op.view(torch.int16), y_lowered.view(torch.int16)):
                            raise AssertionError(f"vxpu::rotary eager vs lowered mismatch on {dev} (inv={inv})")
                else:
                    # On CPU: reference is selected and bit-identical
                    if not torch.equal(y_op.view(torch.int16), y_ref.view(torch.int16)):
                        raise AssertionError(f"vxpu::rotary custom op dispatch mismatch on {dev} (inv={inv})")


def _check_lowering(model, prompts, settings, artifact, workdir):
    """On a GPU: the compiled, split decode with the plan lowered into its
    segments (lower.py), and with those segments as AOTInductor packages
    in the engine's own CUDA graphs (aoti.py), each against the plain
    compiled decode, token-exact. Lowered calls run the same kernels, so
    the vectors must agree. Each variant must really have run as such
    (the plan falls back silently when a requirement is missing). Skipped
    without CUDA or Triton (nothing is lowerable)."""
    import torch
    if not torch.cuda.is_available():
        print("lowering check skipped: no CUDA")
        return True
    # every switch the variants depend on is pinned, so none is inherited
    # from the environment the selftest runs in
    aoti_cache = os.path.join(workdir, "aoti-cache")
    prior_packages = set(os.listdir(aoti_cache)) if os.path.exists(aoti_cache) else set()
    split = {"VXPU_SPLIT_DECODE": "1", "VXPU_SPLIT_COMPILE": "1"}
    variants = {"compiled": {**split, "VXPU_SEGMENT_RUNTIME": "jit", "VXPU_LOWER_DECODE": "0"},
                "lowered": {**split, "VXPU_SEGMENT_RUNTIME": "jit", "VXPU_LOWER_DECODE": "1"},
                "aoti": {**split, "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_LOWER_DECODE": "1",
                         "VXPU_AOTI_CACHE_DIR": aoti_cache}}
    intended = {"compiled": (False, "jit"), "lowered": (True, "jit"), "aoti": (True, "aoti")}  # lower_decode, runtime
    runs = {}
    for label, env in variants.items():
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            runs[label] = record_vxpu_engine(model, "main", prompts, settings, artifact, device="cuda",
                                             compile_decode=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k)
                else:
                    os.environ[k] = v
    from .lower import REPORT_PREFIX
    if not runs["lowered"]["producer"]["plan"]["choices"].get("triton"):
        print("lowering check skipped: no Triton (nothing is lowerable)")
        return True
    ok = True
    for label, (lowered, runtime) in intended.items():  # each run must have run as intended
        choices = runs[label]["producer"]["plan"]["choices"]
        got = (bool(choices.get("lower_decode")), choices.get("segment_runtime"))
        if got != (lowered, runtime):
            print(f"lowering check: the {label} run ran lower_decode={got[0]}, segment_runtime={got[1]}")
            ok = False
    if not ok:
        return False
    for label in ("lowered", "aoti"):
        plan = runs[label]["producer"]["plan"]
        report = [n for n in plan.get("notes", []) if n.startswith(REPORT_PREFIX)]
        if not report or "lowered " not in report[0][len(REPORT_PREFIX):]:
            print(f"lowering check: the {label} run lowered nothing (report={report})")
            ok = False
            continue
        if label == "aoti":
            fallbacks = _aoti_complex_proxy_fallbacks(aoti_cache, prior=prior_packages)
            if fallbacks:
                print(f"lowering check: AOTI packages have complex proxy fallbacks: {fallbacks}")
                ok = False
                continue
        rows, same = compare(runs["compiled"], runs[label], strict_tokens=True)
        print(f"{label} decode against compiled, cuda:")
        print(table(runs["compiled"], runs[label], rows))
        ok = ok and same
    return ok


def _check_host_reads(workdir):
    """The n-gram tables' reads (#364): the CPU kernels' gather_rows gives
    index_select's rows and refuses indices out of range or not on the
    CPU (it reads them on the host); advise_random takes on a
    memory-mapped tensor."""
    import mmap
    import torch
    from . import kernels
    from .rehydrate import advise_random, file_backed
    k = kernels.load()
    if k is None or not hasattr(k, "gather_rows"):
        print("host reads check skipped: no CPU kernels")
        return
    table = torch.arange(64 * 24, dtype=torch.int16).view(64, 24)
    idx = torch.tensor([[3, 63, 0], [17, 17, 5]])
    if not torch.equal(k.gather_rows(table, idx), table[idx]):
        raise SystemExit("gather_rows: rows differ from index_select's")
    for bad in (torch.tensor([64]), torch.tensor([-1])):
        try:
            k.gather_rows(table, bad)
        except RuntimeError as e:
            if "outside" not in str(e):
                raise
        else:
            raise SystemExit(f"gather_rows accepted index {bad.item()} for a 64-row table")
    device = "cuda" if torch.cuda.is_available() else "meta"
    try:
        k.gather_rows(table, idx.to(device))
    except RuntimeError as e:
        if "indices on the CPU" not in str(e):
            raise
    else:
        raise SystemExit(f"gather_rows accepted indices on {device}")
    path = os.path.join(workdir, "advise-random.bin")
    mapped, *_ = file_backed(path, 1 << 16)
    if getattr(mmap, "MADV_RANDOM", None) is not None and not advise_random(mapped):
        raise SystemExit("advise_random: madvise(MADV_RANDOM) was refused on a memory-mapped tensor")

    # The n-gram tables read straight from checkpoint files (#452): gather_table_rows
    # with io_uring and with pread fallback against gather_rows
    tbl_path = os.path.join(workdir, "gather-rows-file.bin")
    w_offset = 128
    rows = table.shape[0]
    scale_table = table[:, :table.shape[1] // 2].contiguous()
    w_row_bytes = table.shape[1] * table.element_size()
    s_row_bytes = scale_table.shape[1] * scale_table.element_size()
    s_offset = w_offset + rows * w_row_bytes + 64
    with open(tbl_path, "wb") as f:
        f.write(b"\0" * w_offset)
        f.write(table.numpy().tobytes())
        f.write(b"\0" * (s_offset - (w_offset + rows * w_row_bytes)))
        f.write(scale_table.numpy().tobytes())

    fd = os.open(tbl_path, os.O_RDONLY)
    try:
        g = torch.Generator().manual_seed(42)
        rand_idx = torch.randint(0, rows, (4, 6), generator=g)
        want_w = k.gather_rows(table, rand_idx)
        want_s = k.gather_rows(scale_table, rand_idx)

        # use_uring=False: pread fallback
        w_got, s_got = k.gather_table_rows(fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, rand_idx, use_uring=False)
        if w_got.numpy().tobytes() != want_w.numpy().tobytes() or s_got.numpy().tobytes() != want_s.numpy().tobytes():
            raise SystemExit("gather_table_rows (pread): rows differ from gather_rows")

        # use_uring=True: io_uring when available, pread fallback otherwise
        w_got_u, s_got_u = k.gather_table_rows(fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, rand_idx, use_uring=True)
        if w_got_u.numpy().tobytes() != want_w.numpy().tobytes() or s_got_u.numpy().tobytes() != want_s.numpy().tobytes():
            raise SystemExit("gather_table_rows (use_uring=True): rows differ from gather_rows")

        if k.io_uring_ready():
            print("host reads: gather_table_rows io_uring matches gather_rows")
        else:
            print("host reads: gather_table_rows io_uring unavailable, fallback to pread verified")

        for bad in (torch.tensor([64]), torch.tensor([-1])):
            try:
                k.gather_table_rows(fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, bad)
            except RuntimeError as e:
                if "outside" not in str(e):
                    raise
            else:
                raise SystemExit(f"gather_table_rows accepted index {bad.item()} for a {rows}-row table")
        try:
            k.gather_table_rows(fd, w_offset, w_row_bytes, s_offset, s_row_bytes, rows, rand_idx.to(device))
        except RuntimeError as e:
            if "indices on the CPU" not in str(e):
                raise
        else:
            raise SystemExit(f"gather_table_rows accepted indices on {device}")
    finally:
        os.close(fd)

    print("host reads: gather_rows matches index_select and refuses bad or device indices; MADV_RANDOM taken")


def _check_local_blobs_copy(workdir):
    """Copying into the local weight tier (localblobs.py) through a fake
    gcloud that copies every file but one and then fails, as gcloud does:
    the complete files are kept, the next copy fetches only the missing
    one, and staging left by a process that no longer runs is removed."""
    import subprocess
    from . import localblobs
    root = os.path.join(workdir, "blob-copy")
    bucket_dir, blob_dir, bindir = (os.path.join(root, d) for d in ("bucket/_blobs", "blobs", "bin"))
    for d in (bucket_dir, blob_dir, bindir):
        os.makedirs(d, exist_ok=True)
    files = {}
    for k in range(3):
        sha = f"{k:064x}"
        with open(os.path.join(bucket_dir, sha), "wb") as f:
            f.write(bytes([k]) * (1000 + k))
        files[sha] = {"size": 1000 + k, "name": f"part{k}", "source": "unused"}
    fake = os.path.join(bindir, "gcloud")
    with open(fake, "w") as f:
        f.write(f"""#!{sys.executable}
import os, shutil, sys
dest, skip, failed = sys.argv[-1], os.environ.get("FAKE_GCLOUD_SKIP"), False
for url in sys.stdin.read().split():
    sha = url.rsplit("/", 1)[1]
    if sha == skip:
        failed = True
        continue
    shutil.copyfile(os.path.join({os.path.dirname(bucket_dir)!r}, "_blobs", sha), os.path.join(dest, sha))
sys.exit(1 if failed else 0)
""")
    os.chmod(fake, 0o755)
    gone = subprocess.Popen(["true"])
    gone.wait()
    stale = os.path.join(blob_dir, ".incoming", f"{gone.pid}-stale")
    os.makedirs(stale)
    skip = next(iter(files))
    env = {"PATH": bindir + os.pathsep + os.environ.get("PATH", ""), "FAKE_GCLOUD_SKIP": skip}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        try:
            localblobs.localize(files, blob_dir, bucket="gs://fake", log=lambda *_: None)
        except RuntimeError as e:
            if "2 of 3 files copied and kept" not in str(e):
                raise
        else:
            raise SystemExit("local blobs: a failed gcloud copy was not reported")
        kept = sorted(n for n in os.listdir(blob_dir) if not n.startswith("."))
        if kept != sorted(set(files) - {skip}) or os.listdir(os.path.join(blob_dir, ".incoming")):
            raise SystemExit(f"local blobs: after a partial copy, kept {kept}, staging "
                             f"{os.listdir(os.path.join(blob_dir, '.incoming'))}")
        os.environ.pop("FAKE_GCLOUD_SKIP")
        _, summary = localblobs.localize(files, blob_dir, bucket="gs://fake", log=lambda *_: None)
        if summary["copied"] != 1 or summary["local"] != 3:
            raise SystemExit(f"local blobs: the retry should copy only the missing file: {summary}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("local blobs copy: a partial copy keeps its complete files, the retry copies the rest, stale staging swept")


def _check_local_blobs(model, prompts, settings, artifact, plain, workdir):
    """The executor machine's local weight tier (localblobs.py): with the
    checkpoint's files in a blob directory, named by sha256, every source
    points there and the engine's tokens are unchanged."""
    import shutil
    from . import localblobs
    with open(os.path.join(artifact, "manifest.json")) as f:
        files = json.load(f)["files"]
    blob_dir = os.path.join(workdir, "blobs")
    os.makedirs(blob_dir, exist_ok=True)
    for sha, entry in files.items():
        shutil.copyfile(entry["source"][len("file://"):], os.path.join(blob_dir, sha))
    local, summary = localblobs.localize(files, blob_dir)
    if summary["local"] != len(files) or any(not e["source"].startswith("file://" + os.path.abspath(blob_dir))
                                             for e in local.values()):
        print(f"local blobs check: sources not localized: {summary}")
        return False
    saved = os.environ.get("VXPU_BLOB_DIR")
    os.environ["VXPU_BLOB_DIR"] = blob_dir
    try:
        got = record_vxpu_engine(model, "main", prompts, settings, artifact)
    finally:
        if saved is None:
            os.environ.pop("VXPU_BLOB_DIR")
        else:
            os.environ["VXPU_BLOB_DIR"] = saved
    if [p["generated_ids"] for p in got["prompts"]] != [p["generated_ids"] for p in plain["prompts"]]:
        print("local blobs check: tokens changed when reading from the local weight tier")
        return False
    print(f"local weight tier: {summary['local']} files read from {blob_dir}, tokens unchanged")
    return True


def _check_compile_names():
    """Names a compile writes (compile.py, modelstore.py, manifest.py): a tag
    must match modelstore's grammar as a whole string (a trailing newline
    passes a $-anchored match, then modelstore refuses it after the
    export), and a file's recorded source is the Hub's only for a commit."""
    from . import manifest, modelstore
    for good in ("main", "314113f8a7da_hero1", "v1.2-x"):
        if not modelstore.TAG_RE.fullmatch(good):
            raise SystemExit(f"tag {good!r} refused")
    for bad in ("v1\n", "-v1", "a/b", "", "x" * 129):
        if modelstore.TAG_RE.fullmatch(bad):
            raise SystemExit(f"tag {bad!r} accepted")
    commit = "314113f8a7dadb388215396ba5fbee2c610a04d6"
    src, read = manifest._hub_source("org/m", commit, "a.safetensors", "http://modelstore")
    if src != f"https://huggingface.co/org/m/resolve/{commit}/a.safetensors" or not read.startswith("http://modelstore/"):
        raise SystemExit(f"a commit's source should be the Hub's: {src}")
    src, read = manifest._hub_source("org/m", "main", "a.safetensors", "http://modelstore")
    if src != read:
        raise SystemExit(f"a moving ref's source should be the endpoint read: {src}")


def _check_chat_telemetry(artifact):
    """Chat (text in, text out; what clients without a tokenizer use)
    returns the turn's execution telemetry when asked, as generate does;
    the router records it for models served by name."""
    from .engine import Engine
    engine = Engine(artifact)
    reply = engine.chat(engine.new_session(), "Is the sky blue?", max_new_tokens=4, telemetry=vxpu_pb2.TELEMETRY_SUMMARY)
    tel = reply.get("telemetry") or {}
    if tel.get("format") != "vxpu-telemetry/1" or tel.get("decode_steps", 0) < 1:
        print(f"chat telemetry check: no telemetry record in the reply ({sorted(reply)})")
        return False
    plain = engine.chat(engine.new_session(), "Is the sky blue?", max_new_tokens=4)
    if "telemetry" in plain or plain["text"] != reply["text"]:
        print("chat telemetry check: telemetry changed the reply, or was recorded unasked")
        return False
    print(f"chat telemetry: {tel['decode_steps']} decode steps recorded, reply unchanged")
    return True


def _check_admin_telemetry():
    """Verify that Client with internal_address routes execute_ids to the
    Executor service on the admin port and receives execution telemetry,
    while generate_ids routes to Engine without telemetry."""
    from concurrent import futures
    import grpc
    from . import client, vxpu_pb2, vxpu_pb2_grpc

    class FakeEngine(vxpu_pb2_grpc.EngineServicer):
        def Generate(self, request, context):
            yield vxpu_pb2.GenerateResponse(token_ids=[10])
            yield vxpu_pb2.GenerateResponse(done=True, finish_reason="eos", generated=1)

    class FakeExecutor(vxpu_pb2_grpc.ExecutorServicer):
        def Execute(self, request, context):
            has_tel = request.telemetry != vxpu_pb2.TELEMETRY_NONE
            tel_json = json.dumps({"format": "vxpu-telemetry/1", "ok": True}) if has_tel else ""
            yield vxpu_pb2.ExecuteResponse(
                response=vxpu_pb2.GenerateResponse(token_ids=[20]))
            yield vxpu_pb2.ExecuteResponse(
                response=vxpu_pb2.GenerateResponse(done=True, finish_reason="eos", generated=1),
                telemetry_json=tel_json)

    eng_srv = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    vxpu_pb2_grpc.add_EngineServicer_to_server(FakeEngine(), eng_srv)
    eng_port = eng_srv.add_insecure_port("127.0.0.1:0")
    eng_srv.start()

    exec_srv = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    vxpu_pb2_grpc.add_ExecutorServicer_to_server(FakeExecutor(), exec_srv)
    exec_port = exec_srv.add_insecure_port("127.0.0.1:0")
    exec_srv.start()

    try:
        c = client.Client(f"127.0.0.1:{eng_port}", internal_address=f"127.0.0.1:{exec_port}")
        sess = client.Session(c, "m:s1")
        # generate_ids hits Engine, no telemetry
        g_tokens = list(sess.generate_ids([1, 2]))
        if g_tokens != [10] or sess.last.telemetry is not None:
            print("admin telemetry check: generate_ids unexpected reply")
            return False
        # execute_ids hits Executor admin listener, with telemetry
        e_tokens = list(sess.execute_ids([1, 2], telemetry=vxpu_pb2.TELEMETRY_SUMMARY))
        if e_tokens != [20] or sess.last.telemetry.get("format") != "vxpu-telemetry/1":
            print("admin telemetry check: execute_ids missing telemetry")
            return False
        c.close()
    finally:
        eng_srv.stop(None)
        exec_srv.stop(None)
    print("admin telemetry: execute_ids reaches the admin listener, returns telemetry, reply unchanged")
    return True


def _check_gpu_experts(model, prompts, settings, artifact, workdir):
    """On a GPU: the hot experts' kernels (triton_ops.fp4_experts_decode)
    against the module's own arithmetic on random fp4 banks, then the tiny
    DeepSeek with half of every layer's experts placed on the GPUs (split
    between two where there are two; experts_gpu.py) against the same
    model computing them all on the CPU (a different summation order: the rounding class, so a near tie
    may flip). Skipped without CUDA and Triton."""
    import torch
    from . import triton_ops
    if not triton_ops.available():
        print("gpu experts check skipped: no CUDA with Triton")
        return True
    from .deepseek_v41 import _fork
    ds = _fork()
    K, I, S, dev = 256, 128, 4, "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    def bank(n_out, n_in):
        w = torch.randint(0, 256, (S, n_out, n_in // 2), dtype=torch.uint8, device=dev, generator=g)
        sc = torch.randint(120, 125, (S, n_out, n_in // 32), dtype=torch.uint8, device=dev, generator=g)
        return w, sc.view(torch.float8_e8m0fnu)
    (w1, s1), (w3, s3), (w2, s2) = bank(I, K), bank(I, K), bank(K, I)
    x = triton_ops.act_quant_dequant(torch.randn(1, K, device=dev, generator=g).to(torch.bfloat16))
    weights = torch.rand(1, 3, device=dev, generator=g)
    slots = torch.tensor([[2, -1, 0]], device=dev)
    y = triton_ops.fp4_experts_decode(x, weights, slots, w1, s1, w3, s3, w2, s2, 10.0)

    class _M:
        swiglu_limit = 10.0
        def _quantized_input(self, t):
            return ds.MoE._quantized_input(self, t)
    ref = torch.zeros(1, K, device=dev)
    for k, sl in enumerate(slots[0].tolist()):
        if sl < 0:
            continue
        deq = [ds.MoE._dequant(w[sl:sl + 1], sc[sl:sl + 1]) for w, sc in ((w1, s1), (w3, s3), (w2, s2))]
        gate, up = (torch.bmm(m, x.view(1, K, 1)).squeeze(-1) for m in deq[:2])
        h = ds.MoE._ffn(_M(), gate, up, weights[0, k:k + 1])
        ref[0] += torch.bmm(deq[2], h.unsqueeze(-1)).squeeze(-1).float()[0]
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    if rel > 1e-2:
        print(f"gpu experts check: the kernels differ from the module's arithmetic (relative {rel:.3g})")
        return False

    with open(os.path.join(artifact, "binding.json")) as f:
        experts = (json.load(f).get("external") or {}).get("experts") or {}
    if not experts:
        print("gpu experts check: the model places no experts externally; kernels checked only")
        return True
    n = len(next(iter(next(iter(experts.values())).values()))["parts"])
    # half of every layer's experts on the GPUs: with two, alternating
    # between them, so every layer has pools on both
    gpus = min(torch.cuda.device_count(), 2)
    held = list(range(0, n, 2))
    placement = {"format": "vxpu-placement/2",
                 "gpus": [{"layers": {layer: held[g::gpus] for layer in experts}} for g in range(gpus)]}
    path = os.path.join(workdir, "placement.json")
    with open(path, "w") as f:
        json.dump(placement, f)
    runs = {}
    for label, value in (("cpu", None), ("gpu", path)):
        saved = os.environ.get("VXPU_GPU_EXPERTS")
        if value is None:
            os.environ.pop("VXPU_GPU_EXPERTS", None)
        else:
            os.environ["VXPU_GPU_EXPERTS"] = value
        try:
            runs[label] = record_vxpu_engine(model, "main", prompts, settings, artifact, device="cuda")
        finally:
            if saved is None:
                os.environ.pop("VXPU_GPU_EXPERTS", None)
            else:
                os.environ["VXPU_GPU_EXPERTS"] = saved
    if "gpu-hot+cpu" not in runs["gpu"]["producer"]["plan"]["selected"].get("vxpu::moe_experts", []):
        print("gpu experts check: the hot experts' implementation never ran")
    runs["cpu"] = _with_env({"VXPU_GPU_EXPERTS": None}, record_vxpu_engine,
                            model, "main", prompts, settings, artifact, device="cuda")
    runs["gpu"] = _with_env({"VXPU_GPU_EXPERTS": path}, record_vxpu_engine,
                            model, "main", prompts, settings, artifact, device="cuda", telemetry=vxpu_pb2.TELEMETRY_SUMMARY)
    if not any(n.startswith("hot experts on") for n in runs["gpu"]["producer"]["plan"].get("notes", [])):
        print("gpu experts check: the placement was not applied")
        return False
    rows, ok = compare(runs["cpu"], runs["gpu"], strict_tokens=True)
    print(f"hot experts on {gpus} GPU(s) (kernels relative error {rel:.2g}) against all on the CPU:")
    print(table(runs["cpu"], runs["gpu"], rows))
    return _check_expert_moves(model, prompts, settings, artifact, workdir, runs, gpus, list(experts), n) and ok


def _check_expert_moves(model, prompts, settings, artifact, workdir, runs, gpus, layers, n):
    """Experts moving with the traffic (experts_gpu.Mover): start from the
    worst placement, the experts the previous run selected least, so moves
    must happen; then the tokens against all on the CPU, and every held
    expert's rows on its GPU against the host's (experts_gpu.verify)."""
    import torch
    from . import experts_gpu
    from .expert_stats import collect
    counts = collect([runs["gpu"]])["decode"]
    cold = {layer: sorted(range(n), key=lambda e: counts[int(layer)][e])[:n // 2] for layer in layers}
    placement = {"format": "vxpu-placement/2",
                 "gpus": [{"layers": {layer: ids[g::gpus] for layer, ids in cold.items()}} for g in range(gpus)]}
    path = os.path.join(workdir, "placement-cold.json")
    with open(path, "w") as f:
        json.dump(placement, f)
    keep = []
    moved = _with_env({"VXPU_GPU_EXPERTS": path, "VXPU_GPU_EXPERTS_MOVES": "2"},
                      record_vxpu_engine, model, "main", prompts, settings, artifact, device="cuda",
                      telemetry=vxpu_pb2.TELEMETRY_SUMMARY, keep=keep)
    mover = keep[0].expert_mover
    recorded = sum(p["telemetry"]["host_ops"].get("decode", {}).get("moe_experts", {}).get("moves_completed", 0.0)
                   for p in moved["prompts"])
    if not (mover.completed and recorded):
        print(f"gpu experts check: from a placement of the least selected, {mover.started} moves started, "
              f"{mover.completed} completed ({recorded:g} per step in the telemetry)")
        return False
    torch.cuda.synchronize()
    for moe in keep[0].external.experts.values():
        res = getattr(moe, "_vxpu_hot", None)
        if res is not None:
            experts_gpu.verify(res)
    rows, ok = compare(runs["cpu"], moved, strict_tokens=True)
    print(f"experts moving with the traffic ({mover.started} moves started, {mover.completed} completed; "
          "the residency matches the host banks) against all on the CPU:")
    print(table(runs["cpu"], moved, rows))
    return ok


def _check_expert_mailboxes(model, prompts, settings, artifact, plain, workdir):
    """Decode's experts as mailboxes (mailboxes.py): with every expert on
    the CPU, bit-identical to the plain engine (each reply is the fused
    kernel's own term, summed in its order). On a GPU also with half of
    every layer's experts in GPU pools, against the plain GPU engine: the
    rounding class (a GPU's replies may differ in the last bf16 bit)."""
    import torch
    keep = []
    got = _with_env({"VXPU_EXPERT_MAILBOXES": "1", "VXPU_GPU_EXPERTS": None}, record_vxpu_engine,
                    model, "main", prompts, settings, artifact, keep=keep)
    selected = {s["impl"] for s in keep[0].plan.explain(keep[0].plan_ops())["ops"]["vxpu::moe_experts"]["selected"]}
    if "mailboxes" not in selected:
        print(f"expert mailboxes check: the mailboxes implementation was not selected ({sorted(selected)})")
        return False
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]])
    rows, _ = compare(plain, got, strict_tokens=True)
    print("expert mailboxes, every expert on the CPU, against the plain engine (must be bit-identical):")
    print(table(plain, got, rows))
    if not same:
        print("expert mailboxes check: not bit-identical with every expert on the CPU")
        return False
    if not torch.cuda.is_available():
        return True
    with open(os.path.join(artifact, "binding.json")) as f:
        experts = (json.load(f).get("external") or {}).get("experts") or {}
    n = len(next(iter(next(iter(experts.values())).values()))["parts"])
    gpus = min(torch.cuda.device_count(), 2)
    held = list(range(0, n, 2))
    path = os.path.join(workdir, "placement-mailboxes.json")
    with open(path, "w") as f:
        json.dump({"format": "vxpu-placement/2",
                   "gpus": [{"layers": {layer: held[g::gpus] for layer in experts}} for g in range(gpus)]}, f)
    ref = _with_env({"VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}, record_vxpu_engine,
                    model, "main", prompts, settings, artifact, device="cuda")
    pooled = _with_env({"VXPU_EXPERT_MAILBOXES": "1", "VXPU_GPU_EXPERTS": path}, record_vxpu_engine,
                       model, "main", prompts, settings, artifact, device="cuda")
    rows, ok = compare(ref, pooled, strict_tokens=True)
    print(f"expert mailboxes with half the experts in {gpus} GPU pool(s), against the plain GPU engine:")
    print(table(ref, pooled, rows))
    return ok


def _check_step_graph(model, prompts, settings, artifact, workdir):
    """On a GPU: the whole decode step in one CUDA graph (stepgraph.py),
    the AOTInductor segments and the host ops as mailboxes the GPU waits
    on, answered by the native step host (each host node's native package,
    #409), against the same dataflow over the same segments run node by
    node with the host ops in the engine: tokens and logits bit-identical,
    the step really captured, and the host thread gone with the engine.
    Skipped without CUDA."""
    import torch
    if not torch.cuda.is_available():
        print("step graph check skipped: no CUDA")
        return True
    base = {"VXPU_DATAFLOW": "1", "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "aoti"),
            "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}
    ref = _with_env({**base, "VXPU_STEP_GRAPH": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True)
    keep = []
    got = _with_env({**base, "VXPU_STEP_GRAPH": "1"}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True, keep=keep)
    if not isinstance(getattr(keep[0]._decode_split, "graph", None), torch.cuda.CUDAGraph):
        print("step graph check: the decode step was never captured")
        return False
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in ref["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in ref["prompts"]])
    rows, _ = compare(ref, got, strict_tokens=True)
    print("the decode step as one CUDA graph, against the same segments run node by node (must be bit-identical):")
    print(table(ref, got, rows))
    import gc
    host = keep[0]._decode_split._host
    keep.clear()  # the engine dropped: its step graph's host thread must let go of the model
    gc.collect()
    if host.running():
        print("step graph check: the host thread outlived its engine")
        return False
    return same


def _check_step_graph_pools(model, prompts, settings, artifact, workdir):
    """On a GPU: the step graph with half of every layer's experts in GPU
    pools (split between two GPUs where there are two: the other GPU's
    graph takes part), the CPU answering the other pairs through its
    mailbox, against the same dataflow run node by node with expert
    mailboxes and the same placement (the same per-pair replies, summed
    in the same order): bit-identical."""
    import torch
    if not torch.cuda.is_available():
        return True
    with open(os.path.join(artifact, "binding.json")) as f:
        experts = (json.load(f).get("external") or {}).get("experts") or {}
    if not experts:
        return True
    n = len(next(iter(next(iter(experts.values())).values()))["parts"])
    gpus = min(torch.cuda.device_count(), 2)  # with two, half on each: the other GPU's graph takes part too
    held = list(range(0, n, 2))
    path = os.path.join(workdir, "placement-step-graph.json")
    with open(path, "w") as f:
        json.dump({"format": "vxpu-placement/2",
                   "gpus": [{"layers": {layer: held[g::gpus] for layer in experts}} for g in range(gpus)]}, f)
    base = {"VXPU_DATAFLOW": "1", "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "aoti"),
            "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": "1", "VXPU_GPU_EXPERTS": path}
    ref = _with_env({**base, "VXPU_STEP_GRAPH": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True, telemetry=vxpu_pb2.TELEMETRY_SUMMARY)
    keep = []
    got = _with_env({**base, "VXPU_STEP_GRAPH": "1"}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda", compile_decode=True, telemetry=vxpu_pb2.TELEMETRY_SUMMARY, keep=keep)
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in ref["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in ref["prompts"]])
    rows, _ = compare(ref, got, strict_tokens=True)
    print(f"the step graph with half the experts in GPU pools on {gpus} GPU(s), against the same run node by node "
          "(must be bit-identical):")
    print(table(ref, got, rows))
    # the experts nodes answered in the graph record their selections too:
    # the statistics placements are made from see the same as node by node
    selections = [[(p.get("telemetry") or {}).get("experts", {}).get("decode") for p in d["prompts"]] for d in (ref, got)]
    if selections[0] != selections[1] or not any(selections[0]):
        print("step graph pools check: decode expert selections differ from the node-by-node run's")
        return False
    return same and isinstance(getattr(keep[0]._decode_split, "graph", None), torch.cuda.CUDAGraph)


def _check_dataflow(model, prompts, settings, artifact, plain):
    """The engine's decode run as a dataflow graph (dataflow.py): the same
    tokens and logits as the plain run, every edge through an inbox. On a
    GPU also with the host-op nodes placed on the CPU: their inputs and
    outputs then cross devices through the inboxes (the GPU run first
    without, as the reference)."""
    import torch
    keep = []
    got = _with_env({"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": None}, record_vxpu_engine,
                    model, "main", prompts, settings, artifact, keep=keep)
    runner = keep[0]._decode_split
    rows, ok = compare(plain, got, strict_tokens=True)
    plan = runner.inbox_plan()
    print(f"{runner.graph.describe()}, {len(plan)} inboxes, {sum(i['bytes'] for i in plan)} bytes a step, "
          "against the plain engine:")
    print(table(plain, got, rows))
    if not torch.cuda.is_available() or not any(n.kind != "dense" for n in runner.graph.nodes):
        return ok
    ref = _with_env({"VXPU_DATAFLOW": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    device="cuda")
    keep = []
    placed = _with_env({"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": "experts=cpu,engram=cpu"}, record_vxpu_engine,
                       model, "main", prompts, settings, artifact, device="cuda", keep=keep)
    crossing = sum(i["bytes"] for i in keep[0]._decode_split.inbox_plan() if not i["device"].startswith("cuda"))
    rows2, ok2 = compare(ref, placed, strict_tokens=True)
    print(f"dataflow on the GPU with the host-op nodes on the CPU ({crossing} bytes a step into CPU inboxes), "
          "against the plain GPU engine:")
    print(table(ref, placed, rows2))
    return ok and ok2


def _with_env(env, fn, *args, **kwargs):
    """fn(*args, **kwargs) with env set (None: unset), restored after."""
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return fn(*args, **kwargs)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _check_mover_lifetime():
    """The expert mover (experts_gpu.Mover) is freed with its engine: its
    worker thread holds it only weakly, so the pools and banks it reaches
    go with it, and the thread exits."""
    import gc
    import threading
    import weakref
    from . import experts_gpu
    mover = experts_gpu.Mover(2, 0.25)
    ref = weakref.ref(mover)
    del mover
    gc.collect()
    if ref() is not None:
        raise SystemExit("expert mover: still referenced after its engine let it go (the worker holds it)")
    deadline = time.time() + 5
    while any(t.name == "vxpu-expert-moves" for t in threading.enumerate()) and time.time() < deadline:
        time.sleep(0.1)
    if any(t.name == "vxpu-expert-moves" for t in threading.enumerate()):
        raise SystemExit("expert mover: its worker thread outlived it")


def _check_node_client():
    """The engine's node client (#405) against a node server that goes
    away and comes back on its port: the call in between fails at once
    (no hang), and the next call connects again."""
    from . import kernels
    k = kernels.load()

    def call(client):
        try:
            client.call(0, [])
        except RuntimeError as e:
            return str(e).splitlines()[0]
        return "answered"

    server = k.NodeServer(0, [], [])  # no nodes: every call is answered with an error
    port = server.port()
    client = k.NodeClient("127.0.0.1", port)
    first = call(client)
    server.stop()
    gone = call(client)
    server = k.NodeServer(port, [], [])
    back = call(client)
    server.stop()
    ok = "no such node" in first and "cannot connect" in gone and "no such node" in back
    print(f"node client: a server that goes away fails the call ({gone!r}), and the next call connects again: {ok}")
    return ok


def _check_cuts(artifact):
    """Where the step can be cut into ranges of whole layers (cuts.py,
    #387): the tiny DeepSeek's layer 1 writes a compressed KV cache layer 2
    reads (kv_source_layers 1 and 3), so a range may start before layers 1
    and 3 but not 2, in decode and prefill alike, and what crosses a cut is
    a few activations."""
    from . import cuts
    d = cuts.describe(artifact)
    why = d["decode"].get(2, {}).get("state", [])
    ok = (d["legal"] == [1, 3] and "model.layers.1.attn.compress_kv_cache" in why
          and all(0 < c["message_bytes"] < 1 << 20 for c in d["decode"].values()))
    print(f"layer cuts: legal before {d['legal']}; not before 2 ({', '.join(why)}); "
          f"{d['decode'][1]['message_bytes']} bytes cross a decode cut")
    if not ok:
        print(f"layer cuts check: {d}")
    return ok


def _check_slices(model, prompts, settings, artifact, plain):
    """The step as ranges of whole layers (dataflow.Chain, #387): decode and
    prefill each cut where cuts.py allows (the tiny DeepSeek's: before
    layers 1 and 3), the ranges run in turn, each taking only the step's
    inputs and what the ranges before it made for it. The same segments
    on the same arguments: tokens and logits bit-identical to the plain
    engine's."""
    from . import cuts, dataflow, segments
    starts = cuts.describe(artifact)["legal"]
    graphs = {}

    def chained(engine):
        engine._decode_run = graphs["decode"] = dataflow.Chain(engine._decode_split.split, starts)
        prefill, _ = segments.split_at_host_ops(engine._prefill)
        engine._prefill = engine._prefill_run = graphs["prefill"] = dataflow.Chain(prefill, starts)

    got = _with_env({"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None,
                     "VXPU_GPU_EXPERTS": None}, record_vxpu_engine, model, "main", prompts, settings, artifact,
                    prepare=chained)
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]])
    # in one process the ranges share the state tensors, so the tokens would match at any cut; what
    # lets them run on different machines is that no state is read by two ranges, as cuts.py says
    with open(os.path.join(artifact, "binding.json")) as f:
        state = set(json.load(f)["state"])

    def shared(chain):
        per = [r.graph.residents() & state for r in chain.runners]
        return sorted(set().union(*(a & b for i, a in enumerate(per) for b in per[i + 1:])))

    overlap = shared(graphs["decode"]) + shared(graphs["prefill"])
    illegal = sorted(set(range(1, max(starts) + 1)) - set(starts))
    wrong = shared(dataflow.Chain(graphs["decode"].split, illegal)) if illegal else ["(no illegal cut to try)"]
    if overlap or not wrong:
        print(f"layer ranges check: state read by two ranges: {overlap}; at the illegal cuts {illegal}: {wrong}")
        same = False
    ranges = ", ".join(f"{len(r.graph.nodes)} nodes" for r in graphs["decode"].runners)
    print(f"the step as {len(starts) + 1} ranges of whole layers, starting at layers 0, "
          f"{', '.join(map(str, starts))} (decode: {ranges}; no state read by two ranges, where cutting before "
          f"layer {illegal} would share {', '.join(wrong)}), against the plain engine (must be bit-identical):")
    rows, _ = compare(plain, got, strict_tokens=True)
    print(table(plain, got, rows))
    if not same:
        print("layer ranges check: not bit-identical")
    return same


def _check_engine_ranges(model, prompts, settings, artifact, plain):
    """Two engines, each running a range of whole layers (#387): the head
    engine layers up to the last legal cut (VXPU_LAYER_RANGE=0:N), the
    other the rest (N:), each holding only its own layers' experts and
    tables and its own sessions' state for them. The head hands each step
    on (Engine.attach_ranges; here in one process, LocalRange), prefill and
    decode, every prompt its own session, and a two-turn conversation in
    one session: tokens and logits bit-identical to the plain engine's."""
    from . import cuts
    from .engine import Engine, LocalRange
    cut = cuts.describe(artifact)["legal"][-1]
    base = {"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}
    tail = _with_env({**base, "VXPU_LAYER_RANGE": f"{cut}:"}, Engine, artifact)
    keep = []
    got = _with_env({**base, "VXPU_LAYER_RANGE": f"0:{cut}"}, record_vxpu_engine, model, "main", prompts, settings,
                    artifact, keep=keep, prepare=lambda head: head.attach_ranges([LocalRange(tail, head.device)]))
    head = keep[0]
    misplaced = ([f"head holds layer {i}" for i, m in head.external.experts.items() if i >= cut and not m.w1.is_meta]
                 + [f"tail holds layer {i}" for i, m in tail.external.experts.items() if i < cut and not m.w1.is_meta]
                 + [f"head holds table {i}" for i in head.external.engram if i >= cut]
                 + [f"tail holds table {i}" for i in tail.external.engram if i < cut])
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]])
    # a conversation: the second turn continues the first's session (for a model whose prefill starts
    # from position 0, the session starts over and the head resets the other range's state too)
    from .engine import SamplingParams
    reference = _with_env(base, Engine, artifact)
    turns = []
    for engine in (reference, head):
        sid, out, ids = engine.new_session(), [], list(ID_PROMPTS[0]["ids"])
        for turn in range(2):
            new = [t for ev in engine.generate(sid, ids, 8, SamplingParams(do_sample=False)) for t in ev["token_ids"]]
            out.append(new)
            ids = ids + new + list(ID_PROMPTS[1]["ids"][:5 + turn])
        turns.append(out)
    if turns[0] != turns[1]:
        print(f"engine ranges check: a two-turn conversation differs: {turns}")
        same = False
    print(f"two engines, layers 0-{cut - 1} and {cut}-last, the head handing each step on "
          f"({len(tail.sessions)} sessions on the other), against the plain engine (must be bit-identical):")
    rows, _ = compare(plain, got, strict_tokens=True)
    print(table(plain, got, rows))
    if misplaced or not same:
        print(f"engine ranges check: {'not bit-identical; ' if not same else ''}{misplaced}")
    return same and not misplaced


def _cuda():
    import torch
    return torch.cuda.is_available()


def _check_range_server(model, prompts, settings, artifact, plain, device="cpu"):
    """Check a range of whole layers served over the network (#387), set up
    the way the coordinator sets it up:
    - an executor loads the model with the layers from the last legal cut
      onward (LoadModel with VXPU_LAYER_RANGE) and serves them on a local
      port (ServeRange, rangeserver.py);
    - the head engine's plan names that port (VXPU_REMOTE_RANGES), and it
      sends each step there: prefills and decode steps, each prompt in its
      own session, and a two-turn conversation.
    Tokens and logits must be bit-identical to the plain engine's. On cuda
    (with the step graph) the range's decode steps run natively in C++."""
    import time
    from . import cuts
    from .engine import Engine, SamplingParams
    cut = cuts.describe(artifact)["legal"][-1]
    base = {"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}
    kw = {}
    if device != "cpu":
        base.update({"VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_STEP_GRAPH": "1"})
        kw = {"device": device, "compile_decode": True}
    if device != "cpu":  # the reference: one engine, the same step graph on the same GPU
        plain = _with_env(base, record_vxpu_engine, model, "main", prompts, settings, artifact, **kw)
    # the other range as the coordinator starts it: an executor loads the model with its range
    # (LoadModel, VXPU_LAYER_RANGE) and serves it (ServeRange); the head is wired by its plan
    import grpc
    from . import vxpu_pb2
    from .server import ExecutorServicer

    class Context:
        code = details = None

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    files = {}
    for name in ("manifest.json", "binding.json", "prefill.pt2", "decode.pt2"):
        with open(os.path.join(artifact, name), "rb") as f:
            files[name] = f.read()
    executor = ExecutorServicer(work_dir=tempfile.mkdtemp(prefix="vxpu-range-"), device=device,
                                compile_decode=device != "cpu")
    plan = {k: v for k, v in {**base, "VXPU_LAYER_RANGE": f"{cut}:"}.items() if v is not None}
    ctx = Context()
    executor.LoadModel(vxpu_pb2.LoadModelRequest(manifest_json=files["manifest.json"].decode(),
                                                 binding_json=files["binding.json"].decode(),
                                                 prefill_graph=files["prefill.pt2"], decode_graph=files["decode.pt2"],
                                                 plan=plan), ctx)
    if ctx.code is not None:  # LoadModel refused the plan or failed, so nothing will start serving
        print(f"range server check: LoadModel failed: {ctx.code} {ctx.details}")
        return False
    port = 0
    for _ in range(600):
        ctx = Context()
        resp = executor.ServeRange(vxpu_pb2.ServeRangeRequest(), ctx)
        if resp.port:
            port = resp.port
            break
        # ServeRange answers FAILED_PRECONDITION while the engine is still
        # loading, which means "ask again". Any other error means the load
        # failed, so stop waiting and report it below.
        if ctx.code is not None and ctx.code != grpc.StatusCode.FAILED_PRECONDITION:
            break
        time.sleep(1)
    if not port:
        print(f"range server check: ServeRange never answered: {ctx.code} {ctx.details}")
        return False
    server = executor.range_server[1]
    try:
        keep = []
        address = f"tcp://127.0.0.1:{port}"
        got = _with_env({**base, "VXPU_LAYER_RANGE": f"0:{cut}", "VXPU_REMOTE_RANGES": f"{cut}:={address}"},
                        record_vxpu_engine, model, "main", prompts, settings, artifact, keep=keep, **kw)
        head = keep[0]
        sid, turns, ids = head.new_session(), [], list(ID_PROMPTS[0]["ids"])
        for turn in range(2):
            new = [t for ev in head.generate(sid, ids, 8, SamplingParams(do_sample=False)) for t in ev["token_ids"]]
            turns.append(new)
            ids = ids + new + list(ID_PROMPTS[1]["ids"][:5 + turn])
        messages, native = server.messages(), server.native_steps()
    finally:
        server.stop()
    reference = _with_env(base, Engine, artifact, **kw)
    sid, want, ids = reference.new_session(), [], list(ID_PROMPTS[0]["ids"])
    for turn in range(2):
        new = [t for ev in reference.generate(sid, ids, 8, SamplingParams(do_sample=False)) for t in ev["token_ids"]]
        want.append(new)
        ids = ids + new + list(ID_PROMPTS[1]["ids"][:5 + turn])
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]] and turns == want)
    print(f"a range of layers {cut}-last on a range server over tcp://, the head sending each step ({messages} messages, "
          f"{native} decode steps replayed natively), against the plain engine (must be bit-identical):")
    rows, _ = compare(plain, got, strict_tokens=True)
    print(table(plain, got, rows))
    if not same:
        print(f"range server check: not bit-identical (conversation {turns} against {want})")
    return same


def _check_native_nodes(model, prompts, settings, artifact, workdir):
    """The decode step's host nodes served native (nodes.py,
    nodeserver.py, #394): a node server for every experts and n-gram node,
    each lowered to the native kernels and compiled by AOTInductor (the
    experts nodes, one package for all layers), sent the inputs the engine
    gave it in a decode step over TCP, replies bit-identically what the
    engine's node returned."""
    import time
    import torch
    from . import nodes, nodeserver
    from .engine import SamplingParams
    keep = []
    _with_env({"VXPU_DATAFLOW": "1", "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None,
               "VXPU_GPU_EXPERTS": None}, record_vxpu_engine, model, "main", prompts, settings, artifact, keep=keep)
    engine = keep[0]
    split = engine._decode_split.split
    host = [n.name for n in engine._decode_split.graph.nodes if n.kind != "dense"]
    seen = {}

    def hook(name):
        def record(module, args, out):
            if name not in seen:  # the first decode step's
                seen[name] = ([a.clone() if isinstance(a, torch.Tensor) else a for a in args], out)
        return record

    hooks = [getattr(split, name).register_forward_hook(hook(name)) for name in host]
    try:
        tok = _tokenizer(model, "main", prompts)
        ids = _encode(tok, next(iter(_prompt_set(tok, prompts))))
        for _ in engine.generate(engine.new_session(), ids, 2, SamplingParams(do_sample=False)):
            pass
    finally:
        for h in hooks:
            h.remove()
    server = nodeserver.start(artifact, host, cache_dir=os.path.join(workdir, "native-nodes"))
    bad, calls = [], 0
    # every experts layer's node is the same code on its own banks: one package
    kinds = nodes.host_calls(split)
    experts = {path for name, path in zip(host, server.packages) if kinds[name][0] == "experts"}
    if len(experts) != 1:
        bad.append(f"{len(experts)} packages for the experts nodes")
    t = time.perf_counter()
    idle = nodeserver.connect(f"127.0.0.1:{server.port()}")  # still connected when the server stops
    try:
        sock = nodeserver.connect(f"127.0.0.1:{server.port()}")
        for i, name in enumerate(host):
            args, want = seen[name]
            got = nodeserver.call(sock, i, [a for a in args if isinstance(a, torch.Tensor)])
            calls += 1
            want = list(want) if isinstance(want, (tuple, list)) else [want]
            if len(got) != len(want) or not all(torch.equal(g, w) for g, w in zip(got, want)):
                bad.append(name)
        sock.close()
    finally:
        served = server.messages()
        server.stop()
    print(f"native nodes: {len(host) - len(bad)} of {len(host)} host nodes (experts, n-gram) bit-identical from a "
          f"node server ({served} messages, {1e6 * (time.perf_counter() - t) / max(calls, 1):.0f} us a call)")
    if bad:
        print(f"native nodes check: {bad} differ from the engine's")
    # stopping the server ends its connections (their threads would
    # otherwise outlive it), so a connected client sees the end at once
    idle.settimeout(5)
    try:
        ended = idle.recv(1) == b""
    except (TimeoutError, OSError):
        ended = False
    idle.close()
    if not ended:
        print("native nodes check: a connection outlived its stopped server")
    return not bad and ended


def _check_remote_nodes(model, prompts, settings, artifact, plain, workdir):
    """Host nodes in another executor (#394), as the coordinator places
    them: the executor describes the decode step's nodes (DescribeNodes),
    and serves every other experts layer's node and the n-gram nodes
    (ServeNodes) from a native node server. The engine, which does not
    load their banks or tables, sends their host ops' inputs to the
    server's mailboxes (remote_nodes), prefill and decode alike. The
    server runs the native kernels on the same arguments, so tokens and
    logits must be bit-identical to the plain engine's."""
    from . import vxpu_pb2
    from .server import ExecutorServicer

    class Context:  # what a gRPC call's context records
        code = details = None

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

        def is_active(self):
            return True

    files = {}
    for name in ("manifest.json", "binding.json", "decode.pt2"):
        with open(os.path.join(artifact, name), "rb") as f:
            files[name] = f.read()
    request = dict(manifest_json=files["manifest.json"].decode(), binding_json=files["binding.json"].decode(),
                   decode_graph=files["decode.pt2"])
    executor, ctx = ExecutorServicer(work_dir=tempfile.mkdtemp(prefix="vxpu-nodes-")), Context()
    described = json.loads(executor.DescribeNodes(vxpu_pb2.DescribeNodesRequest(**request), ctx).nodes_json or "[]")
    host = [n for n in described if "collection" in n]
    if ctx.code is not None or not host or any(n["resident_bytes"] <= 0 for n in host):
        print(f"remote nodes check: DescribeNodes failed or described no host nodes: {ctx.code} {ctx.details} {described}")
        return False
    experts = sorted((n["index"], n["name"]) for n in host if n["collection"] == "experts")
    served = [name for _, name in experts[::2]] + [n["name"] for n in host if n["collection"] == "engram"]
    calls = {n["name"]: (n["collection"], n["index"]) for n in host}
    resp = _with_env({"VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "native-nodes")}, executor.ServeNodes,
                     vxpu_pb2.ServeNodesRequest(nodes=served, **request), ctx)
    if ctx.code is not None or not resp.port:
        print(f"remote nodes check: ServeNodes failed: {ctx.code} {ctx.details}")
        return False
    server = executor.node_server[1]
    graph_ok = True
    try:
        spec = ",".join(f"{name}=tcp://127.0.0.1:{resp.port}/{i}" for i, name in enumerate(served))
        keep = []
        got = _with_env({"VXPU_REMOTE_NODES": spec, "VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None},
                        record_vxpu_engine, model, "main", prompts, settings, artifact, keep=keep)
        external = keep[0].external
        loaded = [f"{kind}.{i}" for name in served for kind, i in [calls[name]]
                  if (kind == "experts" and not external.experts[i].w1.is_meta) or (kind == "engram" and i in external.engram)]
        keep.clear()
        graph_ok = _check_step_graph_remote(model, prompts, settings, artifact, workdir, spec)
        messages = server.messages()
    finally:
        server.stop()
    if loaded:
        print(f"remote nodes check: {loaded} are served remotely but were loaded here too")
        return False
    same = ([p["generated_ids"] for p in got["prompts"]] == [p["generated_ids"] for p in plain["prompts"]]
            and [p["top_logits"] for p in got["prompts"]] == [p["top_logits"] for p in plain["prompts"]])
    rows, _ = compare(plain, got, strict_tokens=True)
    print(f"host nodes {served} served by a native node server ({messages} messages), against the plain engine "
          "(must be bit-identical):")
    print(table(plain, got, rows))
    if not same:
        print("remote nodes check: not bit-identical")
    return same and graph_ok


def _check_step_graph_remote(model, prompts, settings, artifact, workdir, spec):
    """On a GPU: the step graph with the node server's nodes (spec, a
    VXPU_REMOTE_NODES value), against the step graph with every node here.
    The step host (#409) answers both natively, through the native client
    or the node's own package: the same kernels on the same arguments, so
    tokens and logits must be bit-identical. Skipped without CUDA."""
    import torch
    if not torch.cuda.is_available():
        return True
    base = {"VXPU_DATAFLOW": "1", "VXPU_SEGMENT_RUNTIME": "aoti", "VXPU_AOTI_CACHE_DIR": os.path.join(workdir, "aoti"),
            "VXPU_STEP_GRAPH": "1", "VXPU_DATAFLOW_PLACE": None, "VXPU_EXPERT_MAILBOXES": None, "VXPU_GPU_EXPERTS": None}
    here = _with_env(base, record_vxpu_engine, model, "main", prompts, settings, artifact, device="cuda",
                     compile_decode=True)
    there = _with_env({**base, "VXPU_REMOTE_NODES": spec}, record_vxpu_engine, model, "main", prompts, settings,
                      artifact, device="cuda", compile_decode=True)
    same = ([p["generated_ids"] for p in there["prompts"]] == [p["generated_ids"] for p in here["prompts"]]
            and [p["top_logits"] for p in there["prompts"]] == [p["top_logits"] for p in here["prompts"]])
    rows, _ = compare(here, there, strict_tokens=True)
    print("the step graph with host nodes on a node server, against the step graph with all of them here "
          "(must be bit-identical):")
    print(table(here, there, rows))
    if not same:
        print("step graph remote check: not bit-identical")
    return same


def _check_plan_given():
    """The switches the coordinator sends (LoadModelRequest.plan, #394):
    read ahead of the environment and recorded as "coordinator"; only the
    execution and placement switches are taken (not the machine's paths
    or bucket, nor what only the kernels read), never
    empty, and a placement only inline; a placement inline parses like a
    file."""
    from . import experts_gpu
    from .plan import ConfigError, Plan
    got = _with_env({"VXPU_FUSED_EXPERTS": "1"}, Plan, "cpu", triton_available=False,
                    given={"VXPU_FUSED_EXPERTS": "0"})
    if got.choices["fused_experts"].value is not False or got.choices["fused_experts"].source != "coordinator":
        raise SystemExit(f"plan: the coordinator's switch should win over the environment's: {got.choices['fused_experts']}")
    if _with_env({"VXPU_FUSED_EXPERTS": "0"}, Plan, "cpu", triton_available=False).choices["fused_experts"].source != "env":
        raise SystemExit("plan: without the coordinator, the environment's switch should apply")
    for bad in ({"VXPU_NO_SUCH_SWITCH": "1"}, {"VXPU_KERNEL_THREADS": "4"}, {"VXPU_KERNELS": "0"},
                {"VXPU_BLOB_DIR": "/tmp/x"}, {"VXPU_AOTI_CACHE_DIR": "/tmp/x"}, {"VXPU_FUSED_EXPERTS": " "},
                {"VXPU_GPU_EXPERTS": "/etc/placement.json"}, {"VXPU_GPU_EXPERTS": "{not json"},
                {"VXPU_GPU_EXPERTS": "[1, 2]"}, {"VXPU_FUSED_EXPERTS": "maybe"}):
        try:
            Plan("cpu", triton_available=False, given=bad)
        except ConfigError:
            continue
        raise SystemExit(f"plan: {bad} should be refused")
    if experts_gpu.load_placement('{"format": "vxpu-placement/2", "gpus": [{"layers": {"3": [1, 2]}}]}') != [{3: [1, 2]}]:
        raise SystemExit("placement: an inline placement should parse like a file")
    # LoadModel answers every refusal as INVALID_ARGUMENT (final: the router doesn't retry), a
    # switch's own check (a plain ValueError) included, before it touches the artifact
    import grpc
    from . import vxpu_pb2
    from .server import ExecutorServicer

    class Context:
        code = details = None

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    ctx = Context()
    ExecutorServicer(work_dir=tempfile.mkdtemp(prefix="vxpu-plan-"), device="cpu").LoadModel(
        vxpu_pb2.LoadModelRequest(plan={"VXPU_DATAFLOW_PLACE": "experts"}), ctx)
    if ctx.code != grpc.StatusCode.INVALID_ARGUMENT:
        raise SystemExit(f"plan: LoadModel answered a refused plan with {ctx.code} ({ctx.details})")
    print("plan check: the coordinator's switches win over the environment's; others, empty ones and paths are refused")


def _check_dedup_constants(workdir):
    """Assertions on constant deduplication: tiny-deepseek alias mapping,
    the conflicting-names case, and the canonical-in-other-program case."""
    import torch
    import zipfile
    from .export import deduplicate_constants

    art = os.path.join(workdir, "tiny-deepseek-v41-artifact")
    if os.path.exists(art):
        with open(os.path.join(art, "binding.json")) as f:
            binding = json.load(f)
        aliases = binding.get("aliases", {})
        constants = set(binding.get("constant", []))
        for layer in (2, 3):
            fqn = f"model.layers.{layer}.attn.freqs_cis"
            if aliases.get(fqn) != "model.layers.1.attn.freqs_cis" or fqn in constants:
                raise SystemExit(f"rotary table {fqn} was not deduplicated to layer 1")
        if "model.layers.1.attn.freqs_cis" not in constants or "model.layers.0.attn.freqs_cis" not in constants:
            raise SystemExit("canonical rotary tables missing from binding constants")
        with zipfile.ZipFile(os.path.join(art, "decode.pt2")) as z:
            blobs = [f for f in z.namelist() if "constants/tensor_" in f]
            if len(blobs) > 10:
                raise SystemExit(f"expected <= 10 constant blobs in decode.pt2, got {len(blobs)}")

    class MConf(torch.nn.Module):
        def __init__(self, c0, c1):
            super().__init__()
            self.register_buffer("c0", c0, persistent=False)
            self.register_buffer("c1", c1, persistent=False)

        def forward(self, x):
            return x + self.c0 + self.c1

    x, y, z = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
    p_pref = torch.export.export(MConf(x, z), (torch.zeros(1),), strict=False)
    p_dec = torch.export.export(MConf(y, x), (torch.zeros(1),), strict=False)
    aliases = deduplicate_constants({}, p_pref, p_dec)
    if aliases != {}:
        raise SystemExit(f"conflicting names should not produce aliases: {aliases}")
    if "c0" not in p_pref.constants or "c1" not in p_pref.constants:
        raise SystemExit("constants missing from prefill after conflicting-names check")
    if "c0" not in p_dec.constants or "c1" not in p_dec.constants:
        raise SystemExit("constants missing from decode after conflicting-names check")

    class M1(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.register_buffer("c0", t, persistent=False)

        def forward(self, x):
            return x + self.c0

    class M2(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.register_buffer("c1", t, persistent=False)
            self.register_buffer("c2", t, persistent=False)

        def forward(self, x):
            return x + self.c1 + self.c2

    t = torch.tensor([42.0])
    p1 = torch.export.export(M1(t), (torch.zeros(1),), strict=False)
    p2 = torch.export.export(M2(t), (torch.zeros(1),), strict=False)
    aliases = deduplicate_constants({}, p1, p2)
    if aliases != {"c2": "c1"}:
        raise SystemExit(f"expected alias c2 -> c1, got {aliases}")
    if list(p2.constants.keys()) != ["c1"] or list(p1.constants.keys()) != ["c0"]:
        raise SystemExit(
            f"expected p2=['c1'], p1=['c0'], got {list(p2.constants.keys())}, {list(p1.constants.keys())}"
        )
    assert_no_duplicate_constants(p1)
    assert_no_duplicate_constants(p2)


def _check_copy_worker_failure(artifact):
    """Confirm that a copy worker failure in _fetch_bound raises promptly instead of hanging."""
    from .engine import Engine
    call_count = 0

    def fail_hook(*args):
        nonlocal call_count
        call_count += 1
        if call_count >= 5:
            raise RuntimeError("simulated copy worker failure")

    try:
        Engine(artifact, device="cpu", _copy_worker_hook=fail_hook)
        print("copy worker failure check: Engine did not raise on worker failure")
        return False
    except RuntimeError as e:
        if "simulated copy worker failure" not in str(e):
            print(f"copy worker failure check: unexpected error {e}")
            return False
        return True


def _check_probe_failure(artifact):
    """Confirm that a probe load failure cancels dense weight fetching and raises promptly."""
    from unittest.mock import patch
    from .engine import Engine

    def fail_load(*args, **kwargs):
        raise RuntimeError("simulated probe failure")

    with patch("torch.export.load", side_effect=fail_load):
        try:
            Engine(artifact, device="cpu")
            print("probe failure check: Engine did not raise on probe failure")
            return False
        except RuntimeError as e:
            if "simulated probe failure" not in str(e):
                print(f"probe failure check: unexpected error {e}")
                return False
            return True


def _check_external_dir(model, prompts, settings, artifact, ref, workdir):
    """Check that the engine can keep its external tensors (expert banks
    and n-gram tables) in files under external_dir. The first load
    fetches them and leaves no .partial files behind. A second load with
    a deleted bank and an unlocked stray partial file cleans up the stray,
    refetches only that bank, and still matches the eager reference
    token for token."""
    import shutil
    ext_dir = os.path.join(workdir, "tiny-deepseek-v41-external")
    shutil.rmtree(ext_dir, ignore_errors=True)
    from .engine import Engine
    first_engine = Engine(artifact, external_dir=ext_dir)
    if first_engine.external_bytes_fetched <= 0:
        print(f"external_dir check: first load fetched {first_engine.external_bytes_fetched} bytes; expected > 0")
        return False

    ext_files, partial_files = [], []
    for root, _, filenames in os.walk(ext_dir):
        for f in filenames:
            rel = os.path.relpath(os.path.join(root, f), ext_dir)
            if ".partial" in f:
                partial_files.append(rel)
            else:
                ext_files.append(os.path.join(root, f))
    if not ext_files:
        print(f"external_dir check: no completed bank files in {ext_dir}")
        return False
    if partial_files:
        print(f"external_dir check: left behind partial files: {partial_files}")
        return False

    # Simulate an interrupted load: delete one completed bank, and plant an
    # unlocked partial file next to it. The next load must clean up the stray,
    # refetch that bank, leave no partial files behind, and match the reference.
    victim = ext_files[0]
    victim_size = os.path.getsize(victim)
    os.remove(victim)
    stray = victim + ".partial.stray"
    with open(stray, "wb"):
        pass

    kept = []
    got_mmap = record_vxpu_engine(model, "main", prompts, settings, artifact, external_dir=ext_dir, keep=kept)
    if not kept:
        print("external_dir check: record_vxpu_engine did not keep engine")
        return False
    reloaded = kept[0]
    if reloaded.external_bytes_fetched != victim_size:
        print(f"external_dir check: reload refetched {reloaded.external_bytes_fetched} bytes; expected {victim_size}")
        return False
    if os.path.exists(stray):
        print(f"external_dir check: stray file was not removed: {stray}")
        return False
    if not os.path.exists(victim):
        print(f"external_dir check: victim bank file was not restored: {victim}")
        return False

    remaining_partials = [os.path.join(root, f) for root, _, filenames in os.walk(ext_dir)
                          for f in filenames if ".partial" in f]
    if remaining_partials:
        print(f"external_dir check: left behind partial files after reload: {remaining_partials}")
        return False

    rows_mmap, ok_mmap = compare(ref, got_mmap, strict_tokens=True)
    print(table(ref, got_mmap, rows_mmap))
    if not ok_mmap:
        return False
    print("  external_dir ok: crash-safe reload refetched only missing bank and matched reference")
    return True


def _check_engram_file(model, prompts, settings, artifact, ref, workdir):
    """The n-gram tables read straight from checkpoint files (#452): with
    local checkpoint files (file://), tokens match the eager reference, and
    no table files are written under external_dir. Under the step graph,
    tables are copied and the plan notes why (#470)."""
    import hashlib
    import shutil
    from .engine import Engine, FileTable

    with open(os.path.join(artifact, "manifest.json")) as f:
        manifest = json.load(f)
    tensors = manifest["tensors"]
    with open(os.path.join(artifact, "binding.json")) as f:
        binding = json.load(f)

    # The expected hashes for engram table tensors if they were written to external_dir
    engram_specs = binding.get("external", {}).get("engram", {})
    table_shas = set()
    for spec in engram_specs.values():
        for k in ("weight", "scale"):
            r = tensors[spec[k]]
            key = hashlib.sha256(f"{r['file_sha256']}:{r['offset']}:{r['length']}".encode()).hexdigest()
            table_shas.add(key)

    ext_dir = os.path.join(workdir, "deepseek-engram-ext")
    shutil.rmtree(ext_dir, ignore_errors=True)

    kept = []
    got = record_vxpu_engine(model, "main", prompts, settings, artifact, external_dir=ext_dir, keep=kept)

    rows, ok = compare(ref, got, strict_tokens=True)
    if not ok:
        print("engram checkpoint reads: tokens changed when reading from checkpoint files")
        return False

    # When CPU kernels are switched off, file-backed table lookups route to host-lookup and raise (#452)
    import torch
    from . import kernels
    kernels.set_enabled(False)
    try:
        engine = kept[0]
        for table in engine.external.engram.values():
            if getattr(table, "file_backed", None) is not None:
                impl = engine.plan.select("vxpu::engram_rows", "cpu", embed=table)
                if impl.name != "host-lookup":
                    print(f"engram checkpoint reads: expected host-lookup when kernels disabled, got {impl.name}")
                    return False
                try:
                    impl.fn(table, torch.tensor([0]))
                except RuntimeError as e:
                    if "this n-gram table is read from its checkpoint file and needs the CPU kernels" not in str(e):
                        raise
                else:
                    print("engram checkpoint reads: lookup did not raise when kernels are disabled")
                    return False
    finally:
        kernels.set_enabled(True)

    ext_files = []
    if os.path.exists(ext_dir):
        for root, _, filenames in os.walk(ext_dir):
            for f in filenames:
                ext_files.append(f)
    leaked_tables = set(ext_files) & table_shas
    if leaked_tables:
        print(f"engram checkpoint reads: table files written to external_dir: {leaked_tables}")
        return False

    notes = got.get("producer", {}).get("plan", {}).get("notes", [])
    engram_notes = [n for n in notes if "n-gram tables: rows read straight from checkpoint files" in n]
    if not engram_notes:
        print(f"engram checkpoint reads: missing plan note about checkpoint reads in {notes}")
        return False

    # Also test step_graph=True: tables are copied and the plan notes why (#470)
    step_ext = os.path.join(workdir, "deepseek-step-ext")
    shutil.rmtree(step_ext, ignore_errors=True)
    step_engine = Engine(artifact, external_dir=step_ext)
    step_engine.plan.choices["step_graph"].value = True
    _, step_engram = step_engine._fetch_external_tensors(
        step_engine.binding["external"], None, step_ext, None, None)
    step_notes = step_engine.plan.notes
    if not any("step graph native packages require resident tensors" in n for n in step_notes):
        print(f"engram checkpoint reads: missing step_graph plan note in {step_notes}")
        return False
    if any(isinstance(v, FileTable) for v in step_engram.values()):
        print(f"engram checkpoint reads: step_graph used FileTable instead of copying: {step_engram}")
        return False

    print(f"  engram checkpoint reads ok: tokens match reference, no table files in external_dir ({engram_notes[0]})")
    return True


def _check_batched_decode(model, artifact, workdir):
    """Batched decode, the measurement prototype: three different prompts of
    one length decoded together by an artifact exported with batch 3 give
    each prompt the tokens it gets alone."""
    from .engine import Engine, SamplingParams
    from .export import export_artifact
    batched_artifact = os.path.join(workdir, "tiny-deepseek-v41-artifact-b3")
    export_artifact(model, batched_artifact, max_cache_len=1024, batch=3)
    greedy = SamplingParams(do_sample=False)
    prompts = [[(7 * i + 3 * j + 11) % 30000 + 100 for j in range(12)] for i in range(3)]
    single = Engine(artifact)
    alone = []
    for prompt in prompts:
        sid = single.new_session()
        alone.append([t for ev in single.generate(sid, prompt, 16, greedy) for t in ev["token_ids"]])
    together = Engine(batched_artifact).generate_batch(prompts, 16, greedy)["token_ids"]
    differ = [i for i, (a, b) in enumerate(zip(alone, together)) if a != b]
    if differ:
        print(f"batched decode check: rows {differ} differ from decoding the prompt alone")
        return False
    print("batched decode: 3 prompts decoded together match each prompt decoded alone")
    return True


def selftest(workdir, settings):
    """The CI tier: tiny models through export, the in-process engine,
    and a transformers reference, token-exact. No cluster."""
    _check_compare()
    _check_mover_lifetime()
    _check_compile_plan()
    _check_plan_given()
    _check_compile_names()
    if not _check_admin_telemetry():
        raise SystemExit("admin telemetry check failed")
    os.makedirs(workdir, exist_ok=True)
    _check_host_reads(workdir)
    _check_local_blobs_copy(workdir)
    _check_expert_sim()
    _check_rotary()
    from .export import export_artifact
    from .fixtures import tiny_fp8_moe
    models = [
        ("tiny-fp8-moe", tiny_fp8_moe.write(os.path.join(workdir, "tiny-fp8-moe")), "ids"),
        ("tiny-llama", "trl-internal-testing/tiny-random-LlamaForCausalLM", "auto"),
    ]
    try:
        from .fixtures import tiny_qwen4  # hybrid linear/indexed attention, needs transformers >= 5.18
        models.append(("tiny-qwen4", tiny_qwen4.write(os.path.join(workdir, "tiny-qwen4")), "ids"))
    except ImportError as e:
        print(f"tiny-qwen4 fixture skipped: {e}")
    try:
        # DeepSeek-V4.1 layout through the model maker's module (third_party fork):
        # fp8 dense, fp4 experts, compressed sparse attention, engram table
        from .fixtures import tiny_deepseek_v41
        models.append(("tiny-deepseek-v41", tiny_deepseek_v41.write(os.path.join(workdir, "tiny-deepseek-v41")), "ids"))
    except ImportError as e:
        print(f"tiny-deepseek-v41 fixture skipped: {e}")
    failed = []
    for name, model, prompts in models:
        artifact = os.path.join(workdir, name + "-artifact")
        export_artifact(model, artifact, max_cache_len=1024)
        assert_no_duplicate_constants(os.path.join(artifact, "decode.pt2"))
        assert_no_duplicate_constants(os.path.join(artifact, "prefill.pt2"))
        if name == "tiny-deepseek-v41":  # no transformers model: the fork run eagerly is the reference
            ref = record_deepseek_eager(model, "main", prompts, settings)
        else:
            ref = record_transformers(model, "main", prompts, settings)
        got = record_vxpu_engine(model, "main", prompts, settings, artifact)
        for label, doc in (("reference", ref), ("candidate", got)):
            with open(os.path.join(workdir, f"{name}-{label}.json"), "w") as f:
                json.dump(doc, f, indent=1)
        with open(os.path.join(artifact, "binding.json")) as f:
            if json.load(f).get("prefill_from_zero"):
                _check_continuation(artifact, ID_PROMPTS[0]["ids"])
        rows, ok = compare(ref, got, strict_tokens=True)
        print(table(ref, got, rows))
        if not ok:
            failed.append(name)
        notes = got.get("producer", {}).get("plan", {}).get("notes", [])
        load_notes = [n for n in notes if n.startswith("load: ")]
        if not load_notes or "probe " not in load_notes[0] or "graph " not in load_notes[0]:
            print(f"load timings check: missing or incomplete load breakdown in plan notes: {notes}")
            failed.append(name + " (load timings)")
        if load_notes and "bound " in load_notes[0] and load_notes[0].index("bound ") > load_notes[0].index("probe "):
            print(f"load timings check: bound fetch should start before probe in plan notes: {notes}")
            failed.append(name + " (load timings: fetch after probe)")
        if load_notes and "bound " in load_notes[0] and "concurrent phases: bound " not in load_notes[0]:
            print(f"load timings check: expected bound fetch and probe under concurrent phases in plan notes: {notes}")
            failed.append(name + " (load timings: bound not concurrent)")
        if name == "tiny-fp8-moe" and not _check_copy_worker_failure(artifact):
            failed.append(name + " (copy failure)")
        if name == "tiny-fp8-moe" and not _check_probe_failure(artifact):
            failed.append(name + " (probe failure)")
        if name == "tiny-fp8-moe" and not _check_local_blobs(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (local weight tier)")
        if not _check_dataflow(model, prompts, settings, artifact, got):
            failed.append(name + " (dataflow)")
        if name == "tiny-deepseek-v41" and not _check_expert_mailboxes(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (expert mailboxes)")
        if name == "tiny-deepseek-v41" and not _check_batched_decode(model, artifact, workdir):
            failed.append(name + " (batched decode)")
        if name == "tiny-deepseek-v41" and not _check_cuts(artifact):
            failed.append(name + " (layer cuts)")
        if name == "tiny-deepseek-v41" and not _check_slices(model, prompts, settings, artifact, got):
            failed.append(name + " (layer ranges)")
        if name == "tiny-deepseek-v41" and not _check_engine_ranges(model, prompts, settings, artifact, got):
            failed.append(name + " (engine ranges)")
        if name == "tiny-deepseek-v41" and not _check_range_server(model, prompts, settings, artifact, got):
            failed.append(name + " (range server)")
        if name == "tiny-deepseek-v41" and _cuda() and not _check_range_server(model, prompts, settings, artifact, got,
                                                                                device="cuda"):
            failed.append(name + " (range server, step graph)")
        if name == "tiny-deepseek-v41" and not _check_native_nodes(model, prompts, settings, artifact, workdir):
            failed.append(name + " (native nodes)")
        if name == "tiny-deepseek-v41" and not _check_remote_nodes(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (remote nodes)")
        if name == "tiny-deepseek-v41" and not _check_node_client():
            failed.append(name + " (node client)")
        if name == "tiny-deepseek-v41" and not _check_step_graph(model, prompts, settings, artifact, workdir):
            failed.append(name + " (step graph)")
        if name == "tiny-deepseek-v41" and not _check_step_graph_pools(model, prompts, settings, artifact, workdir):
            failed.append(name + " (step graph with pools)")
        if name == "tiny-llama" and not _check_chat_telemetry(artifact):
            failed.append(name + " (chat telemetry)")
        if name == "tiny-deepseek-v41" and not _check_gpu_experts(model, prompts, settings, artifact, workdir):
            failed.append(name + " (gpu experts)")
        if name == "tiny-deepseek-v41" and not _check_external_dir(model, prompts, settings, artifact, ref, workdir):
            failed.append(name + " (external_dir)")
        if name == "tiny-deepseek-v41" and not _check_engram_file(model, prompts, settings, artifact, ref, workdir):
            failed.append(name + " (engram checkpoint reads)")
        if name == "tiny-deepseek-v41" and not _check_telemetry(model, prompts, settings, artifact, got, workdir):
            failed.append(name + " (telemetry)")
        if name == "tiny-deepseek-v41" and not _check_lowering(model, prompts, settings, artifact, workdir):
            failed.append(name + " (lowered)")
    _check_dedup_constants(workdir)
    if failed:
        raise SystemExit(f"vectors differ for: {', '.join(failed)}")
    print("selftest ok:", ", ".join(m[0] for m in models))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--new-tokens", type=int, default=DEFAULTS["new_tokens"])
        p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
        p.add_argument("--logit-positions", type=int, default=None,
                       help="positions to record top-k logits for (default: all of --new-tokens)")

    rec = sub.add_parser("record", help="record vectors with a producer")
    rec.add_argument("--producer", choices=["transformers", "vxpu-engine", "vxpu-router", "vllm", "deepseek-eager"], required=True)
    rec.add_argument("--model", required=True, help="Hub repo id or local checkpoint directory")
    rec.add_argument("--revision", default="main")
    rec.add_argument("--prompts", choices=["auto", "text", "ids"], default="auto",
                     help="text prompts through the chat template, or fixed ids (auto: ids if no tokenizer)")
    rec.add_argument("--artifact", help="vxpu artifact directory (vxpu-* producers)")
    rec.add_argument("--router", help="router host:port (vxpu-router; default: port-forward)")
    rec.add_argument("--router-admin",
                     help="router admin host:port (for --producer vxpu-router --telemetry; default: port-forward to port 50052)")
    rec.add_argument("--device", default="cpu", help="transformers / vxpu-engine producers (vllm places itself)")
    rec.add_argument("--dtype", default="auto", help="transformers / vllm model dtype (auto: the config's)")
    rec.add_argument("--compile", action="store_true", help="vxpu-engine: torch.compile the decode graph")
    rec.add_argument("--telemetry", choices=["summary", "trace"], default=None,
                     help="vxpu-engine / vxpu-router: keep execution telemetry ('summary') or with ordered expert traces ('trace')")
    rec.add_argument("--dequantize", action="store_true",
                     help="transformers producer: expand an fp8 checkpoint to bf16 at load")
    rec.add_argument("--max-model-len", type=int, default=None, help="vllm: context length to allocate for")
    rec.add_argument("--enforce-eager", action="store_true",
                     help="vllm: no CUDA graphs / torch.compile")
    rec.add_argument("-o", "--out", required=True)
    common(rec)
    rec.set_defaults(func=cmd_record)

    cmp_ = sub.add_parser("compare", help="compare two vectors files")
    cmp_.add_argument("reference")
    cmp_.add_argument("candidate")
    cmp_.add_argument("--logit-tolerance", type=float, default=0.025,
                      help="relative to the position's largest logit (bf16 rounding is ~1%%)")
    cmp_.set_defaults(func=cmd_compare)

    st = sub.add_parser("selftest", help="CI tier: tiny models, in-process, token-exact")
    st.add_argument("--workdir", default=None)
    common(st)
    st.set_defaults(func=lambda a: selftest(a.workdir or tempfile.mkdtemp(prefix="vxpu-vectors-"),
                                            _settings(a)))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
