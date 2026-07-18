"""Stage C': rehydrate the cached graphs and generate text.

Loads prefill.pt2 + decode.pt2, places weights (shared between the two
programs) and zero-initialized cache state, retargets to the device,
torch.compiles the decode step on arrival, and runs the token loop.

The decode graph has constant shapes (StaticCache), so compilation
happens once — the first step pays the prepare, every later step reuses
it.
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.export.passes import move_to_device_pass

from executor import DTYPES, fetch_tensor, rope_inv_freq  # noqa: F401


def rehydrate(graph_path, binding, manifest, config, tensors, device):
    """Load a .pt2, place shared tensors into it, return its module."""
    ep = torch.export.load(graph_path)

    def place(fqn, tensor):
        if fqn in ep.state_dict:
            ep.state_dict[fqn] = (
                torch.nn.Parameter(tensor, requires_grad=False)
                if not tensor.is_floating_point() or tensor.dim() > 0
                else tensor
            )
        elif fqn in ep.constants:
            ep.constants[fqn] = tensor
        # A graph may not lift every tensor (prefill vs decode differ).

    for fqn in binding["bound"]:
        place(fqn, tensors[fqn])
    for fqn in binding["derived"]:
        place(fqn, tensors[fqn])
    for fqn in binding["state"]:
        place(fqn, tensors[fqn])

    ex_args, ex_kwargs = ep.example_inputs
    fix = lambda t: (
        torch.zeros(t.shape, dtype=t.dtype)
        if isinstance(t, torch.Tensor) and t.is_meta
        else t
    )
    ep._example_inputs = (
        tuple(fix(t) for t in ex_args),
        {k: fix(v) for k, v in ex_kwargs.items()},
    )
    ep = move_to_device_pass(ep, device)
    return ep


def strip_asserts(module):
    """Remove _assert_tensor_metadata nodes: eager-mode sanity checks
    that Dynamo cannot re-trace on CUDA. ep.module() regenerates them
    in submodules (input guards), so strip every graph in the tree."""
    removed = 0
    for _, sub in module.named_modules():
        if not hasattr(sub, "graph"):
            continue
        for node in list(sub.graph.nodes):
            if node.op == "call_function" and "_assert_tensor_metadata" in str(
                node.target
            ):
                sub.graph.erase_node(node)
                removed += 1
        sub.recompile()
    return removed


def share_state(src_module, dst_module, state_fqns):
    """Alias cache/state buffers so prefill and decode see one cache.

    move_to_device_pass copies tensors per-program, so after building
    both modules we re-point the decode module's state buffers at the
    prefill module's tensors.
    """
    src = dict(src_module.named_buffers())
    shared = 0
    for fqn in state_fqns:
        if fqn in src:
            parent, name = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
            dst_parent = dst_module.get_submodule(parent) if parent else dst_module
            if name in dst_parent._buffers:
                dst_parent._buffers[name] = src[fqn]
                shared += 1
    return shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--binding", default="cached.binding.json")
    parser.add_argument("--prompt", default="Why is the sky blue?")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the decode step on arrival")
    parser.add_argument("--cas", default="cas")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.binding) as f:
        binding = json.load(f)
    os.makedirs(args.cas, exist_ok=True)

    from transformers import AutoTokenizer
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config = CONFIG_MAPPING[manifest["config"]["model_type"]].from_dict(
        manifest["config"]
    )

    # Materialize every tensor once; both graphs share these objects.
    t0 = time.perf_counter()
    tensors, downloaded = {}, 0
    for fqn, ref in binding["bound"].items():
        tensors[fqn], dl = fetch_tensor(ref, manifest["files"], args.cas)
        downloaded += dl
    for fqn in binding["derived"]:
        tensors[fqn] = rope_inv_freq(config)

    # State: shapes/dtypes come from the decode program itself.
    ep_probe = torch.export.load("decode.pt2")
    probe_state = {**ep_probe.state_dict, **ep_probe.constants}
    for fqn in binding["state"]:
        meta = probe_state[fqn]
        tensors[fqn] = torch.zeros(meta.shape, dtype=meta.dtype)

    prefill = rehydrate("prefill.pt2", binding, manifest, config,
                        tensors, args.device).module()
    decode = rehydrate("decode.pt2", binding, manifest, config,
                       tensors, args.device).module()
    if args.compile:
        strip_asserts(decode)
    shared = share_state(prefill, decode, binding["state"])
    load_s = time.perf_counter() - t0
    print(f"generator[{args.device}]: rehydrated 2 graphs in {load_s:.1f}s "
          f"({downloaded / 1e6:.0f} MB downloaded, {shared} state buffers "
          f"shared)")

    if args.compile:
        decode = torch.compile(decode)

    tokenizer = AutoTokenizer.from_pretrained(manifest["source"]["repo"])
    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    )["input_ids"].to(args.device)
    prompt_len = input_ids.shape[1]

    t1 = time.perf_counter()
    logits = prefill(
        input_ids=input_ids,
        cache_position=torch.arange(prompt_len, device=args.device),
    )
    prefill_s = time.perf_counter() - t1

    token_ids, step_times = [], []
    position = prompt_len
    next_id = int(logits[0, -1].argmax())
    for _ in range(args.max_new_tokens):
        if next_id == tokenizer.eos_token_id:
            break
        token_ids.append(next_id)
        t2 = time.perf_counter()
        logits = decode(
            input_ids=torch.tensor([[next_id]], device=args.device),
            cache_position=torch.tensor([position], device=args.device),
        )
        step_times.append(time.perf_counter() - t2)
        next_id = int(logits[0, -1].argmax())
        position += 1

    steady = step_times[3:] or step_times
    print(f"  prompt: {args.prompt!r} ({prompt_len} tokens, prefill "
          f"{prefill_s * 1e3:.0f} ms)")
    print(f"  first steps: "
          + ", ".join(f"{t * 1e3:.0f}" for t in step_times[:3]) + " ms; "
          f"steady state {sum(steady) / len(steady) * 1e3:.1f} ms/token "
          f"({len(steady) / sum(steady):.1f} tok/s)")
    print(f"  answer: {tokenizer.decode(token_ids, skip_special_tokens=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
