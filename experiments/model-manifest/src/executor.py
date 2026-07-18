"""Stage C: rehydrate the interchange artifact and prove it runs.

Consumes manifest + weightless .pt2 + binding. Fetches each tensor by
HTTP range request into a local content-addressed cache (the stand-in
for an OCI registry pull), computes the config-derived buffers, loads
everything into the exported graph, and checks logits against the
ordinary from_pretrained path.

Acceptance: max |logits_exported - logits_reference| ~ 0.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import requests
import torch

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


def fetch_tensor(ref, files, cas_dir):
    """One ranged read per tensor, cached under a content-derived key.

    Returns (tensor, bytes_downloaded) — 0 if served from cache, which
    also deduplicates aliased tensors (tied embeddings).
    """
    key = hashlib.sha256(
        f"{ref['file_sha256']}:{ref['offset']}:{ref['length']}".encode()
    ).hexdigest()
    path = os.path.join(cas_dir, key)
    downloaded = 0
    if not os.path.exists(path):
        url = files[ref["file_sha256"]]["source"]
        end = ref["offset"] + ref["length"] - 1
        resp = requests.get(url, headers={"Range": f"bytes={ref['offset']}-{end}"})
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        downloaded = ref["length"]
    with open(path, "rb") as f:
        data = bytearray(f.read())
    tensor = torch.frombuffer(data, dtype=DTYPES[ref["dtype"]]).view(ref["shape"])
    return tensor, downloaded


def rope_inv_freq(config):
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    theta = (
        config.rope_parameters["rope_theta"]
        if getattr(config, "rope_parameters", None)
        else config.rope_theta
    )
    exponent = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    return 1.0 / (theta**exponent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph", default="model.pt2")
    parser.add_argument("--cas", default="cas")
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda; the graph is retargeted here")
    parser.add_argument("--verify", action="store_true",
                        help="also run from_pretrained and compare logits")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="verify tolerance; cross-device bf16 needs slack")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.graph + ".binding.json") as f:
        binding = json.load(f)
    os.makedirs(args.cas, exist_ok=True)

    from torch.export.passes import move_to_device_pass

    ep = torch.export.load(args.graph)

    def place(fqn, tensor):
        if fqn in ep.state_dict:
            ep.state_dict[fqn] = tensor
        elif fqn in ep.constants:  # non-persistent buffers live here
            ep.constants[fqn] = tensor
        else:
            raise KeyError(f"{fqn} not found in exported program")

    t0 = time.perf_counter()
    state, bytes_fetched = {}, 0
    for fqn, ref in binding["bound"].items():
        # Native dtype: the graph was traced in the config's dtype (bf16)
        state[fqn], downloaded = fetch_tensor(ref, manifest["files"], args.cas)
        place(fqn, torch.nn.Parameter(state[fqn], requires_grad=False))
        bytes_fetched += downloaded

    # Derived tensors (RoPE frequencies): absent from safetensors,
    # computed from config.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config = CONFIG_MAPPING[manifest["config"]["model_type"]].from_dict(
        manifest["config"]
    )
    for fqn in binding["derived"]:
        state[fqn] = rope_inv_freq(config)
        place(fqn, state[fqn])

    # The recorded example inputs are meta too; materialize placeholders
    # so the device pass can walk them.
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

    # The graph was traced on the meta device (weightless thin client);
    # with real weights in place, retarget it to the executor's device.
    ep = move_to_device_pass(ep, args.device)
    module = ep.module()
    load_s = time.perf_counter() - t0

    input_ids = torch.tensor([[1, 9690, 4093, 198, 105, 720, 11, 28]],
                             device=args.device)
    logits = module(input_ids).cpu()
    print(f"executor[{args.device}]: rehydrated {len(state)} tensors in "
          f"{load_s:.1f}s ({bytes_fetched / 1e6:.0f} MB downloaded by range "
          f"request, rest from CAS)")
    print(f"  logits: shape {tuple(logits.shape)}, "
          f"next-token argmax {int(logits[0, -1].argmax())}")

    if args.verify:
        from transformers import AutoModelForCausalLM

        repo = manifest["source"]["repo"]
        ref_model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.bfloat16
        ).eval()
        with torch.no_grad():
            ref_logits = ref_model(
                input_ids=input_ids.cpu(), use_cache=False
            ).logits
        diff = (logits.float() - ref_logits.float()).abs().max().item()
        argmax_ok = bool(
            (logits[0, -1].argmax() == ref_logits[0, -1].argmax()).item()
        )
        ok = diff < args.atol and argmax_ok
        print(f"  verify vs from_pretrained (cpu reference): "
              f"max |diff| = {diff:.2e}, argmax match = {argmax_ok} "
              f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
