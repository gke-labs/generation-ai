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

# Hero run 13, the drafter's acceptance by position: one artifact, greedy, 64 new tokens, every prompt of each list
# decoded alone, on the native loop and on the Python-driven rows (telemetry=True takes that path), with the engine's
# per-position speculation statistics summed over the prompts. One RESULT13 line per (list, path); the tokens written
# beside the script. usage: draft-acceptance.py LABEL ARTIFACT PROMPTS_FILE[,PROMPTS_FILE...] [NEW_TOKENS]
import json, os, statistics, sys, time
label, artifact, lists = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
new_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64
from vxpu.engine import Engine, SamplingParams
greedy = SamplingParams(do_sample=False)
t0 = time.time()
engine = Engine(artifact, device="cuda", compile_decode=True)
load_s = time.time() - t0
here = os.path.dirname(os.path.abspath(__file__))
def encode(path):
    texts = [p.strip() for p in open(path) if p.strip()]
    return [engine.tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                                 return_tensors="pt", return_dict=True)["input_ids"][0].tolist() for t in texts]
first = encode(os.path.join(here, lists[0]))
engine.generate_rows(first[:1], 4, greedy)
t_warm = time.time()
while time.time() - t_warm < 4.0:
    engine.generate_rows(first[:1], 64, greedy)
paths = ["native", "python"] if engine.step_tokens > 1 else ["native"]
tokens_out = {}
for name in lists:
    ids = encode(os.path.join(here, name))
    for path in paths:
        positions = {}
        drafted = accepted = steps = 0
        step_ms = []
        toks = []
        t = time.time()
        for prompt in ids:
            out = engine.generate_rows([prompt], new_tokens, greedy, telemetry=(path == "python"))
            toks.append(out["token_ids"][0])
            step_ms += out["step_ms"]
            drafted += out.get("drafted", 0); accepted += out.get("accepted", 0)
            spec = out.get("speculation") or {}
            steps += spec.get("steps", 0)
            for p in spec.get("positions", []):
                e = positions.setdefault(p["position"], {"issued": 0, "accepted": 0, "conf_acc": 0.0, "conf_rej": 0.0})
                e["issued"] += p["issued"]; e["accepted"] += p["accepted"]
                e["conf_acc"] += (p.get("confidence_accepted") or 0.0) * p["accepted"]
                e["conf_rej"] += (p.get("confidence_rejected") or 0.0) * (p["issued"] - p["accepted"])
        wall = time.time() - t
        rates = {k: {"issued": v["issued"], "accepted": v["accepted"], "rate": round(v["accepted"] / max(1, v["issued"]), 3),
                     "conf_accepted": round(v["conf_acc"] / max(1, v["accepted"]), 3),
                     "conf_rejected": round(v["conf_rej"] / max(1, v["issued"] - v["accepted"]), 3)} for k, v in sorted(positions.items())}
        cond = []
        prev = None
        for k, v in rates.items():
            cond.append(round(v["rate"] / prev, 3) if prev else v["rate"]); prev = v["rate"] or 1e-9
        gen = sum(len(x) for x in toks)
        rec = {"label": f"{label}-{os.path.splitext(name)[0]}-{path}", "artifact": os.path.basename(artifact), "prompts": len(ids),
               "path": path, "step_tokens": engine.step_tokens, "load_s": round(load_s, 1), "new_tokens": new_tokens,
               "steps": len(step_ms), "median_step_ms": round(statistics.median(step_ms), 1) if step_ms else None,
               "tokens": gen, "wall_tok_s": round(gen / wall, 1), "drafted": drafted, "accepted": accepted,
               "acceptance": round(accepted / drafted, 3) if drafted else None,
               "tokens_per_step": round(gen / len(step_ms), 3) if step_ms else None,
               "per_position": rates, "conditional": cond}
        print("RESULT13 " + json.dumps(rec), flush=True)
        tokens_out[f"{os.path.splitext(name)[0]}-{path}"] = toks
json.dump(tokens_out, open(os.path.join(here, f"{label}-tokens.json"), "w"))
print("DONE13", label, flush=True)
