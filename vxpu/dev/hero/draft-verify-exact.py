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

# Hero run 13, the verify step's exactness: the speculative engine with the drafter replaced by (A) perfect drafts, the
# plain run's own next tokens, so nothing is rejected while the tokens agree, and (B) wrong drafts (token 0), so every
# draft is rejected. Under an exact verify both decode the plain run's tokens up to the step's rounding-level near
# ties. Python-driven rows (telemetry=True), so the drafter hook applies. usage: draft-verify-exact.py ARTIFACT PLAIN_TOKENS_JSON LIST N
import json, os, sys, time
artifact, plain_path, lst, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
import vxpu.engine as eng
from vxpu.engine import Engine, SamplingParams
greedy = SamplingParams(do_sample=False)
engine = Engine(artifact, device="cuda", compile_decode=True)
here = os.path.dirname(os.path.abspath(__file__))
texts = [p.strip() for p in open(os.path.join(here, lst)) if p.strip()][:n]
ids = [engine.tokenizer.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)["input_ids"][0].tolist() for t in texts]
plain = json.load(open(plain_path))[os.path.splitext(lst)[0] + "-native"][:n]
block = engine.binding["draft"]["block"]
engine.generate_rows(ids[:1], 4, greedy)
mode = {"kind": "perfect", "seq": None}
def propose(self, rows, ids_, position, count):
    pos = position.tolist()
    out = []
    for row, t in enumerate(ids_):
        if mode["kind"] == "perfect":
            seq = mode["seq"]; p = pos[row]
            d = seq[p + 1:p + 1 + count] if seq is not None and p + 1 < len(seq) else []
            d = d + [0] * (count - len(d))
        else:
            d = [0] * count
        out.append(d)
    rows.confidence = None
    return out
eng._ModelDrafter.propose = propose
def first(x, y): return next((k for k, (a, b) in enumerate(zip(x, y)) if a != b), None)
for kind in ("perfect", "wrong"):
    mode["kind"] = kind
    firsts, acc, dr = [], 0, 0
    t = time.time()
    for prompt, ptoks in zip(ids, plain):
        mode["seq"] = prompt + ptoks
        out = engine.generate_rows([prompt], 64, greedy, telemetry=True)
        firsts.append(first(out["token_ids"][0], ptoks)); acc += out["accepted"]; dr += out["drafted"]
    print("RESULT13V " + json.dumps({"drafts": kind, "artifact": os.path.basename(artifact), "list": lst, "prompts": len(ids),
                                     "identical": sum(1 for f in firsts if f is None), "first_differences": firsts,
                                     "accepted": acc, "drafted": dr, "wall_s": round(time.time() - t, 1)}), flush=True)
print("DONE13V", flush=True)
