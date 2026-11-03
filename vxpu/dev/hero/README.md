# The hero-run instrument

`batches.py` drives N batches of prompts through one engine and prints one `RESULT` JSON line per
batch: the step time, the tokens per second, the aggregate over the batches, and the byte floor of
the step on this hardware, from the artifact's binding and the distinct experts the step selects.
`table.py` turns the lines into the table the design notes use. `hero9-matrix.sh` runs hero 9's
acceptance table on a pod (`common.sh`: the environment, the placement, one process per pod).

On a pod: copy this directory to `/root/hero/hero9`, then `HERO=/root/hero/hero9 SRC=/mnt/ssd/src-h8
/root/hero/hero9/hero9-matrix.sh m1`; results in `/root/hero/hero9/results.{txt,jsonl,md}`.

`league.py` drives the engine using the published benchmark method (sparkDash v1.8.9 "LLM bench", hero run 15).
It runs waves of levels 1, 2, 4, 8, and 16 for prose and code prompts from `league_prompts.json` with 400 tokens and EOS ignored.
It prints one `LEAGUE` JSON line per level with per-stream and aggregate token rates, excluding prefill and the first token.
