# The hero-run instrument

`batches.py` drives N batches of prompts through one engine and prints one `RESULT` JSON line per
batch: the step time, the tokens per second, the aggregate over the batches, and the byte floor of
the step on this hardware, from the artifact's binding and the distinct experts the step selects.
`table.py` turns the lines into the table the design notes use. `hero9-matrix.sh` runs hero 9's
acceptance table on a pod (`common.sh`: the environment, the placement, one process per pod).

On a pod: copy this directory to `/root/hero/hero9`, then `HERO=/root/hero/hero9 SRC=/mnt/ssd/src-h8
/root/hero/hero9/hero9-matrix.sh m1`; results in `/root/hero/hero9/results.{txt,jsonl,md}`.

## Throughput formulas

- `decode_tokens_per_s`: generated tokens over the sum of decode step times in seconds:
  $$\text{decode\_tokens\_per\_s} = \frac{\sum \text{len}(\text{token\_ids})}{\sum \text{step\_ms} / 1000}$$
  For steady steps with one active row, this equals $\frac{\text{tokens\_per\_row\_step}}{\text{median\_step\_ms} / 1000}$ within 1%.
- `aggregate_tokens_per_s`: total generated tokens across all batches over the union of their decode windows:
  $$\text{aggregate\_tokens\_per\_s} = \frac{\text{total\_tokens}}{\text{union}(\text{windows})}$$
  A batch's window is its own decode time ($\sum \text{step\_ms} / 1000$) from the group's start, so for a single batch `aggregate_tokens_per_s` equals `decode_tokens_per_s`.
- `end_to_end_tokens_per_s`: total generated tokens over the full wall duration of the call:
  $$\text{end\_to\_end\_tokens\_per\_s} = \frac{\text{total\_tokens}}{\text{group\_end} - \text{group\_start}}$$

