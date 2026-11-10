# Gemma 4 31B on vXPU from a GPU-less notebook

[`gemma4-on-vxpu.ipynb`](gemma4-on-vxpu.ipynb) runs the examples from the
Hugging Face [Gemma 4 docs](https://huggingface.co/docs/transformers/en/model_doc/gemma4)
against `google/gemma-4-31B-it` from a machine with no GPU. The docs'
code runs as written with one import changed
(`from vxpu import AutoModelForCausalLM`): `from_pretrained` exports a
weightless artifact and ships it to the vXPU router in a GKE cluster;
`generate()` runs the same torch loop and sampling rules on the
executor pod and returns token ids.

## Why the router matters for this model

| | bf16 weights | fits a 24 GB L4? |
|---|---|---|
| gemma-4-E4B-it | 16 GB | yes → GPU executor |
| gemma-4-31B-it | 62.6 GB | no → CPU executor sized from the manifest (~75 GiB RAM) |

The artifact is the same either way; the router reads the manifest's
weight sizes and decides. Decode on a CPU node is memory-bandwidth bound
(roughly a token per second for a 62 GB model), fine for a notebook,
not for serving — a 4-bit artifact or a larger accelerator (e.g. the
96 GB RTX PRO 6000) is the way to put this model on a GPU.

## Run it

```sh
# images (once, from repository root)
gcloud builds submit --config vxpu/cloudbuild.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-executor:v12 .
gcloud builds submit --config vxpu/cloudbuild-router.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-router:v6 .
cd vxpu
export VXPU_EXECUTOR_IMAGE=gcr.io/$PROJECT/vxpu-executor:v12
export VXPU_ROUTER_IMAGE=gcr.io/$PROJECT/vxpu-router:v6
go build -o bin/vxpu ./cmd/vxpu
bin/vxpu up   # router pod + Service; from_pretrained port-forwards to it

# notebook environment (CPU-only torch is fine; pillow/torchvision are
# for the docs' AutoProcessor, which also wraps the image front-end)
uv venv -p 3.13 .venv && uv pip install -e python/ jupyter pillow torchvision
.venv/bin/jupyter lab examples/gemma4/gemma4-on-vxpu.ipynb
```

Or headless:

```sh
.venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 examples/gemma4/gemma4-on-vxpu.ipynb \
    --output /tmp/gemma4-on-vxpu.executed.ipynb
```

The first load downloads 62.6 GB into the router's cache and again into
the executor's RAM; expect 10–20 minutes. Later loads of the same
artifact are a digest match and take seconds.

## Record a run

`dev/tasks/record-notebook-run` executes the notebook with nbclient and
writes a replayable, self-contained HTML page (plus the raw timestamped
events as JSON): cells rendered like a notebook, outputs appearing on
their real schedule, with speed control and capping of idle gaps so a
ten-minute model load takes seconds to watch. A few kilobytes of text
per run, so it can be hosted anywhere a static file can (#195).

```sh
.venv/bin/python dev/tasks/record-notebook-run examples/gemma4/gemma4-on-vxpu.ipynb --out /tmp/demo
open /tmp/demo/gemma4-on-vxpu.html
```

## What you should see

The committed notebook has no outputs (they are environment-specific
and change with every run); this is the shape of a run against the
31B on the CPU executor, lightly abridged.

Loading, after a one-minute export on first use:

```
executor placed and artifact delivered (515s); rehydrating weights...
model ready in 727s
VxpuModelForCausalLM('google/gemma-4-31B-it', router='127.0.0.1:57298', session='…:s0')
```

The docs' function-calling example, decoded with `skip_special_tokens=False`:

```
<|tool_call>call:get_n_day_weather_forecast{format:<|"|>fahrenheit<|"|>,location:<|"|>San Francisco, CA<|"|>,num_days:3}<tool_call|><|tool_response>
```

and after the tool response is appended and the template re-rendered,
only the new tokens are prefilled (157 of 320) before the model answers
from the tool data:

```
The weather forecast for San Francisco, CA over the next 3 days is:
*   **Day 1:** High of 68°F, low of 54°F, and sunny.
*   **Day 2:** High of 65°F, low of 53°F, with fog in the morning.
*   **Day 3:** High of 62°F, low of 52°F, with light rain.

GenerateStats(finish_reason='eos', prompt_tokens=320, prefilled_tokens=157, generated=106, prefill_ms=5652, ms_per_token=1483.5)
```

Streaming multi-turn chat behaves the same way: the follow-up turn
prefills 99 of 119 prompt tokens and answers at ~1.4 s/token.

## What is and is not covered

- Causal LM, function calling (processor-rendered template with tools,
  parsed tool call, tool response round trip), multi-turn chat with
  `TextStreamer`, and the configuration example run.
- Image and audio examples do not: the artifact captures the text
  decoder path only, and the 31B has no audio backbone.
