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
# images (once)
cd vxpu
gcloud builds submit --config cloudbuild.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-executor:v12 .
gcloud builds submit --config cloudbuild-router.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-router:v6 .
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

## What is and is not covered

- Causal LM, function calling (processor-rendered template with tools,
  parsed tool call, tool response round trip), multi-turn chat with
  `TextStreamer`, and the configuration example run.
- Image and audio examples do not: the artifact captures the text
  decoder path only, and the 31B has no audio backbone.
