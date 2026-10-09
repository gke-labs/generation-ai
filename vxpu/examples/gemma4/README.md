# Gemma 4 31B on vXPU from a GPU-less notebook

[`gemma4-on-vxpu.ipynb`](gemma4-on-vxpu.ipynb) runs the examples from the
Hugging Face [Gemma 4 docs](https://huggingface.co/docs/transformers/en/model_doc/gemma4)
against `google/gemma-4-31B-it` from a machine with no GPU: the notebook
exports a weightless artifact, ships it to the vXPU router in a GKE
cluster, and chats over gRPC.

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
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-executor:v8 .
gcloud builds submit --config cloudbuild-router.yaml \
    --substitutions _IMAGE=gcr.io/$PROJECT/vxpu-router:v2 .
export VXPU_EXECUTOR_IMAGE=gcr.io/$PROJECT/vxpu-executor:v8
export VXPU_ROUTER_IMAGE=gcr.io/$PROJECT/vxpu-router:v2
go build -o bin/vxpu ./cmd/vxpu

# notebook environment (CPU-only torch is fine)
uv venv -p 3.13 .venv && uv pip install -e python/ jupyter
.venv/bin/jupyter lab examples/gemma4/gemma4-on-vxpu.ipynb
```

Or headless:

```sh
VXPU_BIN=$PWD/bin/vxpu .venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 examples/gemma4/gemma4-on-vxpu.ipynb \
    --output /tmp/gemma4-on-vxpu.executed.ipynb
```

The first load downloads 62.6 GB into the router's cache and again into
the executor's RAM; expect 10–20 minutes. Later loads of the same
artifact are a digest match and take seconds.

## What is and is not covered

- Causal LM, function calling (via `raw_prompt`: the client renders the
  chat template with tools, the executor tokenizes verbatim), multi-turn
  chat, and the configuration example run.
- Image and audio examples do not: the artifact captures the text
  decoder path only, and the 31B has no audio backbone.
