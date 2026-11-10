# MLX Inference Test

This experiment tests MLX (Machine Learning eXplore) support for LLM inference on both CPU and GPU (using the CUDA backend on Linux).

## Features

- **MLX Support**: Uses `mlx-lm` for loading and running models.
- **CPU vs GPU Benchmarking**: Allows comparing performance between CPU and GPU devices.
- **FunctionGemma**: Specifically tested with `mlx-community/functiongemma-270m-it-6bit`.

## Directory Structure

- `src/`: Python source code.
- `images/`: Dockerfile for the inference image.
- `examples/`: Kubernetes Job manifests.

## Running Locally

To run the inference test locally using Docker:

```bash
# Build the images (passing repository root as context)
docker build -t mlx-inference-cpu -f images/mlx-inference-cpu/Dockerfile $(git rev-parse --show-toplevel)
docker build -t mlx-inference-cuda -f images/mlx-inference-cuda/Dockerfile $(git rev-parse --show-toplevel)

# Run on CPU
docker run --rm mlx-inference-cpu --device cpu

# Run on GPU (requires NVIDIA Container Toolkit)
docker run --rm --gpus all mlx-inference-cuda --device gpu
```

## Running on GKE

Deploy the Kubernetes Jobs:

```bash
kubectl apply -f examples/simple/manifest.yaml
```
