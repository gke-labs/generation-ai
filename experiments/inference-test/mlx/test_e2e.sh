#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "$(kubectl config current-context)" == *"kind-"* ]]; then
    echo "Kind cluster detected. Skipping MLX inference-test e2e tests as they require GPU."
    exit 0
fi

IMAGE_NAME="mlx-inference-cpu"
REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "Building MLX Docker image..."
docker build -t "${IMAGE_NAME}" -f images/mlx-inference-cpu/Dockerfile "${REPO_ROOT}"

echo "Running MLX inference test (CPU)..."
# Using a very small model if possible for testing, or just the default.
# FunctionGemma 270m is small enough.
docker run --rm "${IMAGE_NAME}" --model mlx-community/functiongemma-270m-it-6bit --device cpu

echo "MLX E2E Test Passed!"
