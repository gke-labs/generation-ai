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
    echo "Kind cluster detected. Skipping inference-test e2e tests as they require GPU."
    exit 0
fi

IMAGE_NAME="inference-test-e2e"
REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "Building Docker image..."
docker build -t "${IMAGE_NAME}" -f images/inference-test/Dockerfile "${REPO_ROOT}"

echo "Running single-node test (CPU)..."
docker run --rm "${IMAGE_NAME}" --model facebook/opt-125m

echo "Running distributed test (2 ranks, CPU, FSDP requested but should fallback)..."
docker run --rm --entrypoint torchrun "${IMAGE_NAME}" --nproc_per_node=2 src/main.py --model facebook/opt-125m --enable-fsdp

echo "E2E Test Passed!"
