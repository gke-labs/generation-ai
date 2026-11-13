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

FROM ghcr.io/actions/actions-runner:2.338.2@sha256:6d0b2d80d80f143f77dcb676e50920100bbb547e39e883454d57f45ab8c31300

USER root

# Install runtime dependencies for GPU tests:
# - C and C++ toolchain: Triton builds launcher with gcc, csrc/ requires g++
# - ninja-build: required by torch.utils.cpp_extension to compile vxpu/csrc CPU kernels
# - git: repo operations
# - curl: downloads pinned uv binary
# Note: liburing-dev is omitted because vxpu/csrc builds without it.
# No Docker, no cloud SDK, no kubectl, no pre-installed Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

USER runner
