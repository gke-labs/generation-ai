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

FROM python:3.13-slim

WORKDIR /app

# torch 2.13 dispatches some eager CUDA ops through Triton, which
# JIT-compiles kernels at runtime and requires a C compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Torch is pinned to the exact version that exported the .pt2 artifact:
# serialized ExportedPrograms are not guaranteed portable across torch
# versions. The default Linux wheels include CUDA support, so this one
# image runs on both CPU and GPU nodes.
RUN pip install --no-cache-dir \
    torch==2.13.0 \
    transformers==5.14.0 \
    requests

COPY src/ /app/src/

ENTRYPOINT ["python", "src/executor.py"]
