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

# torch JIT-compiles some eager CUDA ops via Triton at runtime,
# which needs a C compiler even without torch.compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# torch is pinned to the exporting torch version: serialized
# ExportedPrograms promise newer-loads-older, but same-version is the
# tested configuration. The default Linux wheels include CUDA, so one
# image serves CPU and GPU nodes.
# transformers comes from the fork under third_party/ (a submodule), so the
# executor runs the same code the exporter traced; see docs/representation.md.
# When the build context has no submodule checkout (CI checks out without
# submodules), the same commit is cloned instead. TRANSFORMERS_COMMIT must
# equal the submodule's gitlink; TestTransformersPinMatchesSubmodule checks.
ARG TRANSFORMERS_REPO=https://github.com/justinsb/transformers
ARG TRANSFORMERS_COMMIT=4096273a3a5b59ca169b2b81afc889ab62478ab8
COPY third_party/transformers/ /src/transformers/
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN if [ -f /src/transformers/pyproject.toml ]; then \
        TRANSFORMERS=/src/transformers; \
    else \
        TRANSFORMERS="git+${TRANSFORMERS_REPO}@${TRANSFORMERS_COMMIT}"; \
    fi \
    && pip install --no-cache-dir \
        torch==2.14.1 \
        "${TRANSFORMERS}" \
        grpcio \
        "protobuf>=6.30" \
        requests \
        huggingface_hub \
        accelerate \
        sympy \
        pillow

# DeepSeek-V4.1's reference module (no transformers implementation exists)
# comes from its fork under third_party/ the same way; the exporter and
# the executor import it from VXPU_DEEPSEEK_V41. DEEPSEEK_V41_COMMIT must
# equal the submodule's gitlink; TestDeepseekPinMatchesSubmodule checks.
ARG DEEPSEEK_V41_REPO=https://github.com/justinsb/deepseek-v41-flash
ARG DEEPSEEK_V41_COMMIT=09ba52944cd1658bc6fddc34543e7eae3084bb2c
COPY third_party/deepseek_v41/ /src/deepseek_v41/
RUN if [ ! -f /src/deepseek_v41/inference/model.py ]; then \
        rm -rf /src/deepseek_v41 \
        && git clone --quiet "${DEEPSEEK_V41_REPO}" /src/deepseek_v41 \
        && git -C /src/deepseek_v41 checkout --quiet "${DEEPSEEK_V41_COMMIT}" \
        && rm -rf /src/deepseek_v41/.git; \
    fi
ENV VXPU_DEEPSEEK_V41=/src/deepseek_v41/inference

COPY python/vxpu/ /app/vxpu/

# The CPU kernels (vxpu/csrc) are a C++ extension torch builds on first
# use; build it here so a pod never compiles. The cache is keyed by the
# sources' hash, so a changed kernel rebuilds at image build time.
ENV TORCH_EXTENSIONS_DIR=/app/.torch_extensions
RUN pip install --no-cache-dir ninja \
    && cd /app && python -c "import vxpu.kernels as k; assert k.available(), k.why_not()"

ENTRYPOINT ["python", "-m", "vxpu.server"]
