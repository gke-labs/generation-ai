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

export PATH=/mnt/ssd/venv/bin:$PATH
export PYTHONPATH=/mnt/ssd/src
export VXPU_PY=/mnt/ssd/src
export VXPU_DEEPSEEK_V41=/mnt/ssd/deepseek_v41/inference
export DEEPSEEK_KERNELS=aten
export TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext
export OMP_NUM_THREADS=16
export VXPU_KERNEL_THREADS=16
export VXPU_EXTERNAL_DIR=/mnt/ssd/external
export VXPU_EXPERTS_IN_RAM=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Scratch volume caches (avoid filling 100 GB root disk)
export TMPDIR=/mnt/ssd/cache/tmp
export HF_HOME=/mnt/ssd/cache/hf
export XDG_CACHE_HOME=/mnt/ssd/cache/xdg
export TORCHINDUCTOR_CACHE_DIR=/mnt/ssd/cache/inductor
export TRITON_CACHE_DIR=/mnt/ssd/cache/triton
export VXPU_AOTI_CACHE_DIR=/mnt/ssd/cache/aoti
export HF_HUB_OFFLINE=1
export VXPU_BLOB_DIR=/mnt/ssd/blobs
