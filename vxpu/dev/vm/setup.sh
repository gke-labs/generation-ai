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

# Prepare the dev VM: RAID0 its local SSDs at /mnt/ssd, install the
# packages the engine and its kernels need, and create a Python
# environment at the executor image's pins. Writes /mnt/ssd/env.sh.
# Re-running skips what is already done.

set -euo pipefail
. "$(dirname "$0")/common.sh"

TORCH=$(sed -n 's/^ *torch==\([0-9.]*\).*/\1/p' "$VXPU_DIR/images/executor/Dockerfile" | head -1)
TRANSFORMERS_REPO=$(pin TRANSFORMERS_REPO)
TRANSFORMERS_COMMIT=$(pin TRANSFORMERS_COMMIT)
DEEPSEEK_REPO=$(pin DEEPSEEK_V41_REPO)
DEEPSEEK_COMMIT=$(pin DEEPSEEK_V41_COMMIT)
echo "torch $TORCH, transformers $TRANSFORMERS_COMMIT, deepseek_v41 $DEEPSEEK_COMMIT"

# local SSDs as one volume
vm_ssh 'set -e
if ! mountpoint -q /mnt/ssd; then
  devs=$(ls /dev/nvme*n1 | grep -v nvme0n1)
  sudo mdadm --create /dev/md0 --level=0 --raid-devices=$(echo $devs | wc -w) $devs --run >/dev/null 2>&1
  sudo mkfs.ext4 -q -F -E lazy_itable_init=1,lazy_journal_init=1 /dev/md0
  sudo mkdir -p /mnt/ssd
  sudo mount -o noatime /dev/md0 /mnt/ssd
  sudo chown "$USER" /mnt/ssd
fi
mkdir -p /mnt/ssd/blobs /mnt/ssd/src /mnt/ssd/artifact
df -h /mnt/ssd | tail -1'

# the venv module, a C++ compiler and Python headers (the CPU kernels build on first use)
vm_ssh 'sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev g++ >/dev/null
echo "packages ok"'

# the Python environment at the image's pins
vm_ssh "set -e
[ -x /mnt/ssd/venv/bin/python ] || python3 -m venv /mnt/ssd/venv
/mnt/ssd/venv/bin/pip install -q --upgrade pip
/mnt/ssd/venv/bin/pip install -q torch==$TORCH grpcio 'protobuf>=6.30' requests huggingface_hub accelerate sympy pillow ninja safetensors numpy
/mnt/ssd/venv/bin/pip install -q 'git+$TRANSFORMERS_REPO@$TRANSFORMERS_COMMIT'
if [ ! -d /mnt/ssd/deepseek_v41 ]; then git clone -q $DEEPSEEK_REPO /mnt/ssd/deepseek_v41; fi
git -C /mnt/ssd/deepseek_v41 fetch -q origin && git -C /mnt/ssd/deepseek_v41 checkout -q $DEEPSEEK_COMMIT
echo 'python environment ok'"

# the run environment, matching the hybrid executor's settings
vm_ssh 'cores=$(lscpu -p=CORE | grep -v "^#" | sort -u | wc -l)
cat > /mnt/ssd/env.sh <<EOF
export PATH=/mnt/ssd/venv/bin:\$PATH
export PYTHONPATH=/mnt/ssd/src
export VXPU_PY=/mnt/ssd/src
export VXPU_DEEPSEEK_V41=/mnt/ssd/deepseek_v41/inference
export DEEPSEEK_KERNELS=aten
export TORCH_EXTENSIONS_DIR=/mnt/ssd/torch_ext
# one thread per physical core for torch and for the kernels: the
# kernels are memory-bound and oversubscription costs them most
export OMP_NUM_THREADS=$cores
export VXPU_KERNEL_THREADS=$cores
export VXPU_EXTERNAL_DIR=/mnt/ssd/external
export VXPU_EXPERTS_IN_RAM=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
EOF
echo "wrote /mnt/ssd/env.sh ($cores physical cores)"'
