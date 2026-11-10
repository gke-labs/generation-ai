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

# Set up a GPU VM for the vxpu engine, run ON the VM (by setup.sh over
# SSH, or by boot.sh's startup script): the local SSDs as one volume at
# /mnt/ssd, the packages, the Python environment at the executor image's
# pins, and /mnt/ssd/env.sh. Idempotent: an existing array is
# reassembled and mounted, never re-created or reformatted.
#
#   setup-vm.sh TORCH TRANSFORMERS_REPO TRANSFORMERS_COMMIT DEEPSEEK_REPO DEEPSEEK_COMMIT

set -euo pipefail
TORCH=$1 TRANSFORMERS_REPO=$2 TRANSFORMERS_COMMIT=$3 DEEPSEEK_REPO=$4 DEEPSEEK_COMMIT=$5
OWNER=${OWNER:-$(id -un)}
SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=

# local SSDs as one volume, by their stable names (never a device-name
# glob, which can sweep in an attached data disk)
if ! mountpoint -q /mnt/ssd; then
  devs=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | grep -v -- "-part" || true)
  if [ -z "$devs" ]; then echo "no local SSDs under /dev/disk/by-id/google-local-nvme-ssd-*" >&2; exit 1; fi
  md=$($SUDO mdadm --detail --scan 2>/dev/null | awk '{print $2}' | head -1)
  first=$(echo $devs | cut -d" " -f1)
  if [ -z "$md" ] && $SUDO mdadm --examine "$first" >/dev/null 2>&1; then
    $SUDO mdadm --assemble /dev/md0 $devs
    md=/dev/md0
  fi
  if [ -z "$md" ]; then
    $SUDO mdadm --create /dev/md0 --level=0 --raid-devices=$(echo $devs | wc -w) $devs --run
    $SUDO mkfs.ext4 -q -E lazy_itable_init=1,lazy_journal_init=1 /dev/md0
    md=/dev/md0
  fi
  $SUDO mkdir -p /mnt/ssd
  $SUDO mount -o noatime "$md" /mnt/ssd
  $SUDO chown "$OWNER" /mnt/ssd
fi
mkdir -p /mnt/ssd/blobs /mnt/ssd/src /mnt/ssd/artifact
df -h /mnt/ssd | tail -1

# the venv module, a C++ compiler, Python headers and liburing (the CPU kernels build on first use)
if ! dpkg -s python3-venv python3-dev g++ liburing-dev >/dev/null 2>&1; then
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev g++ liburing-dev >/dev/null
fi
echo "packages ok"

# the Python environment at the image's pins
[ -x /mnt/ssd/venv/bin/python ] || python3 -m venv /mnt/ssd/venv
/mnt/ssd/venv/bin/pip install -q --upgrade pip
/mnt/ssd/venv/bin/pip install -q "torch==$TORCH" grpcio 'protobuf>=6.30' requests huggingface_hub accelerate sympy pillow ninja safetensors numpy
/mnt/ssd/venv/bin/pip install -q "git+$TRANSFORMERS_REPO@$TRANSFORMERS_COMMIT"
if [ ! -d /mnt/ssd/deepseek_v41 ]; then git clone -q "$DEEPSEEK_REPO" /mnt/ssd/deepseek_v41; fi
git -C /mnt/ssd/deepseek_v41 fetch -q origin && git -C /mnt/ssd/deepseek_v41 checkout -q "$DEEPSEEK_COMMIT"
echo "python environment ok"

# the run environment, matching the hybrid executor's settings
cores=$(lscpu -p=CORE | grep -v "^#" | sort -u | wc -l)
cat > /mnt/ssd/env.sh <<ENV
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
export HF_HUB_DISABLE_PROGRESS_BARS=1
ENV
echo "wrote /mnt/ssd/env.sh ($cores physical cores)"
