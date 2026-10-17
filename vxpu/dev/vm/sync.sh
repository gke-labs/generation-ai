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

# Copy the working tree's vxpu Python package to the dev VM
# (/mnt/ssd/src/vxpu), replacing what is there. Run after every change.

set -euo pipefail
. "$(dirname "$0")/common.sh"

tarball=$(mktemp -t vxpu-src.XXXXXX)
trap 'rm -f "$tarball"' EXIT
tar --exclude=__pycache__ -czf "$tarball" -C "$VXPU_DIR/python" vxpu
vm_scp "$tarball" /tmp/vxpu-src.tgz
vm_ssh 'rm -rf /mnt/ssd/src/vxpu && mkdir -p /mnt/ssd/src && tar -xzf /tmp/vxpu-src.tgz -C /mnt/ssd/src && echo "synced $(find /mnt/ssd/src/vxpu -name "*.py" | wc -l) files"'
