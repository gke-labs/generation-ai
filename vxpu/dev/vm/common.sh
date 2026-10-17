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

# Shared settings for the dev VM scripts (sourced, not run).

VM=${VM:-vxpu-dev}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# VXPU_DIR=... syncs or reads pins from another checkout of the repo
VXPU_DIR=${VXPU_DIR:-$REPO_ROOT/vxpu}
ZONE_FILE=${ZONE_FILE:-$HOME/.vxpu-dev-zone}

zone() {
  if [ -n "${ZONE:-}" ]; then echo "$ZONE"; return; fi
  if [ -f "$ZONE_FILE" ]; then cat "$ZONE_FILE"; return; fi
  echo "no zone: run create.sh first, or set ZONE" >&2
  return 1
}

vm_ssh() {  # vm_ssh 'command'
  gcloud compute ssh "$VM" --zone "$(zone)" --project "$PROJECT" --quiet --command "$1"
}

vm_scp() {  # vm_scp LOCAL REMOTE
  gcloud compute scp "$1" "$VM:$2" --zone "$(zone)" --project "$PROJECT" --quiet
}

# The executor image's pins, so the VM runs what the cluster runs.
pin() {  # pin ARG_NAME
  sed -n "s/^ARG $1=//p" "$VXPU_DIR/images/executor/Dockerfile"
}
