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

# Create the dev VM in the first zone with capacity: a g4-standard-96
# (two RTX PRO 6000, 368 GB of RAM) with eight local SSDs, Spot unless
# ON_DEMAND=1. Writes the zone to $ZONE_FILE for the other scripts.

set -uo pipefail
. "$(dirname "$0")/common.sh"

MACHINE=${MACHINE:-g4-standard-96}
GPUS=${GPUS:-2}
LOCAL_SSDS=${LOCAL_SSDS:-8}
ZONES=${ZONES:-"us-central1-b us-central1-c us-central1-f us-east4-b us-east4-c us-east5-a us-east5-b us-east5-c us-east1-b us-east1-d us-south1-a us-south1-b us-west1-a us-west1-b us-west1-c us-west4-a us-west4-b us-west4-c"}

args=(--project "$PROJECT" --machine-type "$MACHINE"
      --accelerator "type=nvidia-rtx-pro-6000,count=$GPUS" --maintenance-policy TERMINATE
      --image-project deeplearning-platform-release --image-family common-cu129-ubuntu-2404-nvidia-580
      --boot-disk-type hyperdisk-balanced --boot-disk-size 300GB
      --scopes cloud-platform --labels purpose=vxpu-dev)
for _ in $(seq "$LOCAL_SSDS"); do args+=(--local-ssd=interface=NVME); done
if [ -n "${SERVICE_ACCOUNT:-}" ]; then args+=(--service-account "$SERVICE_ACCOUNT"); fi
if [ -z "${ON_DEMAND:-}" ]; then args+=(--provisioning-model SPOT --instance-termination-action STOP); fi

for z in $ZONES; do
  echo "trying $z"
  out=$(gcloud compute instances create "$VM" --zone "$z" "${args[@]}" 2>&1)
  if echo "$out" | grep -q RUNNING; then
    echo "$z" > "$ZONE_FILE"
    echo "created $VM in $z (zone written to $ZONE_FILE)"
    exit 0
  fi
  echo "  $(echo "$out" | grep -o "does not have enough resources[^.]*\|Quota [^.]*\|[a-z_]*_availability" | head -1)"
done
echo "no zone in the list had capacity; try ZONES=... or later" >&2
exit 1
