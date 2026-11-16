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

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Manifest dry-run and schema validation
ns_args=()
if [ -n "${KUBE_NAMESPACE:-}" ]; then
  ns_args=(-n "$KUBE_NAMESPACE")
fi

if command -v kubectl >/dev/null 2>&1; then
  if kubectl config current-context >/dev/null 2>&1; then
    kubectl apply --dry-run=client "${ns_args[@]}" -f "$DIR/pod.yaml"
    kubectl apply --dry-run=server "${ns_args[@]}" -f "$DIR/pod.yaml"
  fi
fi

if command -v kubeconform >/dev/null 2>&1; then
  kubeconform -summary "$DIR/pod.yaml"
elif command -v go >/dev/null 2>&1; then
  GOTOOLCHAIN=auto go run github.com/yannh/kubeconform/cmd/kubeconform@v0.7.0 -summary "$DIR/pod.yaml"
else
  echo "error: neither kubeconform nor go is available to validate pod.yaml" >&2
  exit 1
fi

# 2. env.sh variables check
(
  # shellcheck source=/dev/null
  source "$DIR/env.sh"
  [ "$VXPU_BLOB_DIR" = "/mnt/ssd/blobs" ]
  [ "$VXPU_EXTERNAL_DIR" = "/mnt/ssd/external" ]
  [ "$HF_HOME" = "/mnt/ssd/cache/hf" ]
  [ "$TMPDIR" = "/mnt/ssd/cache/tmp" ]
  [ "$HF_HUB_OFFLINE" = "1" ]
)

# 3. GPU-free containerized bootstrap test (idempotency, stream copy, report missing)
tar -C "$DIR" -cf - . | docker run -i --rm ubuntu:24.04 bash -c '
  set -euo pipefail
  mkdir -p /pod /tmp/test-ssd && tar -C /pod -xf -
  export VXPU_SSD_DIR=/tmp/test-ssd SKIP_PIP_INSTALL=1

  # Simulate PVC mounts (weights and expert-banks) present at pod start
  mkdir -p /tmp/test-ssd/{blobs,external}
  touch /tmp/test-ssd/blobs/checkpoint.bin /tmp/test-ssd/external/bank.bin

  # Initial run: reports missing scratch components and harness data, exiting non-zero
  set +e
  out=$(/pod/bootstrap.sh)
  rc=$?
  set -e
  [ "$rc" -eq 1 ]
  echo "$out" | grep -q "missing 5 component(s)"
  echo "$out" | grep -q "src: /tmp/test-ssd/src"
  echo "$out" | grep -q "deepseek_v41: /tmp/test-ssd/deepseek_v41"
  echo "$out" | grep -q "artifact: /tmp/test-ssd/artifact"
  echo "$out" | grep -q "league-prompts.json: /tmp/test-ssd/league-prompts.json"
  echo "$out" | grep -q "league-code.jsonl: /tmp/test-ssd/league-code.jsonl"

  # Populate sibling directory
  mkdir -p /tmp/sib/{src,deepseek_v41,artifact,cache}
  touch /tmp/sib/src/a.py /tmp/sib/deepseek_v41/b.py /tmp/sib/artifact/c.json /tmp/sib/cache/cached.bin
  echo "prompts" > /tmp/sib/league-prompts.json
  echo "code" > /tmp/sib/league-code.jsonl

  # Sibling stream over loopback: receiver listens in background, sender streams from sibling
  listen_log=$(mktemp)
  POD_IP=127.0.0.1 /pod/bootstrap.sh --listen 19010 >"$listen_log" 2>&1 &
  listen_pid=$!
  for _ in $(seq 1 50); do
    if grep -q "listening on" "$listen_log"; then
      break
    fi
    sleep 0.1
  done
  grep -q "listening on" "$listen_log"
  VXPU_SSD_DIR=/tmp/sib /pod/bootstrap.sh --send 127.0.0.1 19010
  wait "$listen_pid"

  # Idempotent plain run succeeds cleanly when all components are populated
  /pod/bootstrap.sh

  # Check passes when all components are populated
  /pod/bootstrap.sh --check

  # Incomplete sender fails loudly
  if VXPU_SSD_DIR=/tmp/nonexistent /pod/bootstrap.sh --send 127.0.0.1 19010 2>/dev/null; then
    echo "error: send with missing components should fail" >&2
    exit 1
  fi
'
echo "All hero pod tests passed."
