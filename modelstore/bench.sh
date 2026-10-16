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

set -o errexit
set -o nounset
set -o pipefail

# Benchmark harness for blob read throughput.
# Measures reading 16 GB in 64 MB ranges with 16 and 64 in flight
# against modelstore or against GCS directly.
#
# Usage:
#   ./bench.sh <URL> [BEARER_TOKEN]
#
# Examples:
#   ./bench.sh http://modelstore.modelstore/blobs/83bb444edc7d4a1ed91ab67ad19620e2ed74724cdd62ede330fcaa7748c64404
#   ./bench.sh https://storage.googleapis.com/my-bucket/_blobs/83bb444edc7d4a1ed91ab67ad19620e2ed74724cdd62ede330fcaa7748c64404 "$(gcloud auth print-access-token)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <URL> [BEARER_TOKEN]" >&2
  exit 1
fi

URL="$1"
TOKEN="${2:-}"

TOTAL_GB="${TOTAL_GB:-16}"
RANGE_MB="${RANGE_MB:-64}"
RANGE_SIZE=$((RANGE_MB * 1024 * 1024))
TOTAL_RANGES=$((TOTAL_GB * 1024 / RANGE_MB))

CURL_ARGS=("-s" "-o" "/dev/null")
if [[ -n "${TOKEN}" ]]; then
  CURL_ARGS+=("-H" "Authorization: Bearer ${TOKEN}")
fi

echo "Benchmarking ${URL}"
echo "Total data: ${TOTAL_GB} GB in ${TOTAL_RANGES} ranges of ${RANGE_MB} MB each"

# Generate range specifications
RANGES_FILE=$(mktemp)
trap 'rm -f "${RANGES_FILE}"' EXIT

# Cycle across a 4 GB window if the object is large enough
OBJECT_WINDOW=$((4 * 1024 * 1024 * 1024))
for ((i = 0; i < TOTAL_RANGES; i++)); do
  START=$(((i * RANGE_SIZE) % OBJECT_WINDOW))
  END=$((START + RANGE_SIZE - 1))
  echo "${START}-${END}" >> "${RANGES_FILE}"
done

for CONCURRENCY in 16 64; do
  START_TIME=$(date +%s%N)
  xargs -P "${CONCURRENCY}" -a "${RANGES_FILE}" -I {} curl "${CURL_ARGS[@]}" -r "{}" "${URL}"
  END_TIME=$(date +%s%N)

  NANOS=$((END_TIME - START_TIME))
  DURATION_SEC=$(awk "BEGIN {print ${NANOS} / 1000000000}")
  THROUGHPUT=$(awk "BEGIN {printf \"%.2f\", ${TOTAL_GB} / ${DURATION_SEC}}")

  echo "Concurrency ${CONCURRENCY}: ${TOTAL_GB} GB in ${DURATION_SEC}s -> ${THROUGHPUT} GB/s"
done
