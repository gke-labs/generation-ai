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

# Helper sourced by CI scripts to fetch test models once and serve them through
# a local HTTP mirror with HF_ENDPOINT pointed at it.

HF_MIRROR_PID=""
HF_MIRROR_PORT_FILE=""
HF_MIRROR_HF_HOME=""

stop_hf_mirror() {
  if [ -n "${HF_MIRROR_PID:-}" ]; then
    kill "${HF_MIRROR_PID}" 2>/dev/null || true
    wait "${HF_MIRROR_PID}" 2>/dev/null || true
    HF_MIRROR_PID=""
  fi
  if [ -n "${HF_MIRROR_PORT_FILE:-}" ]; then
    rm -f "${HF_MIRROR_PORT_FILE}"
    HF_MIRROR_PORT_FILE=""
  fi
  if [ -n "${HF_MIRROR_HF_HOME:-}" ]; then
    rm -rf "${HF_MIRROR_HF_HOME}"
    HF_MIRROR_HF_HOME=""
  fi
}

# start_hf_mirror [cache_dir] [python_bin]
# Fetches the CI test model once with retries, starts the local HTTP mirror,
# exports HF_ENDPOINT (and an isolated HF_HOME when not running in Docker), and
# registers an EXIT trap to shut the mirror down cleanly.
start_hf_mirror() {
  local cache_dir="${1:-${REPO_ROOT}/.cache/hf-mirror}"
  local py_bin="${2:-python3}"
  local mirror_script="${REPO_ROOT}/dev/ci/hf_mirror.py"

  "${py_bin}" "${mirror_script}" fetch --cache-dir "${cache_dir}"

  HF_MIRROR_PORT_FILE="$(mktemp "${TMPDIR:-/tmp}/hf-mirror-port.XXXXXX")"
  rm -f "${HF_MIRROR_PORT_FILE}"

  "${py_bin}" "${mirror_script}" serve --cache-dir "${cache_dir}" --host 127.0.0.1 --port 0 --port-file "${HF_MIRROR_PORT_FILE}" &
  HF_MIRROR_PID=$!
  trap stop_hf_mirror EXIT

  local i
  for i in $(seq 1 100); do
    if [ -s "${HF_MIRROR_PORT_FILE}" ]; then
      break
    fi
    if ! kill -0 "${HF_MIRROR_PID}" 2>/dev/null; then
      echo "hf-mirror exited before listening" >&2
      exit 1
    fi
    sleep 0.1
  done
  if [ ! -s "${HF_MIRROR_PORT_FILE}" ]; then
    echo "hf-mirror did not write port file within 10s" >&2
    exit 1
  fi

  local port
  port="$(tr -d '[:space:]' < "${HF_MIRROR_PORT_FILE}")"
  export HF_ENDPOINT="http://127.0.0.1:${port}"
  HF_MIRROR_HF_HOME="$(mktemp -d "${TMPDIR:-/tmp}/hf-mirror-home.XXXXXX")"
  export HF_HOME="${HF_MIRROR_HF_HOME}"
  echo "Local Hugging Face mirror listening at ${HF_ENDPOINT}" >&2
}
