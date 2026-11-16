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

# Bootstrap a hero pod with system packages, a virtualenv, and scratch directories.
# The script can listen for or send components over a pod-to-pod TCP stream.
#
# To restage a recreated hero pod from a surviving sibling:
#   1. Copy the pod scripts to the new pod:
#      tar -C vxpu/dev/hero/pod -cf - . | kubectl exec -i hero-a-0 -- tar -C /mnt/ssd -xf -
#   2. On the new pod (hero-a-0), start the listener:
#      kubectl exec hero-a-0 -- /mnt/ssd/bootstrap.sh --listen 9010
#   3. On the surviving pod (hero-b-0), stream the components:
#      kubectl exec hero-b-0 -- /mnt/ssd/bootstrap.sh --send <hero-a-0 podIP> 9010
#
# Usage:
#   bootstrap.sh
#   bootstrap.sh --listen PORT
#   bootstrap.sh --send IP PORT
#   bootstrap.sh --check
#   bootstrap.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSD="${VXPU_SSD_DIR:-/mnt/ssd}"
REQ="${REQUIREMENTS_FILE:-$SCRIPT_DIR/requirements.txt}"

report_missing() {
  local missing=()
  [ -f "$SSD/env.sh" ] || missing+=("env: $SSD/env.sh")
  [ -x "$SSD/venv/bin/python" ] || missing+=("venv: $SSD/venv/bin/python")
  for d in src deepseek_v41 artifact blobs external; do
    [ -d "$SSD/$d" ] && [ -n "$(ls -A "$SSD/$d" 2>/dev/null)" ] || missing+=("$d: $SSD/$d")
  done
  for f in league-prompts.json league-code.jsonl; do
    [ -f "$SSD/$f" ] && [ -s "$SSD/$f" ] || missing+=("$f: $SSD/$f")
  done
  for c in tmp hf xdg inductor triton aoti; do
    [ -d "$SSD/cache/$c" ] || missing+=("cache: $SSD/cache/$c")
  done

  if [ ${#missing[@]} -eq 0 ]; then
    echo "hero pod: all components present"
    return 0
  fi
  echo "hero pod: missing ${#missing[@]} component(s):"
  for m in "${missing[@]}"; do
    echo "  - $m"
  done
  return 1
}

MODE="bootstrap"
LISTEN_PORT=""
SEND_IP=""
SEND_PORT=""

if [ $# -gt 0 ]; then
  case "$1" in
    --help|-h)
      echo "Usage: $0 [--listen PORT] | [--send IP PORT] | --check | --help"
      echo "Bootstrap packages, a virtual environment, and scratch directories for a hero pod."
      echo "Options:"
      echo "  --listen PORT   Listen on PORT and extract components received from a sibling pod."
      echo "  --send IP PORT  Send components from this pod to IP:PORT on a sibling pod."
      echo "  --check         Print which components are missing."
      echo "  --help          Show this help message."
      exit 0
      ;;
    --check)
      report_missing
      exit $?
      ;;
    --listen)
      MODE="listen"
      if [ $# -lt 2 ] || [ -z "$2" ]; then
        echo "error: --listen requires a PORT argument" >&2
        exit 1
      fi
      LISTEN_PORT="$2"
      ;;
    --send)
      MODE="send"
      if [ $# -lt 3 ] || [ -z "$2" ] || [ -z "$3" ]; then
        echo "error: --send requires IP and PORT arguments" >&2
        exit 1
      fi
      SEND_IP="$2"
      SEND_PORT="$3"
      ;;
    *)
      echo "error: unrecognized argument '$1'" >&2
      echo "Usage: $0 [--listen PORT] | [--send IP PORT] | --check | --help" >&2
      exit 1
      ;;
  esac
fi

if [ "$MODE" = "send" ]; then
  items=(src deepseek_v41 artifact cache league-prompts.json league-code.jsonl)
  for item in "${items[@]}"; do
    if [ ! -e "$SSD/$item" ]; then
      echo "error: sender is missing required component $SSD/$item" >&2
      exit 1
    fi
  done

  echo "hero pod: streaming components to $SEND_IP:$SEND_PORT..."
  tar -C "$SSD" -cf - "${items[@]}" | python3 -c '
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
with s:
    while True:
        chunk = sys.stdin.buffer.read(65536)
        if not chunk: break
        s.sendall(chunk)
s.close()
' "$SEND_IP" "$SEND_PORT"
  echo "hero pod: sent components to $SEND_IP:$SEND_PORT"
  exit 0
fi

SUDO=
if [ "$(id -u)" -ne 0 ]; then
  SUDO=sudo
fi

mkdir -p "$SSD"/{src,deepseek_v41,artifact,torch_ext} \
         "$SSD"/cache/{tmp,hf,xdg,inductor,triton,aoti}
if [ "$SCRIPT_DIR" != "$SSD" ]; then
  for f in env.sh bootstrap.sh requirements.txt; do
    [ -f "$SCRIPT_DIR/$f" ] && cp -p "$SCRIPT_DIR/$f" "$SSD/$f"
  done
fi

pkgs=()
for p in python3-venv python3-dev g++ liburing-dev git rsync curl; do
  dpkg -s "$p" >/dev/null 2>&1 || pkgs+=("$p")
done
if [ ${#pkgs[@]} -gt 0 ]; then
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${pkgs[@]}" >/dev/null
fi

[ -x "$SSD/venv/bin/python" ] || python3 -m venv "$SSD/venv"
if [ -f "$REQ" ] && [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  stamp="$SSD/venv/.req.sha256"
  req_hash=$(sha256sum "$REQ" | awk '{print $1}')
  if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$req_hash" ]; then
    "$SSD/venv/bin/pip" install -q --upgrade pip
    "$SSD/venv/bin/pip" install -q -r "$REQ"
    echo "$req_hash" > "$stamp"
  fi
fi

if [ "$MODE" = "listen" ]; then
  LISTEN_HOST="${POD_IP:-$(hostname -I | awk '{print $1}')}"
  echo "hero pod: listening on $LISTEN_HOST:$LISTEN_PORT..."
  python3 -c '
import socket, sys
port, host = int(sys.argv[1]), sys.argv[2]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host, port))
s.listen(1)
conn, _ = s.accept()
with conn:
    while True:
        chunk = conn.recv(65536)
        if not chunk: break
        sys.stdout.buffer.write(chunk)
s.close()
' "$LISTEN_PORT" "$LISTEN_HOST" | tar -C "$SSD" -xf -
  echo "hero pod: received and extracted components"
fi

report_missing
