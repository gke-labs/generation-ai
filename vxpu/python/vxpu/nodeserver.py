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

"""A node server: some of the decode step's dataflow nodes, served native (#394).

Python sets a node server up and then gets out of the way:

  1. the artifact's decode graph, loaded without its weights and split at
     its host ops as the engine splits it, names the nodes (submod_N) and
     gives each node's input shapes;
  2. the nodes' residents (the banks of the experts layers they call, the
     n-gram tables) are loaded as the engine loads them;
  3. each node is lowered and compiled to a native package (nodes.py),
     or found in the package cache: nodes that compute the same thing on
     different residents (every experts layer) share one;
  4. the C++ NodeServer (csrc/node_server.cpp) loads the packages and
     answers the nodes' mailboxes over TCP on its own threads.

From then on no message passes through Python. A message names the node
by its index in the server's list (start returns the names in order) and
carries the node's inputs; the reply carries its outputs. The wire format
is in node_server.cpp; call() here is a client for tests and tools.
"""

import json
import os
import socket
import struct
import types

import torch

from . import aoti, kernels, nodes


def _example_inputs(sub):
    """Inputs of the node's shapes and dtypes (zeros), from its graph."""
    out = []
    for n in sub.graph.nodes:
        if n.op == "placeholder":
            val = n.meta["val"]
            out.append(torch.zeros(tuple(val.shape), dtype=val.dtype))
    return out


def _residents(artifact_dir, subs):
    """The modules the nodes' host ops read, loaded as the engine loads
    them: an External with the experts layers and n-gram tables they
    call, and nothing else."""
    from . import deepseek_v41 as ds
    from .engine import external_files_dir, load_engram_tables, load_expert_layers
    from .plan import Plan
    with open(os.path.join(artifact_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(artifact_dir, "binding.json")) as f:
        binding = json.load(f)
    plan = Plan("cpu")
    if plan["blob_dir"]:  # this machine's local weight tier
        from . import localblobs
        manifest["files"], _ = localblobs.localize(manifest["files"], plan["blob_dir"], plan["blob_bucket"])
    wanted = {"experts": set(), "engram": set()}
    for sub in subs:
        kind, index = nodes.host_call(sub)
        wanted[kind].add(index)
    mm = external_files_dir(plan["external_dir"], manifest)
    external = types.SimpleNamespace(experts={}, engram={})
    specs = binding.get("external") or {}
    if wanted["experts"]:
        args = ds.model_args(manifest["source"]["repo"], int(binding.get("max_cache_len", 1024)))
        external.experts, _ = load_expert_layers(manifest, specs.get("experts", {}), args, None, mm,
                                                 layers=wanted["experts"])
    if wanted["engram"]:
        external.engram, _ = load_engram_tables(manifest, specs.get("engram", {}), None, mm, layers=wanted["engram"])
    return external


def start(artifact_dir, names, port=0, cache_dir=None):
    """Serve the nodes `names` of the artifact's decode step on `port`
    (0: any free one). Returns the kernels' NodeServer (port(), stop());
    a message's node index is the node's position in `names`."""
    k = kernels.load()
    if k is None:
        raise RuntimeError(f"a node server needs the CPU kernels: {kernels.why_not()}")
    split = nodes.decode_split(artifact_dir)
    subs = []
    for name in names:
        sub = getattr(split, name, None)
        if sub is None or not nodes.is_native(sub):
            raise ValueError(f"{name} is not a host node of this artifact's decode step")
        subs.append(sub)
    external = _residents(artifact_dir, subs)
    from .plan import Plan
    cache = aoti.Packages(cache_dir or Plan("cpu")["aoti_cache_dir"])  # the dense segments' cache too
    paths, residents = [], []
    for sub in subs:
        path, res = nodes.native_package(cache, sub, external, _example_inputs(sub))
        paths.append(path)
        residents.append(list(res))
    server = k.NodeServer(port, paths, residents)
    server.external = external  # the residents stay alive as long as the server
    server.packages = paths
    print(f"[vxpu] node server on port {server.port()}: {', '.join(names)} ({len(set(paths))} packages, "
          f"{cache.built} built)", flush=True)
    return server


# ---------------------------------------------------------------------------
# a client (node_server.cpp's wire format), for tests and tools

def _send_tensor(sock, t):
    t = t.contiguous().cpu()
    sock.sendall(struct.pack("<BB", _DTYPE_CODES[t.dtype], t.dim()) + struct.pack(f"<{t.dim()}q", *t.shape))
    sock.sendall(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("the node server closed the connection")
        buf += chunk
    return bytes(buf)


def _recv_tensor(sock):
    code, ndim = struct.unpack("<BB", _recv_exact(sock, 2))
    shape = struct.unpack(f"<{ndim}q", _recv_exact(sock, 8 * ndim)) if ndim else ()
    dtype = _CODE_DTYPES[code]
    t = torch.empty(shape, dtype=dtype)
    data = _recv_exact(sock, t.numel() * t.element_size())
    if data:
        t.view(torch.uint8).view(-1).copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    return t


def call(sock, node, inputs):
    """Send node `node` (its index) its inputs on a connected socket and
    return its outputs."""
    sock.sendall(b"VXN1" + struct.pack("<II", node, len(inputs)))
    for t in inputs:
        _send_tensor(sock, t)
    if _recv_exact(sock, 4) != b"VXR2":
        raise ConnectionError("not a node server reply (VXR2)")
    (status,) = struct.unpack("<I", _recv_exact(sock, 4))
    if status:
        (length,) = struct.unpack("<I", _recv_exact(sock, 4))
        raise RuntimeError(f"node {node}: {_recv_exact(sock, length).decode()}")
    (count,) = struct.unpack("<I", _recv_exact(sock, 4))
    outputs = [_recv_tensor(sock) for _ in range(count)]
    _recv_exact(sock, 16)  # the server's read and run times
    return outputs


def connect(address):
    """A connection to a node server at host:port."""
    host, port = address.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


ECHO_NODE = 0xFFFFFFFF  # replies with the request's tensors (node_server.cpp's kEchoNode)


def echo(address, nbytes=8192, count=1000):
    """Round trips to a node server's echo node through the native client,
    each sending and getting back nbytes (a decode step's experts message
    is ~8 KB): the wire's own cost, with no work. Returns microseconds per
    round trip, sorted."""
    import time
    k = kernels.load()
    if k is None:
        raise RuntimeError(f"the echo needs the CPU kernels: {kernels.why_not()}")
    host, port = address.rsplit(":", 1)
    client = k.NodeClient(host, int(port))
    payload = torch.zeros(nbytes, dtype=torch.uint8)
    for _ in range(min(count, 100)):  # warm the connection
        client.call(ECHO_NODE, [payload])
    times = []
    for _ in range(count):
        t = time.perf_counter()
        client.call(ECHO_NODE, [payload])
        times.append((time.perf_counter() - t) * 1e6)
    return sorted(times)


def main(argv=None):
    """python -m vxpu.nodeserver echo HOST:PORT [--bytes N] [--count N] | serve-echo [--port N]"""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m vxpu.nodeserver")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("echo", help="round trips to a node server's echo node: the wire, with no work")
    e.add_argument("address")
    e.add_argument("--bytes", type=int, default=8192)
    e.add_argument("--count", type=int, default=1000)
    se = sub.add_parser("serve-echo", help="a node server with no nodes, only the echo node, until interrupted")
    se.add_argument("--port", type=int, default=50061)
    args = parser.parse_args(argv)
    if args.cmd == "serve-echo":
        import time
        server = kernels.load().NodeServer(args.port, [], [])
        print(f"[vxpu] echo node server on port {server.port()}", flush=True)
        while True:
            time.sleep(3600)
    times = echo(args.address, args.bytes, args.count)
    pct = {p: times[min(len(times) - 1, int(p / 100 * len(times)))] for p in (50, 90, 99)}
    print(json.dumps({"address": args.address, "bytes": args.bytes, "count": args.count,
                      "us_p50": round(pct[50], 1), "us_p90": round(pct[90], 1), "us_p99": round(pct[99], 1)}))


# c10::ScalarType codes, as the server reads and writes them
_DTYPE_CODES = {torch.uint8: 0, torch.int8: 1, torch.int16: 2, torch.int32: 3, torch.int64: 4, torch.float16: 5,
                torch.float32: 6, torch.float64: 7, torch.bool: 11, torch.bfloat16: 15}
_CODE_DTYPES = {v: k for k, v in _DTYPE_CODES.items()}


if __name__ == "__main__":
    main()
