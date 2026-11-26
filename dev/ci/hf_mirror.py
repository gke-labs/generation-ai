#!/usr/bin/env python3
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
# pyright: strict

"""Test and CI-only local HTTP mirror for Hugging Face Hub repositories.

Fetches test repositories once with retries into a local cache directory and
serves them over HTTP with Hugging Face Hub compatible metadata, tree,
paths-info, and byte-range file endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import http.server
import json
import os
import posixpath
import shutil
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, TypedDict, TypeGuard, TypeVar

DEFAULT_REPOS: tuple[str, ...] = ("trl-internal-testing/tiny-random-LlamaForCausalLM",)
DEFAULT_UPSTREAM = "https://huggingface.co"
MIRROR_META_FILE = ".mirror-meta.json"
MIRROR_REVISION = "main"
CHUNK_SIZE = 256 * 1024
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})

_T = TypeVar("_T")


class CachedFile(TypedDict):
    rfilename: str
    size: int
    sha256: str


class RepoMeta(TypedDict):
    repo_id: str
    revision: str
    upstream_sha: str
    files: list[CachedFile]


def _quote_path(path: str) -> str:
    return "/".join(urllib.parse.quote(seg, safe="") for seg in path.split("/"))


def _is_ascii_uint(s: str) -> bool:
    return bool(s) and s.isascii() and s.isdecimal()


def _safe_relpath(raw: str) -> str | None:
    """Validate an already-decoded relative repository file path."""
    trimmed = raw.lstrip("/")
    if not trimmed or chr(0) in trimmed or chr(92) in trimmed:
        return None
    norm = posixpath.normpath(trimmed)
    if norm in ("", ".", "..", MIRROR_META_FILE) or norm.startswith("../"):
        return None
    return norm


def _repo_dir(cache_dir: str, repo_id: str) -> str:
    parts = [p for p in repo_id.split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise ValueError(f"invalid repository id: {repo_id!r}")
    return os.path.join(cache_dir, "repos", *parts)


def _sha256_file(path: str) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_dict(val: object) -> TypeGuard[dict[object, object]]:
    return isinstance(val, dict)


def _is_list(val: object) -> TypeGuard[list[object]]:
    return isinstance(val, list)


def _as_dict(val: object) -> dict[str, Any] | None:
    return {k: v for k, v in val.items() if isinstance(k, str)} if _is_dict(val) else None


def _as_list(val: object) -> list[Any] | None:
    return list(val) if _is_list(val) else None


def _load_valid_meta(cache_dir: str, repo_id: str) -> RepoMeta | None:
    rdir = _repo_dir(cache_dir, repo_id)
    meta_path = os.path.join(rdir, MIRROR_META_FILE)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            raw = _as_dict(json.load(f))
    except (OSError, json.JSONDecodeError):
        return None
    files = _as_list(raw.get("files")) if raw and raw.get("repo_id") == repo_id else None
    if not raw or not files:
        return None
    verified: list[CachedFile] = []
    for entry in files:
        d = _as_dict(entry)
        if not d:
            return None
        name, size, sha = d.get("rfilename"), d.get("size"), d.get("sha256")
        if not isinstance(name, str) or not isinstance(size, int) or not isinstance(sha, str):
            return None
        if _safe_relpath(name) != name:
            return None
        fpath = os.path.join(rdir, name)
        if not os.path.isfile(fpath) or _sha256_file(fpath) != (sha, size):
            return None
        verified.append({"rfilename": name, "size": size, "sha256": sha})
    rev, upstream = raw.get("revision"), raw.get("upstream_sha")
    return {
        "repo_id": repo_id,
        "revision": rev if isinstance(rev, str) else MIRROR_REVISION,
        "upstream_sha": upstream if isinstance(upstream, str) else MIRROR_REVISION,
        "files": verified,
    }


def _with_retries(url: str, fn: Callable[[], _T], *, retries: int, backoff_base: float) -> _T:
    """Run fn with exponential backoff and Retry-After support on transient errors."""
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After") if e.headers else None
            code = e.code
            if code not in RETRYABLE_HTTP_CODES or attempt + 1 >= attempts:
                raise
            e.close()
            delay = min(30.0, backoff_base * (2**attempt))
            if retry_after and _is_ascii_uint(retry_after.strip()):
                delay = min(60.0, max(delay, float(retry_after.strip())))
            reason = f"HTTP {code}"
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as e:
            if attempt + 1 >= attempts:
                raise
            delay = min(30.0, backoff_base * (2**attempt))
            reason = f"{type(e).__name__}: {e}"
        print(f"hf-mirror: retrying {url} after {reason} in {delay:.1f}s ({attempt + 1}/{attempts})", file=sys.stderr, flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _request(url: str, token: str | None) -> urllib.request.Request:
    headers = {"User-Agent": "vxpu-ci-hf-mirror/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _download_sibling(
    base: str,
    repo_id: str,
    upstream_sha: str,
    sib: dict[str, Any],
    staging_dir: str,
    *,
    retries: int,
    timeout: float,
    backoff_base: float,
    token: str | None,
) -> CachedFile | None:
    raw_name = sib.get("rfilename")
    if not isinstance(raw_name, str):
        return None
    rfilename = _safe_relpath(raw_name)
    if rfilename is None:
        raise RuntimeError(f"unsafe file path in metadata: {raw_name!r}")
    lfs = _as_dict(sib.get("lfs"))
    raw_sha = lfs.get("sha256") if lfs else None
    expected_sha = raw_sha if isinstance(raw_sha, str) else None
    lfs_size, sib_size = (lfs.get("size") if lfs else None), sib.get("size")
    expected_size = lfs_size if isinstance(lfs_size, int) else (sib_size if isinstance(sib_size, int) else None)

    url = f"{base}/{_quote_path(repo_id)}/resolve/{urllib.parse.quote(upstream_sha, safe='')}/{_quote_path(rfilename)}"
    dest = os.path.join(staging_dir, rfilename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _download() -> tuple[str, int]:
        with urllib.request.urlopen(_request(url, token), timeout=timeout) as resp:
            digest, written = hashlib.sha256(), 0
            with open(dest, "wb") as out:
                while chunk := resp.read(CHUNK_SIZE):
                    out.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
        sha = digest.hexdigest()
        if expected_size is not None and written != expected_size:
            raise IOError(f"size mismatch for {url}: expected {expected_size}, got {written}")
        if expected_sha is not None and sha != expected_sha:
            raise IOError(f"sha256 mismatch for {url}: expected {expected_sha}, got {sha}")
        return sha, written

    sha, size = _with_retries(url, _download, retries=retries, backoff_base=backoff_base)
    return {"rfilename": rfilename, "size": size, "sha256": sha}


def fetch_repo(
    cache_dir: str,
    repo_id: str,
    *,
    revision: str = MIRROR_REVISION,
    upstream: str = DEFAULT_UPSTREAM,
    retries: int = 5,
    timeout: float = 60.0,
    backoff_base: float = 2.0,
) -> RepoMeta:
    """Fetch repo_id into cache_dir once with retries, reusing valid cached files."""
    if (existing := _load_valid_meta(cache_dir, repo_id)) is not None:
        total = sum(f["size"] for f in existing["files"])
        print(f"hf-mirror: using cached {repo_id} ({len(existing['files'])} files, {total} bytes)", file=sys.stderr, flush=True)
        return existing

    token, base = (os.environ.get("HF_TOKEN") or None), upstream.rstrip("/")
    api_url = f"{base}/api/models/{_quote_path(repo_id)}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    print(f"hf-mirror: fetching {repo_id}@{revision} from {base}", file=sys.stderr, flush=True)

    def _read_meta() -> dict[str, Any]:
        with urllib.request.urlopen(_request(api_url, token), timeout=timeout) as resp:
            info = _as_dict(json.loads(resp.read().decode("utf-8")))
        if info is None:
            raise RuntimeError(f"unexpected metadata payload from {api_url}")
        return info

    info = _with_retries(api_url, _read_meta, retries=retries, backoff_base=backoff_base)
    raw_sha = info.get("sha")
    upstream_sha = raw_sha if isinstance(raw_sha, str) and raw_sha else revision
    siblings = _as_list(info.get("siblings"))
    if not siblings:
        raise RuntimeError(f"no files listed in metadata from {api_url}")

    target_dir = _repo_dir(cache_dir, repo_id)
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".staging-", dir=os.path.dirname(target_dir))
    try:
        cached: list[CachedFile] = []
        for sib in siblings:
            if (d := _as_dict(sib)) is None:
                continue
            if (item := _download_sibling(base, repo_id, upstream_sha, d, staging_dir, retries=retries, timeout=timeout, backoff_base=backoff_base, token=token)) is not None:
                cached.append(item)
        cached.sort(key=lambda item: item["rfilename"])
        meta: RepoMeta = {"repo_id": repo_id, "revision": revision, "upstream_sha": upstream_sha, "files": cached}
        with open(os.path.join(staging_dir, MIRROR_META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write(os.linesep)
        shutil.rmtree(target_dir, ignore_errors=True)
        os.replace(staging_dir, target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    total = sum(f["size"] for f in cached)
    print(f"hf-mirror: cached {repo_id} ({len(cached)} files, {total} bytes)", file=sys.stderr, flush=True)
    return meta


class _MirrorStore:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self._lock = threading.Lock()
        self._metas: dict[str, RepoMeta] = {}

    def get_repo(self, repo_id: str) -> RepoMeta | None:
        with self._lock:
            if repo_id in self._metas:
                return self._metas[repo_id]
        meta_path = os.path.join(_repo_dir(self.cache_dir, repo_id), MIRROR_META_FILE)
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, encoding="utf-8") as f:
            meta: RepoMeta = json.load(f)
        with self._lock:
            self._metas[repo_id] = meta
        return meta

    def file_path(self, repo_id: str, rfilename: str) -> str:
        return os.path.join(_repo_dir(self.cache_dir, repo_id), rfilename)


def _parse_single_range(header: str, total_size: int) -> tuple[int, int] | None:
    if not header.startswith("bytes=") or total_size <= 0:
        return None
    spec = header[len("bytes=") :].strip()
    if "," in spec or "-" not in spec:
        return None
    start_s, end_s = (part.strip() for part in spec.split("-", 1))
    if start_s == "":
        return (max(0, total_size - int(end_s)), total_size - 1) if _is_ascii_uint(end_s) and int(end_s) > 0 else None
    if not _is_ascii_uint(start_s) or int(start_s) >= total_size:
        return None
    start = int(start_s)
    if end_s == "":
        return start, total_size - 1
    if not _is_ascii_uint(end_s):
        return None
    end = min(int(end_s), total_size - 1)
    return (start, end) if start <= end else None


def _hf_entries(meta: RepoMeta) -> list[dict[str, Any]]:
    return [
        {"type": "file", "oid": f["sha256"], "size": f["size"], "path": f["rfilename"], "lfs": {"oid": f["sha256"], "size": f["size"], "pointerSize": 134}}
        for f in meta["files"]
    ]


class _MirrorServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: _MirrorStore) -> None:
        self.store = store
        super().__init__(address, MirrorHandler)


class MirrorHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _store(self) -> _MirrorStore:
        assert isinstance(self.server, _MirrorServer)
        return self.server.store

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_bytes(self, status: int, body: bytes, *, content_type: str = "application/json", headers: dict[str, str] | None = None, head_only: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head_only and body:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any, *, headers: dict[str, str] | None = None, head_only: bool = False) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), headers=headers, head_only=head_only)

    def _send_hub_error(self, code: str, *, head_only: bool = False) -> None:
        self._send_json(404, {"error": code}, headers={"X-Error-Code": code}, head_only=head_only)

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    def do_GET(self) -> None:
        self._dispatch(head_only=False)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        if path.startswith("/api/models/") and "/paths-info/" in path:
            repo_raw, rev_raw = path[len("/api/models/") :].split("/paths-info/", 1)
            self._handle_paths_info(urllib.parse.unquote(repo_raw), urllib.parse.unquote(rev_raw), body)
            return
        self._send_json(404, {"error": "not found"})

    def _dispatch(self, *, head_only: bool) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if path == "/healthz":
            self._send_bytes(200, b"ok" + b"\x0a", content_type="text/plain", head_only=head_only)
            return
        if path.startswith("/api/models/"):
            rest = path[len("/api/models/") :]
            if "/tree/" in rest:
                repo_raw, rev_and_sub = rest.split("/tree/", 1)
                self._handle_tree(urllib.parse.unquote(repo_raw), rev_and_sub, head_only=head_only)
                return
            if "/revision/" in rest:
                repo_raw, rev_raw = rest.split("/revision/", 1)
                self._handle_metadata(urllib.parse.unquote(repo_raw), urllib.parse.unquote(rev_raw), head_only=head_only)
                return
            self._handle_metadata(urllib.parse.unquote(rest), MIRROR_REVISION, head_only=head_only)
            return
        if "/resolve/" in path:
            repo_raw, tail = path.lstrip("/").split("/resolve/", 1)
            if "/" not in tail:
                self._send_hub_error("EntryNotFound", head_only=head_only)
                return
            rev_raw, file_raw = tail.split("/", 1)
            self._handle_resolve(urllib.parse.unquote(repo_raw), urllib.parse.unquote(rev_raw), file_raw, head_only=head_only)
            return
        self._send_json(404, {"error": "not found"}, head_only=head_only)

    def _lookup_repo(self, repo_id: str, revision: str, *, head_only: bool) -> RepoMeta | None:
        try:
            meta = self._store().get_repo(repo_id)
        except ValueError:
            meta = None
        if meta is None:
            self._send_hub_error("RepoNotFound", head_only=head_only)
            return None
        if revision not in (MIRROR_REVISION, meta["revision"], meta["upstream_sha"]):
            self._send_hub_error("RevisionNotFound", head_only=head_only)
            return None
        return meta

    def _handle_metadata(self, repo_id: str, revision: str, *, head_only: bool) -> None:
        if (meta := self._lookup_repo(repo_id, revision, head_only=head_only)) is None:
            return
        siblings = [
            {"rfilename": f["rfilename"], "size": f["size"], "blobId": f["sha256"], "lfs": {"sha256": f["sha256"], "oid": f["sha256"], "size": f["size"], "pointerSize": 134}}
            for f in meta["files"]
        ]
        # Return the moving revision ("main") rather than a 40-hex commit so
        # vxpu.manifest._hub_source records this mirror URL instead of
        # rewriting file sources to https://huggingface.co.
        self._send_json(200, {"id": repo_id, "modelId": repo_id, "sha": MIRROR_REVISION, "private": False, "gated": False, "siblings": siblings}, head_only=head_only)

    def _handle_tree(self, repo_id: str, rev_and_sub: str, *, head_only: bool) -> None:
        rev_raw, _, sub_raw = rev_and_sub.partition("/")
        if sub_raw.strip("/"):
            sub = _safe_relpath(urllib.parse.unquote(sub_raw))
            if sub is None:
                self._send_hub_error("EntryNotFound", head_only=head_only)
                return
        else:
            sub = ""
        if (meta := self._lookup_repo(repo_id, urllib.parse.unquote(rev_raw), head_only=head_only)) is None:
            return
        entries = _hf_entries(meta)
        if sub:
            prefix = sub.rstrip("/") + "/"
            entries = [e for e in entries if e["path"].startswith(prefix)]
        self._send_json(200, entries, head_only=head_only)

    def _handle_paths_info(self, repo_id: str, revision: str, body: bytes) -> None:
        if (meta := self._lookup_repo(repo_id, revision, head_only=False)) is None:
            return
        wanted: set[str] = set()
        if "application/json" in self.headers.get("Content-Type", ""):
            try:
                parsed = _as_dict(json.loads(body.decode("utf-8")))
                for p in (_as_list(parsed.get("paths")) if parsed else None) or []:
                    if isinstance(p, str):
                        wanted.add(p.lstrip("/"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        else:
            for p in urllib.parse.parse_qs(body.decode("utf-8", errors="replace")).get("paths", []):
                wanted.add(p.lstrip("/"))
        self._send_json(200, [e for e in _hf_entries(meta) if e["path"] in wanted])

    def _handle_resolve(self, repo_id: str, revision: str, file_raw: str, *, head_only: bool) -> None:
        if (meta := self._lookup_repo(repo_id, revision, head_only=head_only)) is None:
            return
        rel = _safe_relpath(urllib.parse.unquote(file_raw))
        entry = next((f for f in meta["files"] if f["rfilename"] == rel), None) if rel else None
        if rel is None or entry is None:
            self._send_hub_error("EntryNotFound", head_only=head_only)
            return
        total_size = entry["size"]
        if (range_spec := self._resolve_range(total_size)) is None:
            return
        status, start, end = range_spec
        length = (end - start + 1) if total_size > 0 else 0
        etag = f'"{entry["sha256"]}"'
        headers = {
            "Content-Type": "application/json" if rel.endswith(".json") else "application/octet-stream",
            "Accept-Ranges": "bytes",
            "ETag": etag,
            "X-Linked-Etag": etag,
            "X-Linked-Size": str(total_size),
            "X-Repo-Commit": MIRROR_REVISION,
            "Content-Length": str(length),
        }
        if status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if not head_only and length > 0:
            self._stream_file(self._store().file_path(repo_id, rel), start, length)

    def _resolve_range(self, total_size: int) -> tuple[int, int, int] | None:
        range_header = self.headers.get("Range")
        if range_header is None:
            return 200, 0, total_size - 1
        if (parsed := _parse_single_range(range_header, total_size)) is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        return 206, parsed[0], parsed[1]

    def _stream_file(self, path: str, start: int, length: int) -> None:
        with open(path, "rb") as f:
            if start > 0:
                f.seek(start)
            remaining = length
            while remaining > 0 and (chunk := f.read(min(CHUNK_SIZE, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)


def _cmd_fetch(args: argparse.Namespace) -> int:
    for repo in args.repo or list(DEFAULT_REPOS):
        fetch_repo(args.cache_dir, repo, revision=args.revision, upstream=args.upstream, retries=args.retries, timeout=args.timeout)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    server = _MirrorServer((args.host, args.port), _MirrorStore(args.cache_dir))
    bound_host, bound_port = server.server_address[:2]
    if args.port_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.port_file)), exist_ok=True)
        tmp_path = f"{args.port_file}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(f"{bound_port}{os.linesep}")
        os.replace(tmp_path, args.port_file)

    def _shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    print(f"hf-mirror: serving {args.cache_dir} on http://{bound_host}:{bound_port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Hugging Face Hub HTTP mirror for CI/tests")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch test repositories into the cache directory")
    p_fetch.add_argument("--cache-dir", required=True, help="Directory holding mirrored repos")
    p_fetch.add_argument("--repo", action="append", help="Hugging Face repo id (repeatable)")
    p_fetch.add_argument("--revision", default=MIRROR_REVISION, help="Hub revision to fetch")
    p_fetch.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="Upstream Hub base URL")
    p_fetch.add_argument("--retries", type=int, default=5, help="Retry count on transient errors")
    p_fetch.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_serve = sub.add_parser("serve", help="Serve cached repositories over HTTP")
    p_serve.add_argument("--cache-dir", required=True, help="Directory holding mirrored repos")
    p_serve.add_argument("--host", default="127.0.0.1", help="Listen address")
    p_serve.add_argument("--port", type=int, default=0, help="Listen port (0 chooses a free port)")
    p_serve.add_argument("--port-file", help="Write bound port number to this file once ready")
    p_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
