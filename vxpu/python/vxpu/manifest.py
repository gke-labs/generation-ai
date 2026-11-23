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

"""Content-addressed model manifests, built from Hub metadata only.

A manifest describes a model without containing it: the architecture
config plus a table binding every tensor to
(file_sha256, byte_offset, byte_length, dtype, shape). It is built from
kilobytes of metadata — per-file sha256 (computed at upload, served by
the Hub API) and range-fetched safetensors headers — never from weight
bytes. The sha256 of the manifest is the model's digest.
"""

import hashlib
import json
import os
import struct
import urllib.parse

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import constants as hf_constants

MANIFEST_FORMAT = "vxpu-manifest/v1alpha1"


def _resolve_local_hub_file(url):
    """Resolves a Hugging Face resolve URL to a local cached file path, or None."""
    if not isinstance(url, str):
        return None
    if "/resolve/" not in url:
        return None
    prefix, rest = url.split("/resolve/", 1)
    if "/" not in rest:
        return None
    commit, filename = rest.split("/", 1)
    parsed = urllib.parse.urlparse(prefix)
    repo_id = parsed.path.lstrip("/")
    if not repo_id or not filename:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(repo_id, filename, revision=commit)
        if path is None:
            path = try_to_load_from_cache(repo_id, filename)
        if path and isinstance(path, str) and os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def _find_cached_snapshot(repo_id, revision="main"):
    """(snapshot_dir, commit_sha) if repo_id is in local Hugging Face cache, else None."""
    try:
        from huggingface_hub import try_to_load_from_cache
        cfg_path = try_to_load_from_cache(repo_id, "config.json", revision=revision)
        if cfg_path is None and revision == "main":
            cfg_path = try_to_load_from_cache(repo_id, "config.json")
        if cfg_path is not None and isinstance(cfg_path, str) and os.path.exists(cfg_path):
            snapshot_dir = os.path.dirname(cfg_path)
            commit_sha = os.path.basename(snapshot_dir)
            return snapshot_dir, commit_sha
    except Exception:
        pass
    return None


def _is_hub_url(url):
    """Returns True if url points to Hugging Face Hub or an external host."""
    try:
        endpoint = hf_constants.ENDPOINT.rstrip("/")
        if url.startswith(endpoint):
            return True
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname in ("127.0.0.1", "localhost", "::1"):
            return False
        return True
    except Exception:
        return True


def _fetch_range(url, start, length):
    local_file = _resolve_local_hub_file(url)
    if local_file is not None:
        with open(local_file, "rb") as f:
            f.seek(start)
            content = f.read(length)
        if len(content) != length:
            raise RuntimeError(
                f"expected {length} bytes from {local_file}, got {len(content)} bytes")
        return content
    if url.startswith("file://"):
        with open(url[len("file://"):], "rb") as f:
            f.seek(start)
            content = f.read(length)
        if len(content) != length:
            raise RuntimeError(
                f"expected {length} bytes from {url}, got {len(content)} bytes")
        return content
    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")
    if offline and _is_hub_url(url):
        raise RuntimeError(f"HF_HUB_OFFLINE is enabled but {url} is not in local cache")
    resp = requests.get(
        url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    resp.raise_for_status()
    if resp.status_code != 206:
        raise RuntimeError(
            f"expected HTTP 206 Partial Content, got {resp.status_code}")
    content = resp.content
    if len(content) != length:
        raise RuntimeError(
            f"expected {length} bytes from {url}, got {len(content)} bytes")
    return content


def _local_shards(directory):
    """(sha256, size, name, file:// source) for each safetensors file in
    a local checkpoint directory (a save_pretrained output)."""
    shards = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(directory, name)
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        shards.append((digest.hexdigest(), os.path.getsize(path), name,
                       "file://" + os.path.abspath(path)))
    return shards


def is_commit(ref):
    """Whether ref is a Hub commit (40 hex digits), which names fixed bytes,
    rather than a ref such as "main" that moves."""
    return isinstance(ref, str) and len(ref) == 40 and all(c in "0123456789abcdef" for c in ref)


def _hub_source(repo_id, commit, filename, endpoint):
    """(recorded source, URL to read) for one of a Hub repository's files.
    The canonical Hub URL is recorded for a commit: a weight store fetches
    a blob it lacks from there, and only a commit names fixed bytes. When
    the endpoint names no commit (modelstore holding a model it did not
    import from the Hub answers with the ref it was asked for), the source
    is the endpoint that was read, not a Hub URL for bytes that may never
    have been on the Hub."""
    read = f"{endpoint}/{repo_id}/resolve/{commit}/{filename}"
    if is_commit(commit):
        return f"https://huggingface.co/{repo_id}/resolve/{commit}/{filename}", read
    return read, read


def build_manifest(repo_id, revision="main"):
    """Build a manifest for a Hugging Face model repo, or for a local
    checkpoint directory (file:// sources; sha256 computed here).

    For a Hub repo this downloads no weight bytes: only file metadata
    and safetensors headers (a few KB per shard, via HTTP range
    requests).
    """
    if os.path.isdir(repo_id):
        directory = os.path.abspath(repo_id)
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        shards = [(sha, size, name, source, source) for sha, size, name, source in _local_shards(directory)]
        commit_sha, revision = "local", "local"
        repo_id = directory
        pinned = False
    else:
        cached_snapshot = _find_cached_snapshot(repo_id, revision=revision)
        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in ("1", "ON", "YES", "TRUE")
        if cached_snapshot is not None:
            snapshot_dir, commit_sha = cached_snapshot
            with open(os.path.join(snapshot_dir, "config.json")) as f:
                config = json.load(f)
            endpoint = hf_constants.ENDPOINT.rstrip("/")
            pinned = is_commit(commit_sha)
            shards = []
            for sha, size, name, local_source in _local_shards(snapshot_dir):
                hub_source, _ = _hub_source(repo_id, commit_sha, name, endpoint)
                shards.append((sha, size, name, hub_source, local_source))
        elif offline:
            raise RuntimeError(f"HF_HUB_OFFLINE is enabled but model {repo_id} is not in local cache")
        else:
            info = HfApi().model_info(repo_id, revision=revision,
                                      files_metadata=True)
            commit_sha = getattr(info, "sha", None) or revision
            with open(hf_hub_download(repo_id, "config.json",
                                      revision=commit_sha)) as f:
                config = json.load(f)
            # The headers are read through the configured endpoint
            # (HF_ENDPOINT), which in a cluster is modelstore's Hub-compatible
            # API; what is recorded as each file's source is _hub_source's.
            endpoint = hf_constants.ENDPOINT.rstrip("/")
            pinned = is_commit(commit_sha)
            shards = [(s.lfs.sha256, s.size, s.rfilename) + _hub_source(repo_id, commit_sha, s.rfilename, endpoint)
                      for s in info.siblings if s.rfilename.endswith(".safetensors")]

    files, tensors = {}, {}
    for sha, size, name, source, read_url in shards:
        files[sha] = {"size": size, "name": name, "source": source}
        (header_len,) = struct.unpack("<Q", _fetch_range(read_url, 0, 8))
        header = json.loads(_fetch_range(read_url, 8, header_len))
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            tensors[name] = {
                "file_sha256": sha,
                "offset": data_start + begin,
                "length": end - begin,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
            }

    return {
        "format": MANIFEST_FORMAT,
        "source": {"repo": repo_id, "revision": revision, "commit_sha": commit_sha, "commit_pinned": pinned},
        "config": config,
        "files": files,
        "tensors": tensors,
    }
