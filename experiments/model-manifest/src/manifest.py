"""Stage A: build a content-addressed model manifest from Hub metadata only.

No weight bytes are downloaded. Sources:
  - config.json                        (architecture description)
  - repo file metadata                 (per-file sha256, computed at upload)
  - safetensors headers, range-fetched (tensor name/dtype/shape/byte-range)

Each tensor is referenced by the 5-tuple
    (file_sha256, byte_offset, byte_length, dtype, shape)
where byte_offset is absolute within the file, so a consumer can fetch a
single tensor with one HTTP range request against any blob store that can
serve the file's content hash. The 4-tuple enrichment (per-tensor sha256)
is a later, one-pass publish-time step; nothing here requires it.
"""

import argparse
import hashlib
import json
import struct
import sys

import requests
from huggingface_hub import HfApi, hf_hub_download

MANIFEST_FORMAT = "model-manifest/v0"


def fetch_range(url: str, start: int, length: int) -> bytes:
    resp = requests.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    resp.raise_for_status()
    return resp.content


def build_manifest(repo_id: str, revision: str = "main") -> dict:
    api = HfApi()
    info = api.model_info(repo_id, revision=revision, files_metadata=True)

    with open(hf_hub_download(repo_id, "config.json", revision=revision)) as f:
        config = json.load(f)

    files = {}
    tensors = {}
    header_bytes_fetched = 0
    shards = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    for shard in shards:
        sha = shard.lfs.sha256
        files[sha] = {
            "size": shard.size,
            "name": shard.rfilename,
            "source": f"https://huggingface.co/{repo_id}/resolve/{revision}/{shard.rfilename}",
        }
        url = files[sha]["source"]
        (header_len,) = struct.unpack("<Q", fetch_range(url, 0, 8))
        header = json.loads(fetch_range(url, 8, header_len))
        header_bytes_fetched += 8 + header_len
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

    manifest = {
        "format": MANIFEST_FORMAT,
        "source": {"repo": repo_id, "revision": revision},
        "config": config,
        "files": files,
        "tensors": tensors,
    }
    manifest["stats"] = {
        "num_files": len(files),
        "num_tensors": len(tensors),
        "total_weight_bytes": sum(f["size"] for f in files.values()),
        "metadata_bytes_fetched": header_bytes_fetched,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    manifest = build_manifest(args.repo_id, args.revision)
    payload = json.dumps(manifest, indent=1, sort_keys=True)
    manifest_sha = hashlib.sha256(payload.encode()).hexdigest()

    out = args.output or args.repo_id.replace("/", "--") + ".manifest.json"
    with open(out, "w") as f:
        f.write(payload)

    s = manifest["stats"]
    print(f"manifest: {out}")
    print(f"  model digest (sha256 of manifest): {manifest_sha}")
    print(f"  {s['num_tensors']} tensors in {s['num_files']} files, "
          f"{s['total_weight_bytes'] / 1e9:.2f} GB of weights")
    print(f"  built from {s['metadata_bytes_fetched']:,} bytes of fetched metadata "
          f"({len(payload):,} byte manifest); weight bytes downloaded: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
