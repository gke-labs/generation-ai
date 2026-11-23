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


"""The part of modelstore's registry API that vxpu writes and checks.

modelstore (../../../modelstore) serves a subset of the OCI distribution
API. A model is a repository whose manifest has one layer per file (the
file's path in the layer's title annotation), the format `modelstore
upload` writes; modelstore's Hub-compatible routes then serve the files
by name (`/<repository>/resolve/<tag>/<path>`). vxpu stores what it
compiles the same way, under its own repository names.
"""

from collections.abc import Mapping
import hashlib
import json
import os
import re

import requests

MEDIA_TYPE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_TYPE_FILE = "application/vnd.gke-labs.model.file.v1"
ARTIFACT_TYPE = "application/vnd.gke-labs.vxpu.artifact.v1+json"
EMPTY_CONFIG = {"mediaType": "application/vnd.oci.empty.v1+json",
                "digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a", "size": 2}
ANNOTATION_TITLE = "org.opencontainers.image.title"
# the model an artifact was compiled from. Keys of vxpu's own: modelstore
# reads its Hub keys (vnd.gke-labs.huggingface.*) on a repository as that
# repository's own identity, and serves the commit as X-Repo-Commit
ANNOTATION_SOURCE_REPO = "vnd.gke-labs.vxpu.source.repo"
ANNOTATION_SOURCE_REVISION = "vnd.gke-labs.vxpu.source.revision"
ANNOTATION_SOURCE_COMMIT = "vnd.gke-labs.vxpu.source.commit"

TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}")  # with fullmatch: modelstore's tag grammar, whole string


def artifact_repository(repo_id):
    """The repository vxpu keeps a model's compiled artifacts in."""
    return "vxpu-artifacts/" + repo_id.lower()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_exists(base_url, name, tag):
    resp = requests.head(f"{base_url.rstrip('/')}/v2/{name}/manifests/{tag}",
                         headers={"Accept": MEDIA_TYPE_MANIFEST}, timeout=60)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def push_directory(
    base_url: str,
    name: str,
    tag: str,
    directory: str,
    annotations: Mapping[str, str] | None = None,
    artifact_type: str = ARTIFACT_TYPE,
) -> str:
    """Push every file under directory as a layer of repository name,
    tagged tag (blobs modelstore already holds are not sent again).
    Returns the manifest's digest."""
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"{tag!r} is not a valid tag")
    base = base_url.rstrip("/")
    layers = []
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, directory).replace(os.sep, "/")
            digest = "sha256:" + _sha256(path)
            size = os.path.getsize(path)
            layers.append({"mediaType": MEDIA_TYPE_FILE, "digest": digest, "size": size,
                           "annotations": {ANNOTATION_TITLE: rel}})
            head = requests.head(f"{base}/v2/{name}/blobs/{digest}", timeout=60)
            if head.status_code == 200:
                continue
            with open(path, "rb") as f:
                resp = requests.post(f"{base}/v2/{name}/blobs/uploads/", params={"digest": digest}, data=f,
                                     headers={"Content-Type": "application/octet-stream",
                                              "Content-Length": str(size)}, timeout=3600)
            if resp.status_code != 201:
                raise RuntimeError(f"uploading {rel} to {name}: HTTP {resp.status_code} {resp.text[:200]}")
    layers.sort(key=lambda layer: layer["annotations"][ANNOTATION_TITLE])
    manifest = {"schemaVersion": 2, "mediaType": MEDIA_TYPE_MANIFEST, "artifactType": artifact_type,
                "config": EMPTY_CONFIG, "layers": layers, "annotations": dict(annotations or {})}
    body = json.dumps(manifest, sort_keys=True).encode()
    resp = requests.put(f"{base}/v2/{name}/manifests/{tag}", data=body,
                        headers={"Content-Type": MEDIA_TYPE_MANIFEST}, timeout=300)
    if resp.status_code != 201:
        raise RuntimeError(f"writing manifest {name}:{tag}: HTTP {resp.status_code} {resp.text[:200]}")
    digest = resp.headers.get("Docker-Content-Digest")
    if digest:
        return digest
    return "sha256:" + hashlib.sha256(body).hexdigest()


PACKAGE_ARTIFACT_TYPE = "application/vnd.gke-labs.vxpu.package.v1"


def package_repository(repo_id: str) -> str:
    """The repository vxpu keeps a model's compiled packages in."""
    return "vxpu-packages/" + repo_id.lower()

