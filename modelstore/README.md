# Model Store

modelstore is the cluster's registry for models: an OCI registry whose
blobs are model files and vxpu graphs, backed by a Google Cloud Storage
bucket. A model's weight files, its exported graphs and its vLLM recipe
share one content-addressed store, and the same sha256 names a file on
the Hugging Face Hub, in a vxpu manifest and as an OCI blob, so nothing
is re-hashed on the way in and executors range-read exactly what the
Hub published.

## What it serves

- **The OCI distribution API** (`/v2/`), a deliberate subset: blob
  `HEAD`/`GET` with HTTP `Range` (the vxpu executor's read path), the
  two-step monolithic upload (`POST` for a session, `PUT` with the
  digest and body) and the single-`POST` form, which is what `oras`
  and `modelstore upload` use, cross-repository mounts (blobs are
  shared by digest), manifest `PUT`/`GET`/`HEAD` for image manifests
  and indexes with every referenced blob checked to exist, and tag
  listing. Chunked uploads (`PATCH`) and deletes answer `405`; `crane`
  pushes blobs with `PATCH` and so cannot push here yet.
- **Imports from the Hugging Face Hub** without the bytes passing
  through a client: `POST /models/import {"repo": "org/name",
  "revision": "main"}` streams every file into the store (LFS files by
  their published sha256, small files hashed on the way), then writes
  an artifact manifest with one layer per file, tagged with the
  revision and the commit it resolved to. `GET /models/<name>` reports
  progress while it runs; the result is durable in the registry. The
  per-blob form, `POST /blobs/<sha>/import {"url": ...}`, is what the
  vxpu router uses. Sources must be `https` on the configured Hub
  host: this pod holds the bucket credentials and is not a general
  fetcher. Set `HF_TOKEN` for gated repositories. A repository whose
  lowercased name is already held by a different repository is refused
  with `409`.
- **Hub-compatible reads**: `/api/models/<repo>` and
  `/<repo>/resolve/<rev>/<file>` are served from the manifest's layer
  annotations, so transformers, vLLM and `huggingface_hub` load from
  the store unchanged with `HF_ENDPOINT=http://modelstore.modelstore`.
- **Plain blob routes**, `GET`/`HEAD`/`PUT /blobs/<sha256>`, the same
  blobs without the registry's naming, for the vxpu router today.

Repository names are the Hub id lowercased (`Qwen/Qwen3-8B-FP8` is
`qwen/qwen3-8b-fp8`); the exact id is in the manifest's annotations.
Manifests and tags live in the bucket next to the blobs
(`_blobs/<sha256>`, `_refs/manifests/...`, `_refs/tags/...`), so a pod
can be replaced freely and a local directory works for kind and tests.

## Deploying

One-time, per project: a regional bucket and workload-identity grants.
Readers (vXPU executors and router, test pods in `default`) get
`objectViewer`; the store gets `objectAdmin`.

```sh
PROJECT=justinsb-knotai-dev
NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
BUCKET=gs://vxpu-$PROJECT-us-central1
POOL=projects/$NUMBER/locations/global/workloadIdentityPools/$PROJECT.svc.id.goog

gcloud storage buckets create $BUCKET --project=$PROJECT --location=us-central1 \
    --uniform-bucket-level-access
# uploads stage under _tmp/ and are removed after publish; this collects
# any left behind by a pod replaced mid-upload (with --blob-store=gs://BUCKET/prefix
# the rule must match "prefix/_tmp/" instead)
gcloud storage buckets update $BUCKET --lifecycle-file=<(echo \
    '{"rule":[{"action":{"type":"Delete"},"condition":{"age":1,"matchesPrefix":["_tmp/"]}}]}')
gcloud storage buckets add-iam-policy-binding $BUCKET \
    --member="principal://iam.googleapis.com/$POOL/subject/ns/modelstore/sa/modelstore" \
    --role=roles/storage.objectAdmin
for sa in ns/default/sa/vxpu-executor ns/default/sa/vxpu-router ns/default/sa/default; do
  gcloud storage buckets add-iam-policy-binding $BUCKET \
      --member="principal://iam.googleapis.com/$POOL/subject/$sa" \
      --role=roles/storage.objectViewer
done
```

Then the image and the manifest:

```sh
gcloud builds submit --config cloudbuild.yaml --project=$PROJECT \
    --substitutions _IMAGE=gcr.io/$PROJECT/generation-ai/modelstore:v1 .
sed -e "s#image: modelstore:latest#image: gcr.io/$PROJECT/generation-ai/modelstore:v1#" \
    -e "s#gs://BUCKET#$BUCKET#" k8s/manifest.yaml | kubectl apply -f -
```

The manifest carries placeholders for the image and the bucket; the
server refuses to start if the bucket is not reachable with the pod's
identity, so a wrong name fails at rollout, not on the first upload.

**Upgrading from the StatefulSet / CRD version.** Earlier manifests ran
modelstore as a StatefulSet with a PVC and kept an index in a `Model`
CRD. `kubectl apply` removes neither; delete them first:

```sh
kubectl -n modelstore delete statefulset modelstore --ignore-not-found
kubectl -n modelstore delete pvc -l app=modelstore --ignore-not-found
kubectl delete crd models.generationai.labs.gke.io --ignore-not-found
```

Blobs on the old PVC are not migrated; re-import or re-upload them.

## Usage

```sh
kubectl -n modelstore port-forward svc/modelstore 8080:80 &
# import a repository (returns at once; poll for progress)
curl -X POST localhost:8080/models/import -d '{"repo": "Qwen/Qwen3-8B-FP8", "revision": "main"}'
curl localhost:8080/models/qwen/qwen3-8b-fp8
# the artifact, as any OCI client sees it
curl localhost:8080/v2/qwen/qwen3-8b-fp8/tags/list
curl localhost:8080/v2/qwen/qwen3-8b-fp8/manifests/main
oras manifest fetch localhost:8080/qwen/qwen3-8b-fp8:main --plain-http
# a file, with a range
curl -r 0-99 localhost:8080/Qwen/Qwen3-8B-FP8/resolve/main/config.json
```

From a pod: `HF_ENDPOINT=http://modelstore.modelstore` and the usual
`from_pretrained("Qwen/Qwen3-8B-FP8")`. From vxpu: `vxpu mirror REPO`
runs the import and follows it.

### Upload from a directory

`modelstore upload --model-name org/name --model-dir DIR [--tag latest]`
hashes every file, pushes the blobs the registry does not have, and
writes the manifest and tag through the registry API.

### Profiling and Benchmarking

`modelstore` serves standard `net/http/pprof` endpoints under `/debug/pprof/`:

```sh
# Fetch a 30s CPU profile from a pod
curl -o cpu.pprof http://modelstore.modelstore/debug/pprof/profile?seconds=30
go tool pprof -http=:8081 cpu.pprof

# Inspect active goroutines and heap allocations
curl http://modelstore.modelstore/debug/pprof/goroutine?debug=1
curl http://modelstore.modelstore/debug/pprof/heap
```

To benchmark blob read throughput with 16 and 64 concurrent readers:

```sh
# Run 16 GB read benchmark against a modelstore blob
./bench.sh http://modelstore.modelstore/blobs/<sha256>

# Run directly against GCS to compare line rates
./bench.sh https://storage.googleapis.com/<bucket>/_blobs/<sha256> "$(gcloud auth print-access-token)"
```

## What comes next

The vxpu artifact (manifest, binding, graphs) joins the weights as a
second manifest under the same tag through an index per repository and
revision, written by an in-cluster converter (`vxpu convert`); the
router then loads a model by reference. See the issue tracker (#247,
#254, #255, #256).
