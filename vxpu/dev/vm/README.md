# A GPU dev VM for engine work

Iterating on the executor's engine and kernels inside Kubernetes costs a
model load per experiment: each pod refetches the weights through
modelstore (14-25 minutes for DeepSeek-V4.1-Flash's 510 GB) and loses
its local SSD when it ends. For engine and kernel work, a plain VM with
the checkpoint on local SSD turns an experiment into minutes:

- the checkpoint is fetched once from GCS with gcloud's parallel client
  (510 GB in about 5 minutes) and read from local NVMe after that;
- the engine reloads in about 3 minutes (the expert banks from local
  files into RAM, the n-gram tables memory-mapped);
- code changes are copied over and run directly, no image build.

The cluster remains the place for end-to-end checks of the router and
executors; the VM is for the engine, the kernels and measurements.

## Steps

```sh
# 1. find capacity and create the VM (Spot by default; ON_DEMAND=1 for on-demand)
vxpu/dev/vm/create.sh

# 2. RAID the local SSDs, install the Python environment at the image's pins
vxpu/dev/vm/setup.sh

# 3. fetch an artifact's weights to local SSD and stage the artifact
vxpu/dev/vm/fetch.sh /path/to/artifact

# 4. copy the working tree's vxpu code (repeat after every change)
vxpu/dev/vm/sync.sh

# 5. run things
gcloud compute ssh vxpu-dev --zone ZONE --command '. /mnt/ssd/env.sh && cd /mnt/ssd && python -m vxpu.vectors record ...'

# 6. delete it when done (Spot or not, an idle GPU VM costs money)
vxpu/dev/vm/delete.sh
```

`create.sh` writes the zone it found to `~/.vxpu-dev-zone`; the other
scripts read it. `VM` (default `vxpu-dev`) and `PROJECT` override the
defaults.

## Notes

- The bucket must be readable by the VM's service account (the
  project's default compute service account unless `SERVICE_ACCOUNT` is
  set): `gcloud storage buckets add-iam-policy-binding gs://BUCKET
  --member=serviceAccount:SA --role=roles/storage.objectViewer`.
- Local SSD contents are lost when the VM stops (and when Spot
  preempts it); `fetch.sh` again takes minutes.
- The environment follows the executor image: torch and the forks of
  transformers and the DeepSeek reference code at the commits
  `vxpu/images/executor/Dockerfile` pins, from the checkout the scripts
  run in (`VXPU_DIR=/other/checkout/vxpu` points the scripts at another
  checkout, for its pins and its code). The CPU kernels, where the code
  has them, build on first use (g++, ninja, and Python headers are
  installed by `setup.sh`).
- Capacity for g4-standard-96 varies by zone and over the day; `ZONES`
  overrides the search list. Cross-region reads from the bucket cost
  egress (about $10 for a 510 GB checkpoint).
