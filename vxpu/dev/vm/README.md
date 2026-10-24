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
gcloud compute ssh vxpu-dev --zone ZONE --command '. /mnt/ssd/env.sh && cd /mnt/ssd && python -m vxpu.vectors_test record ...'

# 6. delete it when done (Spot or not, an idle GPU VM costs money)
vxpu/dev/vm/delete.sh
```

`create.sh` writes the zone it found to `~/.vxpu-dev-zone`; the other
scripts read it. `VM` (default `vxpu-dev`) and `PROJECT` override the
defaults. The scripts connect directly to the VM's external IP by default.
Set `VXPU_DEV_IAP=1` to tunnel through Identity-Aware Proxy (IAP) instead.

## As an executor for the router

The VM can also serve models registered by name (`vxpu register`, #357)
as an external executor: the router in the cluster places models on it
and dials it on its internal IP (GKE pods and VMs share the project's
VPC). Its local SSD is its weight tier: the executor reads checkpoint
files from `/mnt/ssd/blobs` and copies the ones it lacks from the weight
bucket first (`VXPU_BLOB_DIR`, `VXPU_BLOB_BUCKET`), so no `fetch.sh` is
needed for models the router sends.

```sh
vxpu/dev/vm/create.sh && vxpu/dev/vm/setup.sh && vxpu/dev/vm/sync.sh
vxpu/dev/vm/executor.sh                       # prints ADDRESS
vxpu up --external-executor ADDRESS           # the router places named models there
vxpu register deepseek-ai/DeepSeek-V4.1-Flash
vxpu ask --model deepseek-ai/DeepSeek-V4.1-Flash "Is the sky blue?"
```

`executor.sh stop` stops it and `executor.sh log` follows its log.

Where SSH to the VM is not possible (a project whose security policy
removes custom firewall rules, so neither direct SSH nor IAP reaches
port 22), `boot.sh` does the same without SSH: it uploads the code and a
bootstrap script to `gs://BUCKET/_dev/$VM/`, sets the VM's startup script
to run it, and resets the VM; the bootstrap runs the setup
(`remote/setup-vm.sh`, the steps `setup.sh` runs over SSH) and starts the
executor as a systemd service, logging to the serial console, which
`boot.sh` follows until the executor listens. Re-run it after a code
change; a reset keeps the local SSDs. The
VM's code comes from `sync.sh`, not an image, so keep it at the commit
of the executor image the router compiles with.

## Notes

- With `VXPU_DEV_IAP=1`, the scripts reach the VM over SSH through IAP,
  which works from any network. `create.sh` tags the VM `vxpu-dev`, and
  the project needs a firewall rule allowing tcp:22 from IAP's range to
  that tag, once: `gcloud compute firewall-rules create vxpu-dev-iap-ssh
  --network default --allow tcp:22 --source-ranges 35.235.240.0/20
  --target-tags vxpu-dev` (IAP still requires IAM permission on the VM).

- The bucket must be readable by the VM's service account (the
  project's default compute service account unless `SERVICE_ACCOUNT` is
  set): `gcloud storage buckets add-iam-policy-binding gs://BUCKET
  --member=serviceAccount:SA --role=roles/storage.objectViewer`.
- Local SSD contents are lost when the VM stops (and when Spot
  preempts it): after a restart run `setup.sh` (it rebuilds the volume
  and the environment) and then `fetch.sh` (minutes from a bucket in the
  same region). A plain reboot keeps the SSDs; `setup.sh` reassembles
  the array without reformatting it.
- `create.sh` refuses to run when a VM named `$VM` already exists in any
  zone, so a re-run cannot leave a second billed VM behind.
- The VM runs with read-only storage scope (`SCOPES` overrides): the
  scripts only read the bucket, and the VM loads pickled artifacts.
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
