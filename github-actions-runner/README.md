# Bring-Your-Own GPU GitHub Actions Runner

This directory configures a bring-your-own GitHub Actions runner for this repository using GitHub's Actions Runner Controller (ARC) in runner-scale-set mode. Runner pods execute in a Kubernetes cluster with NVIDIA L4 GPUs. A dedicated GPU presubmit workflow (`.github/workflows/gpu-presubmits.yaml`) dispatches GPU-only tests to this runner for allowlisted authors.

## Security Model

This repository is public. A self-hosted runner executes pull request code that may be hostile. The security model assumes that code running on the runner cannot be trusted and gives it nothing worth taking, bounding what an approved run can do to burning time on an L4.

A container escape through the host kernel or the GPU driver is not excluded. If a job escapes the container to the host node, the impact is bounded by the node pool configuration: the pool is dedicated and tainted, so the node runs no other workload, and its service account possesses no permissions beyond pulling the runner image.

The architecture employs defense-in-depth isolation:

- **No cluster credentials**: The runner pod runs with `automountServiceAccountToken: false` using a dedicated ServiceAccount that has no Role or ClusterRole bindings. The pod cannot authenticate to or communicate with the Kubernetes API.
- **No privilege escalation or root access**: The container runs as non-root (`runAsNonRoot: true`, `runAsUser: 1001`), drops all Linux capabilities (`capabilities: { drop: ["ALL"] }`), and disables privilege escalation (`allowPrivilegeEscalation: false`). This disables `sudo` and prevents jobs from becoming root. Pull request jobs cannot run `apt-get install` at test time; the runner container image carries all required tools. The pod also applies the `RuntimeDefault` seccomp profile.
- **Resource and disk bounds**: The container enforces limits on CPU (4 cores), memory (16 GiB), and ephemeral storage (30 GiB). The work directory `emptyDir` volume enforces `sizeLimit: 30Gi`, sized against an estimated peak usage of ~10–12 GiB (~3 GiB for downloaded CUDA PyTorch wheels, ~5 GiB for the extracted virtual environment and build cache, and ~2–4 GiB for temporary files and caches) to prevent node disk exhaustion.
- **Scoped controller RBAC**: The controller values set `flags.watchSingleNamespace: arc-runners`. ARC does not install a `ClusterRole`. The controller is restricted to managing pods, secrets, service accounts, and roles in `arc-runners`, while the listener pod manages its own ephemeral secrets in `arc-systems`.
- **No cloud identity**: The runner pod has no cloud credentials. The runner node pool must use a Google Cloud service account that possesses no IAM roles beyond pulling images from the container registry, and Workload Identity (GKE metadata server) must be enabled on the pool so the node cannot access other service account tokens.
- **Dedicated, tainted node pool**: The runner pods require a dedicated node pool carrying taint and label `generation-ai.gke.io/runner=true:NoSchedule`. Other cluster workloads cannot schedule onto these nodes, ensuring untrusted runner pods never share a node with sensitive workloads.
- **Strict network isolation**: A Kubernetes NetworkPolicy isolates the `arc-runners` namespace. It allows DNS resolution only to the cluster DNS pods (`kube-dns` in `kube-system`) and outbound HTTPS (TCP 443) only to the public internet. It blocks internal cluster IP ranges (RFC 1918, CGNAT) and the link-local metadata server at `169.254.169.254`.
- **No repository write tokens**: Presubmit jobs run with `contents: read` permissions. The runner does not persist credentials in `.git/config`.
- **Ephemeral runner pods**: Each workflow job executes in a fresh pod that is deleted by ARC immediately upon job completion.
- **Workflow approval boundary**: The `if:` condition in `.github/workflows/gpu-presubmits.yaml` filters runs to allowlisted authors on this repository. However, a pull request from a fork can edit workflow files. The security boundary is the repository setting **Require approval for all external contributors**. The maintainer must enable this setting.

## Preconditions

Before deploying the runner, verify the following prerequisites on your Kubernetes cluster:

### 1. NetworkPolicy Enforcement

The NetworkPolicy requires cluster-level policy enforcement to function. On GKE, verify that Dataplane V2 is enabled:

```bash
gcloud container clusters describe <CLUSTER_NAME> \
  --region=<REGION> \
  --format="value(networkConfig.datapathProvider)"
```

The output must be `ADVANCED_DATAPATH`. If NetworkPolicy enforcement is disabled, network isolation rules will not be applied.

### 2. Dedicated Runner Node Pool

Create a dedicated node pool for runner workloads with NVIDIA L4 GPUs, the runner taint, and the GKE metadata server enabled:

```bash
gcloud container node-pools create vxpu-gpu-runner-pool \
  --cluster=<CLUSTER_NAME> \
  --region=<REGION> \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1,gpu-driver-version=default \
  --node-taints=generation-ai.gke.io/runner=true:NoSchedule \
  --node-labels=generation-ai.gke.io/runner=true \
  --workload-metadata=GKE_METADATA \
  --service-account=<MINIMAL_RUNNER_NODE_SA> \
  --spot
```

The service account `<MINIMAL_RUNNER_NODE_SA>` should only have permission to pull the runner image (e.g. `roles/artifactregistry.reader`).

## Initial Setup

The maintainer performs this setup once.

### 1. Create the GitHub App

Two options exist for registering the runner. The maintainer must choose between registering at the repository level or at the organization level:

- **Option A: Repository-level App**. The App is created for the repository (`gke-labs/generation-ai`). It requires repository permissions for **Actions** (read-only), **Metadata** (read-only), and **Administration** (read and write). This setup requires repository administrator privileges. If the credential leaks, an attacker gains administrative control over repository settings.
- **Option B: Organization-level App with a runner group**. The App is created under the organization (`gke-labs`) and scoped to a dedicated runner group restricted to this repository. It requires organization permissions for **Self-hosted runners** (read and write), **Actions** (read-only), and **Metadata** (read-only), requiring no repository administration privileges. This setup requires organization owner privileges. If the credential leaks, an attacker can register or deregister runners in the organization runner group, but cannot alter repository settings.

The maintainer chooses.

#### Setup for Option A (Repository-level)

1. In GitHub, navigate to **Settings** > **Developer settings** > **GitHub Apps** > **New GitHub App**.
2. Name the App (for example, `generation-ai-gpu-runner`).
3. Set Homepage URL to `https://github.com/gke-labs/generation-ai`.
4. Uncheck **Webhook** > **Active**.
5. Grant the following repository permissions:
   - **Actions**: Read-only
   - **Administration**: Read and write
   - **Metadata**: Read-only
6. Click **Create GitHub App**, note the **App ID**, and generate and download a private key (`.pem` file).
7. Install the App onto `gke-labs/generation-ai` and note the **Installation ID**.

#### Setup for Option B (Organization-level Runner Group)

1. In the GitHub organization settings, navigate to **Developer settings** > **GitHub Apps** > **New GitHub App**.
2. Name the App (for example, `gke-labs-gpu-runner`).
3. Set Homepage URL to `https://github.com/gke-labs`.
4. Uncheck **Webhook** > **Active**.
5. Grant the following permissions:
   - Organization permissions: **Self-hosted runners** (read and write)
   - Repository permissions: **Actions** (read-only), **Metadata** (read-only)
6. Click **Create GitHub App**, note the **App ID**, and generate and download a private key (`.pem` file).
7. In organization **Settings** > **Actions** > **Runner groups**, create a runner group (e.g. `vxpu-gpu-runners`), and restrict its repository access to `gke-labs/generation-ai`.
8. Install the App onto the `gke-labs` organization and note the **Installation ID**.

### 2. Create the Kubernetes Secret

Create the secret in the `arc-runners` namespace using the App credentials (the command is the same for both options):

```bash
kubectl --context "<KUBE_CONTEXT>" create namespace arc-runners

kubectl --context "<KUBE_CONTEXT>" create secret generic arc-github-app-secret \
  --namespace arc-runners \
  --from-literal=github_app_id="<YOUR_APP_ID>" \
  --from-literal=github_app_installation_id="<YOUR_INSTALLATION_ID>" \
  --from-file=github_app_private_key="/path/to/private-key.pem"
```

> **Security Note**: The GitHub App private key can manage runners (and, for Option A, repository configuration). Only cluster operators with read permissions in `arc-runners` can access this Secret (runner pods cannot, having no ServiceAccount token). **Delete the downloaded `.pem` file from your local disk** once the Secret is created.

### 3. Build and Push the Runner Image

Build the runner image tagged by git commit SHA:

```bash
COMMIT_SHA="$(git rev-parse --short HEAD)"
IMAGE="<YOUR_REGISTRY>/actions-runner-gpu:${COMMIT_SHA}"

docker build -t "${IMAGE}" -f github-actions-runner/Dockerfile github-actions-runner/
docker push "${IMAGE}"
```

### 4. Install Controller and Runner Scale Set

The two registration options differ in their configuration URL and runner group parameters:

**For Option A (Repository-level)**:
Run `install` pointing to the repository URL:

```bash
./github-actions-runner/install "<KUBE_CONTEXT>" "${IMAGE}" "https://github.com/gke-labs/generation-ai" "arc-github-app-secret"
```

**For Option B (Organization-level Runner Group)**:
Pass the organization URL and supply the runner group name via Helm `--set runnerGroup="<RUNNER_GROUP_NAME>"` (or set `runnerGroup` in `values-runner-set.yaml`):

```bash
./github-actions-runner/install "<KUBE_CONTEXT>" "${IMAGE}" "https://github.com/gke-labs" "arc-github-app-secret"
```
*(If setting `runnerGroup` at install time via Helm CLI, pass `--set runnerGroup="vxpu-gpu-runners"` to the runner scale set upgrade).*

The install script prints the target context and namespaces before applying any resources.

### 5. Enable External Contributor Workflow Approval

In the GitHub repository:
1. Navigate to **Settings** > **Actions** > **General**.
2. Under **Fork pull request workflows from outside collaborators**, select **Require approval for all external contributors**.
3. Save settings.

### 6. Enable the GPU Presubmit

Navigate to **Settings** > **Secrets and variables** > **Actions** > **Variables**:
- Name: `GPU_RUNNER_ENABLED`
- Value: `true`

Once enabled, pull requests opened by allowlisted authors will run `test-vxpu-gpu`.

## Disabling the Presubmit

To disable the GPU presubmit without deleting runner resources, unset `GPU_RUNNER_ENABLED` or set it to `false`. Workflow runs will automatically skip the GPU presubmit job.

## Removing the Runner

To remove runner resources, run:

```bash
helm --kube-context "<KUBE_CONTEXT>" uninstall vxpu-gpu-l4 --namespace arc-runners
helm --kube-context "<KUBE_CONTEXT>" uninstall arc --namespace arc-systems
kubectl --context "<KUBE_CONTEXT>" delete -f github-actions-runner/networkpolicy.yaml
kubectl --context "<KUBE_CONTEXT>" delete -f github-actions-runner/namespace.yaml
kubectl --context "<KUBE_CONTEXT>" delete namespace arc-systems
```
