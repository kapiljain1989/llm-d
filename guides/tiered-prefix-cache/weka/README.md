# Well-lit Path: WEKA GPU Direct Storage

This guide demonstrates how to deploy llm-d with WEKA storage using GPU
Direct Storage (GDS) for high-performance data transfer between GPUs and
storage. It covers both PVC and host-path storage configurations.

## Overview

WEKA provides high-performance shared storage with GPU Direct Storage (GDS) support, enabling direct data transfer between GPUs and storage, bypassing CPU and system memory for reduced latency.

The WEKA GDS integration includes:

1. **InitContainers** - Automated setup that loads GDS kernel modules (`nvidia_fs` and `nvidia_peermem`) and creates cufile.json configuration
2. **Volume Mounts** - Mounts cufile.json from `~/amg_stable/cufile.json` on the host to `/etc/cufile.json` in the container
3. **Storage Options** - Supports both PersistentVolumeClaim (PVC) and host-path storage configurations
4. **Startup Probe** - Verifies GDS readiness (kernel modules loaded + cufile.json valid) before starting the container

## Architecture

The manifests use a layered kustomize structure for Prefill/Decode disaggregation:

**Key features:**

- **Prefill/Decode disaggregation**: Separate deployments optimized for each phase
- **Decode** has routing-sidecar for coordinating with prefill instances
- **Prefill** has no routing-sidecar, handles initial prompt processing
- **Storage organized by type**: Choose `pvc/` or `host/` based on your storage setup

## Prerequisites

- Have the [proper client tools installed on your local system](../../prereq/client-setup/README.md) to use this guide
- WEKA storage system configured with:
  - WEKA CSI driver installed (for PVC storage option) - see [WEKA CSI Plugin documentation](https://docs.weka.io/appendices/weka-csi-plugin)
  - WEKA filesystem mounted on nodes (for hostPath storage option)
  - GPU Direct Storage (GDS) enabled:
    - NVIDIA GPUs with GPUDirect Storage capability
    - NVIDIA driver version 450.80.02 or later
    - kernel modules: `nvidia-fs` and `nvidia_peermem` must be available on host
    - WEKA client with GDS support installed
    - **RHEL nodes only**: Install `nvidia-gds` package on the host (`dnf install nvidia-gds-12-9`)
- AMG Utils for GDS configuration:
  - The `amgctl` tool will be run via initContainer to create `~/amg_stable/cufile.json` on each node
  - The file is then mounted into the container at `/etc/cufile.json`
- Create Installation Namespace:

  ```bash
  export NAMESPACE=weka
  kubectl create namespace ${NAMESPACE}
  ```

  **Note:** This guide uses `weka` as the namespace, which is hardcoded in the kustomization files. If you want to use a different namespace, update the `namespace:` field in:
  - `./manifests/vllm/overlays/host/kustomization.yaml`
  - `./manifests/vllm/overlays/pvc/kustomization.yaml`
  - `./manifests/gateway/overlays/istio/kustomization.yaml`

- Gateway API implementation deployed (Istio) - see [Gateway control plane setup](../../prereq/gateway-provider/README.md) if needed

## Installation

**Note:** The example deployment manifests use the `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` model.

### Deploy vLLM Model Servers

Choose either PVC or host-path storage based on your WEKA setup.

#### Option 1: PVC Storage (Recommended)

##### 1. Configure WEKA StorageClass

Configure and deploy the WEKA CSI StorageClass following the [WEKA backend guide](../storage/manifests/backends/weka/README.md).

Set your storage class name for use in the next step:

```bash
export STORAGE_CLASS=weka-csi-sc
```

##### 2. Create the PVC

Create a PersistentVolumeClaim named `wekafs-amg` with 100Gi storage using the storage guide's PVC template:

```bash
envsubst < ../storage/manifests/pvc.yaml | \
  sed -e 's/llm-d-kv-cache-storage/wekafs-amg/' \
      -e 's/18000Gi/100Gi/' | \
  kubectl apply -f - -n ${NAMESPACE}
```

##### 3. Deploy both decode and prefill with PVC storage

   ```bash
   kubectl apply -k ./manifests/vllm/overlays/pvc
   ```

   This creates:

- ServiceAccount: `weka-vllm`
- Deployment `decode`:
  - 1 replica with 4 GPUs (tensor-parallel), 16 CPUs, port 8200
  - InitContainers: `routing-proxy`, `create-cufile-on-node` (amg-utils)
- Deployment `prefill`:
  - 4 replicas, each with 1 GPU, 8 CPUs, port 8000
  - InitContainers: `create-cufile-on-node` (amg-utils)

#### Option 2: Host-Path Storage

1. If WEKA is mounted at a different location than `/mnt/weka`, update the `path` in `./manifests/vllm/overlays/host/kustomization.yaml`:

   ```yaml
   patches:
     - target:
         kind: Deployment
       patch: |-
         - op: replace
           path: /spec/template/spec/volumes/1
           value:
             name: weka-storage
             hostPath:
               path: /mnt/weka  # Replace with your WEKA mount path
               type: Directory
   ```

2. Deploy both decode and prefill with host-path storage

   ```bash
   kubectl apply -k ./manifests/vllm/overlays/host
   ```

   This creates:
   - ServiceAccount: `weka-vllm`
   - Deployment `decode`: 1 replica with 4 GPUs (tensor-parallel), 16 CPUs, port 8200
     - InitContainers: `routing-proxy`, `create-cufile-on-node` (amg-utils)
   - Deployment `prefill`: 4 replicas, each with 1 GPU, 8 CPUs, port 8000
     - InitContainers: `create-cufile-on-node` (amg-utils)

### Deploy InferencePool

Deploy the InferencePool and inference scheduler:

**Note:** You can customize the InferencePool or EndpointPickerConfig by editing `./manifests/inferencepool.values.yaml`.

```bash
helm install weka-vllm \
    -n ${NAMESPACE} \
    -f ./manifests/inferencepool.values.yaml \
    oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool --version v1.3.0
```

This creates:

- InferencePool: `weka-vllm`
- ServiceAccount: `weka-vllm-epp`
- Deployment: `weka-vllm-epp` (runs `llm-d-inference-scheduler` image)
- Service: `weka-vllm-epp`
- ConfigMap: `weka-vllm-epp` (contains EndpointPickerConfig)
- DestinationRule: `weka-vllm-epp` (controller traffic for connection limits and TLS for service `weka-vllm-epp`)
- Role: `weka-vllm-epp`
- RoleBinding: `weka-vllm-epp`

### Deploy Gateway

**Important:** Deploy the Gateway after the InferencePool, as the HTTPRoute references the InferencePool backend (`weka-vllm`).

**Note:** By default, the Gateway service type is `LoadBalancer`. If you want to use `ClusterIP` instead, add the following patch to `./manifests/gateway/overlays/istio/kustomization.yaml` in the `patches:` section before deploying:

```yaml
  - target:
      kind: Gateway
      name: llm-d-inference-gateway
    patch: |-
      - op: add
        path: /metadata/annotations
        value:
          networking.istio.io/service-type: ClusterIP
```

Deploy the Gateway and HTTPRoute resources:

```bash
kubectl apply -k ./manifests/gateway/overlays/istio
```

This creates:

- Gateway: `llm-d-inference-gateway`
- HTTPRoute: `llm-d-route`
- ConfigMap: `llm-d-inference-gateway`

Alternatively, you can manually add the annotation after deployment to change to `ClusterIP`:

```bash
kubectl annotate gateway llm-d-inference-gateway \
  -n ${NAMESPACE} \
  networking.istio.io/service-type=ClusterIP
```
