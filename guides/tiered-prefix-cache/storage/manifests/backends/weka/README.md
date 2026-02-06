# WEKA CSI StorageClass

This directory contains the StorageClass definition for WEKA CSI driver integration.

## Prerequisites

- WEKA CSI driver installed in your cluster
- WEKA cluster configured and accessible
- CSI secret created (default name: `weka-csi-cluster` in namespace `weka`)

For WEKA CSI driver installation instructions, see the [WEKA CSI Plugin documentation](https://docs.weka.io/appendices/weka-csi-plugin).

## Configuration

Please update the following parameters in [storage_class.yaml](./storage_class.yaml) to match your WEKA cluster configuration:

- `filesystemName`: Your WEKA filesystem name (default: `default`)
- `mountOptions`: Adjust performance parameters as needed for your workload

## Deployment

Deploy the StorageClass:

```bash
kubectl apply -f ./storage_class.yaml
```

This creates a StorageClass named `weka-csi-sc` that will be used by the PVC.

## Cleanup

To remove the StorageClass:

```bash
kubectl delete -f ./storage_class.yaml
```

**Note:** Ensure no PVCs are using this StorageClass before deletion.
