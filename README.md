# PointCMT-main

This repository demonstrates the replacement of the Point Cloud branch with the MeshNet branch in PointCMT.

## Overview

PointCMT (Cross-Modality Transformer) is a neural network architecture for 3D data processing. This repository shows how the original Point Cloud processing branch has been replaced with a MeshNet branch for processing 3D mesh data.

## Key Changes

- **Original**: Used `PointCloudBranch` for processing point cloud data (B, N, 3)
- **Updated**: Now uses `MeshNetBranch` for processing mesh data (B, F, C, N)

## Architecture

### Current Implementation (with MeshNet)

```python
from model import PointCMT

# Create model with MeshNet branch
model = PointCMT(num_classes=40, input_channels=6, feature_dim=512)

# Input: mesh data (batch_size, num_faces, channels, neighbors)
mesh_data = torch.randn(2, 512, 6, 3)
output = model(mesh_data)  # Output: (2, 40) classification logits
```

## Files

- `model.py`: Main PointCMT model with MeshNet branch
- `meshnet.py`: MeshNet branch implementation
- `test_model.py`: Test script for the current implementation
- `comparison_example.py`: Detailed comparison showing before/after changes

## Testing

Run the test to verify the MeshNet branch integration:

```bash
python test_model.py
```

Run the comparison example to see the changes:

```bash
python comparison_example.py
```

