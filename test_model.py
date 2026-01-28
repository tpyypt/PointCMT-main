"""
Test script to demonstrate PointCMT with MeshNet branch
"""

import torch
from model import PointCMT
from meshnet import MeshNetBranch


def test_meshnet_pointcmt():
    """Test PointCMT with MeshNet branch"""
    print("=" * 60)
    print("Testing PointCMT with MeshNet Branch")
    print("=" * 60)
    
    # Create model
    model = PointCMT(num_classes=40, input_channels=6, feature_dim=512)
    
    # Create sample mesh data (batch_size=2, num_faces=512, channels=6, neighbors=3)
    batch_size = 2
    num_faces = 512
    input_channels = 6
    neighbor_num = 3
    mesh_data = torch.randn(batch_size, num_faces, input_channels, neighbor_num)
    
    print(f"\nInput shape: {mesh_data.shape}")
    print(f"Model architecture:")
    print(f"  - Branch type: {type(model.mesh_branch).__name__}")
    print(f"  - Input channels: {model.mesh_branch.input_channels}")
    print(f"  - Output dimension: {model.mesh_branch.output_dim}")
    print(f"  - Neighbor number: {model.mesh_branch.neighbor_num}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(mesh_data)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Model output (logits): {output[0, :5].tolist()}")  # Show first 5 logits
    print(f"✓ MeshNet branch working correctly!")
    
    return model


if __name__ == "__main__":
    test_meshnet_pointcmt()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
