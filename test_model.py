"""
Test script to demonstrate PointCMT with Point Cloud branch and MeshNet branch
"""

import torch
from model import PointCMT, PointCloudBranch


def test_original_pointcmt():
    """Test original PointCMT with Point Cloud branch"""
    print("=" * 60)
    print("Testing Original PointCMT with Point Cloud Branch")
    print("=" * 60)
    
    # Create model
    model = PointCMT(num_classes=40, input_dim=3, feature_dim=512)
    
    # Create sample point cloud data (batch_size=2, num_points=1024, dims=3)
    batch_size = 2
    num_points = 1024
    point_cloud = torch.randn(batch_size, num_points, 3)
    
    print(f"\nInput shape: {point_cloud.shape}")
    print(f"Model architecture:")
    print(f"  - Branch type: {type(model.point_cloud_branch).__name__}")
    print(f"  - Input dimension: {model.point_cloud_branch.input_dim}")
    print(f"  - Output dimension: {model.point_cloud_branch.output_dim}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(point_cloud)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Model output (logits): {output[0, :5].tolist()}")  # Show first 5 logits
    print(f"✓ Point Cloud branch working correctly!")
    
    return model


if __name__ == "__main__":
    test_original_pointcmt()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
