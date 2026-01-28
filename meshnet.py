"""
MeshNet Branch Implementation for PointCMT
Based on MeshNet architecture for 3D mesh processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshConvolution(nn.Module):
    """Mesh convolution layer that processes faces and their neighbors"""
    
    def __init__(self, in_channels, out_channels, neighbor_num=3):
        super(MeshConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neighbor_num = neighbor_num
        
        # Spatial and structural feature aggregation
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, 
                                      kernel_size=(1, neighbor_num))
        self.structural_conv = nn.Conv2d(in_channels, out_channels, 
                                        kernel_size=(1, neighbor_num))
        
    def forward(self, spatial_features, structural_features):
        """
        Forward pass for mesh convolution
        
        Args:
            spatial_features: Spatial features (B, C, F, neighbor_num)
            structural_features: Structural features (B, C, F, neighbor_num)
            
        Returns:
            features: Aggregated features (B, out_channels, F, 1)
        """
        spatial_out = self.spatial_conv(spatial_features)
        structural_out = self.structural_conv(structural_features)
        
        # Combine spatial and structural information
        features = spatial_out + structural_out
        
        return features


class MeshBlock(nn.Module):
    """Mesh processing block with convolution and pooling"""
    
    def __init__(self, in_channels, out_channels, neighbor_num=3):
        super(MeshBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = MeshConvolution(in_channels, out_channels, neighbor_num)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, spatial_features, structural_features):
        """
        Forward pass for mesh block
        
        Args:
            spatial_features: Spatial features
            structural_features: Structural features
            
        Returns:
            features: Processed features
        """
        x = self.conv(spatial_features, structural_features)
        x = self.bn(x)
        x = F.relu(x)
        
        return x


class MeshNetBranch(nn.Module):
    """MeshNet branch for mesh data processing in PointCMT"""
    
    def __init__(self, input_channels=6, hidden_dims=[64, 128, 256], 
                 output_dim=512, neighbor_num=3):
        super(MeshNetBranch, self).__init__()
        self.input_channels = input_channels  # Faces with normals (3 vertices * 2 features)
        self.output_dim = output_dim
        self.neighbor_num = neighbor_num
        
        # Initial feature projection
        self.input_conv = nn.Conv2d(input_channels, hidden_dims[0], 
                                   kernel_size=(1, 1))
        self.input_bn = nn.BatchNorm2d(hidden_dims[0])
        
        # Mesh convolution blocks
        self.mesh_blocks = nn.ModuleList()
        in_ch = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            self.mesh_blocks.append(MeshBlock(in_ch, hidden_dim, neighbor_num))
            in_ch = hidden_dim
        
        # Final feature extraction
        self.final_conv = nn.Conv2d(hidden_dims[-1], output_dim, 
                                   kernel_size=(1, 1))
        self.final_bn = nn.BatchNorm2d(output_dim)
        
    def forward(self, mesh_data):
        """
        Forward pass for MeshNet branch
        
        Args:
            mesh_data: Mesh face features of shape (B, F, input_channels, neighbor_num)
                      where F is number of faces
            
        Returns:
            features: Global mesh features of shape (B, output_dim)
        """
        # mesh_data: (B, F, C, neighbor_num) -> (B, C, F, neighbor_num)
        x = mesh_data.permute(0, 2, 1, 3)
        
        # Initial feature extraction
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Split into spatial and structural features
        # For simplicity, we'll use the same features for both
        spatial_features = x
        structural_features = x
        
        # Process through mesh blocks
        for mesh_block in self.mesh_blocks:
            x = mesh_block(spatial_features, structural_features)
            spatial_features = x
            structural_features = x
        
        # Final feature extraction
        x = F.relu(self.final_bn(self.final_conv(x)))
        
        # Global max pooling over faces and neighbors
        x = torch.max(x, dim=3)[0]  # Pool over neighbors
        x = torch.max(x, dim=2)[0]  # Pool over faces
        
        return x
