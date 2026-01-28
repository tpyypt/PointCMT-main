"""
PointCMT: Cross-Modality Transformer with Point Cloud and Mesh Branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from meshnet import MeshNetBranch


class PointCloudBranch(nn.Module):
    """Point Cloud processing branch for PointCMT"""
    
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=512):
        super(PointCloudBranch, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Point cloud feature extraction layers
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        self.conv4 = nn.Conv1d(hidden_dim, output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        """
        Forward pass for point cloud branch
        
        Args:
            x: Point cloud data of shape (B, N, 3) where B is batch size, N is number of points
            
        Returns:
            features: Global features of shape (B, output_dim)
        """
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global max pooling
        x = torch.max(x, 2)[0]
        
        return x


class PointCMT(nn.Module):
    """
    PointCMT: Cross-Modality Transformer
    Currently uses MeshNet branch for 3D data processing
    """
    
    def __init__(self, num_classes=40, input_channels=6, feature_dim=512):
        super(PointCMT, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # MeshNet branch for 3D mesh data processing
        self.mesh_branch = MeshNetBranch(
            input_channels=input_channels,
            hidden_dims=[64, 128, 256],
            output_dim=feature_dim,
            neighbor_num=3
        )
        
        # Classifier head
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        """
        Forward pass for PointCMT
        
        Args:
            x: Input data (mesh) of shape (B, F, input_channels, neighbor_num)
            
        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        # Extract features using MeshNet branch
        features = self.mesh_branch(x)
        
        # Classification head
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits
