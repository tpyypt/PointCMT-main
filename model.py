"""
PointCMT: Cross-Modality Transformer with MeshNet Branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from meshnet import MeshNetBranch


class PointCMT(nn.Module):
    """
    PointCMT: Cross-Modality Transformer
    Currently uses MeshNet branch for 3D data processing
    """
    
    def __init__(self, num_classes=40, input_channels=6, feature_dim=512):
        super(PointCMT, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
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
