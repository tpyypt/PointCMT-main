import torch
import torch.nn as nn
from models.MeshNet import MeshNet


class MeshNetPointCMT(nn.Module):
    """
    使用MeshNet替换PointCMT中的点云分支
    保持与原PointNet2相同的接口,便于集成到PointCMT训练流程
    """

    def __init__(self, num_class, mesh_cfg=None):
        super().__init__()

        meshnet_cfg = mesh_cfg or {
            'structural_descriptor': {
                'num_kernel': 64,
                'sigma': 0.2
            },
            'mesh_convolution': {
                'aggregation_method': 'Concat'  # 可选: 'Concat', 'Max', 'Average'
            },
            'mask_ratio': 0.2,  # 训练时随机mask比例
            'dropout': 0.3,
            'num_classes': num_class
        }

        # MeshNet主干 (输出1024维全局特征)
        self.meshnet_backbone = MeshNet(cfg=meshnet_cfg, require_fea=True)

        # 特征映射层: 1024 -> 512 (对齐PointNet2特征维度)
        self.feature_projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 分类头 (保持与PointNet2一致)
        self.fc_layer = nn.Linear(512, num_class)

    def forward(self, mesh_data=None, fc_only=False, mvf=None):
        """
        前向传播,保持与原PointNet2相同的接口

        参数:
            mesh_data: dict, 包含 'centers', 'corners', 'normals', 'neighbor_index'
            fc_only: bool, 是否只运行分类头(用于多视图特征分类)
            mvf: Tensor, 多视图特征 [B, 512]

        返回:
            out: dict, 包含 'logit'
            mesh_feature: Tensor [B, 512], 投影后的Mesh特征
        """

        # 模式1: 仅分类头(用于多视图->点云的交叉模态训练)
        if fc_only:
            # mvf应该已经是512维
            logit = self.fc_layer(mvf)
            return {'logit': logit}

        # 模式2: 完整前向传播 Mesh -> 特征 -> 分类
        centers = mesh_data['centers']
        corners = mesh_data['corners']
        normals = mesh_data['normals']
        neighbor_index = mesh_data['neighbor_index']

        # MeshNet提取全局特征 (1024维)
        _, mesh_global_feat = self.meshnet_backbone(
            centers, corners, normals, neighbor_index
        )

        # 特征投影: 1024 -> 512
        mesh_feature = self.feature_projection(mesh_global_feat)

        # 分类
        logit = self.fc_layer(mesh_feature)

        out = {'logit': logit}
        return out, mesh_feature
