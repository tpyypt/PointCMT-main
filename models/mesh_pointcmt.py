import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshPointCMT(nn.Module):
    """Compatibility wrapper: use MeshNet as the "3D branch" in PointCMT.

    The existing PointCMT training code (train_pointcmt.py) expects the 3D branch
    to behave like `models/pointnet2.py::PointNet2`:

      - forward(pc, fc_only=False) -> (out_dict, feat_3d)
      - forward(mvf=..., fc_only=True) -> out_dict

    Where:
      - out_dict = {'logit': logits}
      - feat_3d is a global feature with dim=512 (because CMPG decoder expects 512)

    MeshNet's native penultimate feature is 256-dim (after `classifier[:-1]`).
    We project it to 512 to keep the rest of PointCMT unchanged.
    """

    def __init__(self, meshnet: nn.Module, num_class: int, proj_dim: int = 512):
        super().__init__()
        self.meshnet = meshnet

        # MeshNet penultimate feature dim is 256 (per models/MeshNet.py).
        self.mesh_feat_dim = 256
        self.proj_dim = proj_dim

        self.proj = nn.Linear(self.mesh_feat_dim, proj_dim)

        # A shared classifier head that takes 512-d features.
        # This keeps `fc_only=True` working with offline MVCNN features (512-d).
        self.fc = nn.Linear(proj_dim, num_class)

        # Ensure MeshNet returns (cls, fea)
        if hasattr(self.meshnet, 'require_fea'):
            self.meshnet.require_fea = True

    def forward(self, pc=None, fc_only: bool = False, mvf=0):
        # Keep PointCMT's original interface:
        #   - fc_only path uses mvf as the input feature
        if fc_only:
            mv_feature = mvf
            if not torch.is_tensor(mv_feature):
                raise TypeError(
                    f"fc_only=True expects mvf as a Tensor, got {type(mv_feature)}")
            logit = self.fc(mv_feature)
            return {'logit': logit}

        # Mesh path: `pc` can be either a tuple/list of four tensors
        # (centers, corners, normals, neighbor_index), or a dict with these keys.
        if isinstance(pc, (tuple, list)):
            if len(pc) != 4:
                raise ValueError(
                    f"Mesh input tuple must be (centers,corners,normals,neighbor_index), got len={len(pc)}")
            centers, corners, normals, neighbor_index = pc
        elif isinstance(pc, dict):
            centers = pc['centers']
            corners = pc['corners']
            normals = pc['normals']
            neighbor_index = pc['neighbor_index']
        else:
            raise TypeError(
                "MeshPointCMT expects `pc` as (centers,corners,normals,neighbor_index) tuple/list or a dict.")

        device = next(self.parameters()).device
        centers = centers.to(device)
        corners = corners.to(device)
        normals = normals.to(device)
        neighbor_index = neighbor_index.to(device)

        # MeshNet returns (cls, fea_norm) when require_fea=True
        cls, fea = self.meshnet(centers, corners, normals, neighbor_index)

        # Be robust to MeshNet's normalization implementation.
        fea = F.normalize(fea, p=2, dim=1)

        # Project 256 -> 512 to match PointCMT's CMPG decoder input.
        feat_3d = self.proj(fea)

        logit = self.fc(feat_3d)
        out = {'logit': logit}
        return out, feat_3d
