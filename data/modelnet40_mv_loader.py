#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from glob import glob

import numpy as np
import torch
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import Dataset

from utils.pc_utils import translate_pointcloud


def load_mv_data(root, partition):
    dat_path = os.path.join(root, f"modelnet40_{partition}_2048pts_20views.dat")
    with open(dat_path, "rb") as f:
        all_pc, all_mv, all_label = pickle.load(f)
    print("load mv data")
    print("The size of %s data is %d" % (partition, len(all_pc)))
    return all_pc, all_mv, all_label


# ModelNet40 标准 40 类（很多仓库都用这个顺序）
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio",
    "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox"
]


def _scan_mesh_npz(mesh_root, part):
    """
    扫描 mesh_root/<class>/<part>/*.npz
    返回：class_name -> [npz_path_sorted]
    """
    by_cls = {}
    if mesh_root is None:
        return by_cls
    pattern = os.path.join(mesh_root, "*", part, "*.npz")
    paths = glob(pattern)
    paths.sort()
    for p in paths:
        # .../<mesh_root>/<class>/<part>/<stem>.npz
        cls = os.path.basename(os.path.dirname(os.path.dirname(p)))
        by_cls.setdefault(cls, []).append(p)
    return by_cls


class ModelNet40(Dataset):
    """
    PointCMT dat: pointcloud + 20-view + label
    + (可选) MeshNet npz: faces/neighbors -> centers/corners/normals/neighbors
    """

    def __init__(
        self,
        data_path,
        num_points=1024,
        partition="train",
        generate=False,
        mesh_root=None,
        mesh_partition=None,
    ):
        self.pc_data, self.mv_data, self.mv_label = load_mv_data(data_path, partition)

        self.num_views = 20
        self.num_points = num_points
        self.partition = partition
        self.generate = generate

        self.transform_mv = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # ---- mesh 对齐相关 ----
        self.mesh_root = mesh_root
        self.mesh_part = mesh_partition if mesh_partition is not None else partition

        self.mesh_by_cls = _scan_mesh_npz(self.mesh_root, self.mesh_part)
        self.use_mesh = self.mesh_root is not None

        # 用于对齐后的索引：valid_indices 里存 dat 的真实 index
        # 同时 aligned_mesh_paths 与 valid_indices 一一对应
        self.valid_indices = None
        self.aligned_mesh_paths = None

        if self.use_mesh:
            # 统计 dat 每类数量、mesh 每类数量
            dat_cnt = {c: 0 for c in MODELNET40_CLASSES}
            for y in self.mv_label:
                cls = MODELNET40_CLASSES[int(y)]
                dat_cnt[cls] += 1

            mesh_cnt = {c: len(self.mesh_by_cls.get(c, [])) for c in MODELNET40_CLASSES}

            # 每类取 min(dat, mesh) 来保证不会“错位滑动”
            keep_cnt = {c: min(dat_cnt[c], mesh_cnt[c]) for c in MODELNET40_CLASSES}

            total_dat = len(self.mv_label)
            total_mesh = sum(mesh_cnt.values())
            total_keep = sum(keep_cnt.values())

            if total_keep != total_dat:
                missing_total = total_dat - total_keep
                # 打印哪些类不一致（你这里会看到 bookshelf 少 1）
                bad = [
                    (c, dat_cnt[c], mesh_cnt[c], keep_cnt[c])
                    for c in MODELNET40_CLASSES
                    if dat_cnt[c] != mesh_cnt[c]
                ]
                print("[WARN] dat 与 mesh 数量不一致，将按“类别计数”严格对齐并跳过多余 dat 样本。")
                print(f"[WARN] dat={total_dat}, mesh={total_mesh}, keep={total_keep}, dropped={missing_total}")
                for c, a, b, k in bad:
                    print(f"  - {c}: dat={a}, mesh={b}, keep={k}")

            # 构造对齐列表：遍历 dat 的顺序，对每个类别只保留前 keep_cnt[cls] 个
            used_dat = {c: 0 for c in MODELNET40_CLASSES}
            used_mesh = {c: 0 for c in MODELNET40_CLASSES}

            valid_indices = []
            aligned_mesh_paths = []

            for i, y in enumerate(self.mv_label):
                cls = MODELNET40_CLASSES[int(y)]
                if used_dat[cls] >= keep_cnt[cls]:
                    continue  # 该类多余的 dat 样本直接跳过

                # 取该类的第 used_mesh[cls] 个 mesh npz
                lst = self.mesh_by_cls.get(cls, [])
                j = used_mesh[cls]
                if j >= len(lst):
                    # 理论上不会发生（因为 keep_cnt= min）
                    continue

                valid_indices.append(i)
                aligned_mesh_paths.append(lst[j])

                used_dat[cls] += 1
                used_mesh[cls] += 1

            self.valid_indices = valid_indices
            self.aligned_mesh_paths = aligned_mesh_paths

            assert len(self.valid_indices) == len(self.aligned_mesh_paths), "internal align error"
        else:
            # 不使用 mesh：保持原始 dat 行为
            self.valid_indices = list(range(len(self.mv_label)))
            self.aligned_mesh_paths = None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, item):
        real_i = self.valid_indices[item]

        data_dict = {}

        # ----- multiview -----
        _views = []
        views = self.mv_data[real_i]
        label = int(self.mv_label[real_i])

        for i in range(self.num_views):
            _views.append(self.transform_mv(Image.fromarray(views[i], "RGB")))
        views = torch.stack(_views, 0)

        # ----- pointcloud（先保留，后续你可不用） -----
        pointcloud = np.array(self.pc_data[real_i][: self.num_points, 0:3])

        if self.partition == "train" and not self.generate:
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        pointcloud = pointcloud.astype("float32")

        data_dict["pointcloud"] = pointcloud
        data_dict["multiview"] = views
        data_dict["label"] = label

        # ----- mesh (centers/corners/normals/neighbors) -----
        if self.aligned_mesh_paths is not None:
            npz_path = self.aligned_mesh_paths[item]  # 注意：这里用 item（对齐后的下标）
            d = np.load(npz_path)

            face = d["faces"]            # (F, 15)
            neighbor_index = d["neighbors"]  # (F, 3)

            face = torch.from_numpy(face).float().permute(1, 0).contiguous()  # (15, F)
            neighbor_index = torch.from_numpy(neighbor_index).long()          # (F, 3)

            centers, corners, normals = face[:3], face[3:12], face[12:]
            corners = corners - torch.cat([centers, centers, centers], 0)

            data_dict["mesh_centers"] = centers
            data_dict["mesh_corners"] = corners
            data_dict["mesh_normals"] = normals
            data_dict["mesh_neighbors"] = neighbor_index
            data_dict["mesh_path"] = npz_path

        return data_dict


class ModelNet40_OfflineFeatures(Dataset):
    def __init__(self, root, split="train", mesh_root=None, mesh_partition=None):
        self.root = root
        self.split = split
        feature_path = root + r"/modelnet40_%s_mvf.pth" % self.split
        self.dataset = torch.load(feature_path)
        self.mesh_root = mesh_root
        self.mesh_part = mesh_partition if mesh_partition is not None else split
        self.mesh_by_cls = _scan_mesh_npz(self.mesh_root, self.mesh_part)
        self.use_mesh = self.mesh_root is not None

        self.valid_indices = None
        self.aligned_mesh_paths = None

        if self.use_mesh:
            dat_cnt = {c: 0 for c in MODELNET40_CLASSES}
            for _, label, _ in self.dataset:
                cls = MODELNET40_CLASSES[int(label)]
                dat_cnt[cls] += 1

            mesh_cnt = {c: len(self.mesh_by_cls.get(c, [])) for c in MODELNET40_CLASSES}
            keep_cnt = {c: min(dat_cnt[c], mesh_cnt[c]) for c in MODELNET40_CLASSES}

            total_dat = len(self.dataset)
            total_mesh = sum(mesh_cnt.values())
            total_keep = sum(keep_cnt.values())

            if total_keep != total_dat:
                missing_total = total_dat - total_keep
                bad = [
                    (c, dat_cnt[c], mesh_cnt[c], keep_cnt[c])
                    for c in MODELNET40_CLASSES
                    if dat_cnt[c] != mesh_cnt[c]
                ]
                print("[WARN] dat 与 mesh 数量不一致，将按“类别计数”严格对齐并跳过多余 dat 样本。")
                print(f"[WARN] dat={total_dat}, mesh={total_mesh}, keep={total_keep}, dropped={missing_total}")
                for c, a, b, k in bad:
                    print(f"  - {c}: dat={a}, mesh={b}, keep={k}")

            used_dat = {c: 0 for c in MODELNET40_CLASSES}
            used_mesh = {c: 0 for c in MODELNET40_CLASSES}

            valid_indices = []
            aligned_mesh_paths = []

            for i, (_, label, _) in enumerate(self.dataset):
                cls = MODELNET40_CLASSES[int(label)]
                if used_dat[cls] >= keep_cnt[cls]:
                    continue

                lst = self.mesh_by_cls.get(cls, [])
                j = used_mesh[cls]
                if j >= len(lst):
                    continue

                valid_indices.append(i)
                aligned_mesh_paths.append(lst[j])

                used_dat[cls] += 1
                used_mesh[cls] += 1

            self.valid_indices = valid_indices
            self.aligned_mesh_paths = aligned_mesh_paths
            assert len(self.valid_indices) == len(self.aligned_mesh_paths), "internal align error"
        else:
            self.valid_indices = list(range(len(self.dataset)))
            self.aligned_mesh_paths = None

    def __len__(self):
        return len(self.valid_indices)

    def _get_item(self, index):
        points, label, mvf = self.dataset[index]

        if self.split == "train":
            points_np = points.numpy() if torch.is_tensor(points) else np.asarray(points)
            points_np = translate_pointcloud(points_np)
            np.random.shuffle(points_np)
            points = points_np

        return points, label, mvf

    def __getitem__(self, index):
        real_i = self.valid_indices[index]
        points, label, mvf = self._get_item(real_i)
        data_dict = {
            "pointcloud": points,
            "multiview": mvf,
            "label": label,
        }

        if self.aligned_mesh_paths is not None:
            npz_path = self.aligned_mesh_paths[index]
            d = np.load(npz_path)
            face = d["faces"]
            neighbor_index = d["neighbors"]

            face = torch.from_numpy(face).float().permute(1, 0).contiguous()
            neighbor_index = torch.from_numpy(neighbor_index).long()

            centers, corners, normals = face[:3], face[3:12], face[12:]
            corners = corners - torch.cat([centers, centers, centers], 0)

            data_dict["mesh_centers"] = centers
            data_dict["mesh_corners"] = corners
            data_dict["mesh_normals"] = normals
            data_dict["mesh_neighbors"] = neighbor_index
            data_dict["mesh_path"] = npz_path

        return data_dict
