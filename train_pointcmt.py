import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random

import os
import numpy as np
import argparse
import pprint
import importlib
import yaml
import models

from time import time
from emdloss import emd_module
from data.modelnet40_mv_loader import ModelNet40_OfflineFeatures, ModelNet40
from utils.all_utils import PerfTrackTrain, PerfTrackVal, TrackTrain, smooth_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pc_to_dummy_mesh(pc):
    """
    临时把点云 [B, N, 3] 转成 MeshNet 需要的 (centers,corners,normals,neighbor_index)。

    适配你当前这版 MeshNet：
      - normals: [B, 3, N]
      - neighbor_index: [B, N, 3]  (注意：是3维！K=3)
    """
    assert torch.is_tensor(pc) and pc.dim() == 3 and pc.size(-1) == 3, pc.shape
    B, N, _ = pc.shape
    device = pc.device
    dtype = pc.dtype

    # centers: [B, 3, N]
    centers = pc.permute(0, 2, 1).contiguous()

    # corners: [B, 9, N]
    corners = pc.unsqueeze(2).repeat(1, 1, 3, 1)                 # [B, N, 3, 3]
    corners = corners.reshape(B, N, 9).permute(0, 2, 1).contiguous()  # [B, 9, N]

    # normals: [B, 3, N]
    normals = torch.zeros(B, 3, N, device=device, dtype=dtype)
    normals[:, 2, :] = 1.0

    # neighbor_index: [B, N, 3] 且 dtype long
    K = 3
    idx = torch.arange(N, device=device, dtype=torch.long)       # [N]
    neigh = torch.stack([(idx + i + 1) % N for i in range(K)], dim=1)  # [N, 3]
    neighbor_index = neigh.unsqueeze(0).repeat(B, 1, 1).contiguous()   # [B, N, 3]

    return (centers, corners, normals, neighbor_index)


def build_mesh_data(data_batch):
    if all(k in data_batch for k in ('mesh_centers', 'mesh_corners', 'mesh_normals', 'mesh_neighbors')):
        return {
            'centers': data_batch['mesh_centers'].cuda(),
            'corners': data_batch['mesh_corners'].cuda(),
            'normals': data_batch['mesh_normals'].cuda(),
            'neighbor_index': data_batch['mesh_neighbors'].cuda(),
        }

    pc_in = data_batch['pointcloud']
    if torch.is_tensor(pc_in) and pc_in.dim() == 3 and pc_in.size(-1) == 3:
        centers, corners, normals, neighbor_index = pc_to_dummy_mesh(pc_in.cuda())
        return {
            'centers': centers,
            'corners': corners,
            'normals': normals,
            'neighbor_index': neighbor_index,
        }
    raise KeyError("MeshNet expects mesh data; verify mesh_root and dataset mesh fields.")



def normalize_batch(data_batch):
    """
    Normalize batch from DataLoader into a dict with keys:
      - 'pointcloud'
      - 'multiview'
      - 'label'

    Supports:
      - dict batch (already OK)
      - list/tuple batch of length >= 3
        typical: [pc, mv, label] or [mv, pc, label]
    """
    if isinstance(data_batch, dict):
        # already the expected format
        return data_batch

    if not isinstance(data_batch, (list, tuple)):
        raise TypeError(f"Unsupported batch type: {type(data_batch)}")

    if len(data_batch) < 3:
        raise ValueError(f"Unexpected batch length: {len(data_batch)} (expected >= 3)")

    a, b, c = data_batch[0], data_batch[1], data_batch[2]

    # label 通常是 1D LongTensor [B]
    # 有些实现可能把 label 放在第 0/1 位，这里做个泛化：从前三个里找 1D tensor 当 label
    items = [a, b, c]
    label_idx = None
    for idx, it in enumerate(items):
        if torch.is_tensor(it) and it.dim() == 1:
            label_idx = idx
            break
    if label_idx is None:
        # fallback：第三个当 label（多数 dataset 这样）
        label_idx = 2

    label = items[label_idx]
    rest = [items[i] for i in range(3) if i != label_idx]
    x1, x2 = rest[0], rest[1]

    def is_mv_feat(x):
        # OfflineFeatures 的 mv_feature 常见是 [B, 512]（dim==2）
        return torch.is_tensor(x) and x.dim() == 2

    def is_mv_image(x):
        # 如果是原始多视图图像：[B, V, C, W, H]（dim==5）
        return torch.is_tensor(x) and x.dim() == 5

    def is_point_tensor(x):
        # 点云常见：[B, N, 3]
        return torch.is_tensor(x) and x.dim() == 3 and x.size(-1) == 3

    def is_mesh_struct(x):
        # MeshNet 可能是 tuple/list/dict（centers,corners,normals,neighbor_index）
        return isinstance(x, (tuple, list, dict))

    # 判谁是 multiview
    if is_mv_image(x1) or is_mv_feat(x1):
        multiview, pointcloud = x1, x2
    elif is_mv_image(x2) or is_mv_feat(x2):
        multiview, pointcloud = x2, x1
    else:
        # 兜底：如果一个是点云tensor/mesh_struct，另一个就当 multiview
        if is_point_tensor(x1) or is_mesh_struct(x1):
            pointcloud, multiview = x1, x2
        else:
            pointcloud, multiview = x2, x1

    return {
        'pointcloud': pointcloud,
        'multiview': multiview,
        'label': label
    }


def get_loss(task, loss_name, data_batch, out):
    """
    Returns the tensor loss function
    :param task:
    :param loss_name:
    :param data_batch: batched data; note not applied data_batch
    :param out: output from the model
    :param dataset_name:
    :return: tensor
    """
    if task == 'cls':
        label = data_batch['label'].to(out['logit'].device)
        if loss_name == 'cross_entropy':
            loss = F.cross_entropy(out['logit'], label)
        elif loss_name == 'smooth':
            loss = smooth_loss(out['logit'], label)
        else:
            assert False
    return loss


def validate(loader, model, task='cls'):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0

    with torch.no_grad():
        time5 = time()
        for i, data_batch in enumerate(loader):
            time1 = time()
            time2 = time()

            data_batch = normalize_batch(data_batch)
            if cfg.model_name == 'meshnet':
                mesh_data = build_mesh_data(data_batch)
                out, _ = model(mesh_data)
            else:
                pc_in = data_batch['pointcloud']
                if torch.is_tensor(pc_in):
                    pc_in = pc_in.cuda()
                out, _ = model(pc_in)

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    return perf.agg()


def scale(gt_pc, pr_pc):
    B = gt_pc.shape[0]
    min_gt = gt_pc.min(axis=1)[0]
    max_gt = gt_pc.max(axis=1)[0]
    min_pr = pr_pc.min(axis=1)[0]
    max_pr = pr_pc.max(axis=1)[0]
    length_gt = torch.abs(max_gt - min_gt)
    length_pr = torch.abs(max_pr - min_pr)
    diff_gt = length_gt.max(axis=1, keepdim=True)[0] - length_gt
    diff_pr = length_pr.max(axis=1, keepdim=True)[0] - length_pr
    size_pr = length_pr.max(axis=1)[0]
    size_gt = length_gt.max(axis=1)[0]
    scaling_factor_gt = 1. / size_gt
    scaling_factor_pr = 1. / size_pr
    new_min_gt = (min_gt - diff_gt) / 2.
    new_min_pr = (min_pr - diff_pr) / 2.
    box_min = torch.ones_like(new_min_gt) * -0.5
    adjustment_factor_gt = box_min - (scaling_factor_gt * new_min_gt.permute((1, 0))).permute((1, 0))
    adjustment_factor_pr = box_min - (scaling_factor_pr * new_min_pr.permute((1, 0))).permute((1, 0))
    pred_scaled = (pr_pc.permute(2, 1, 0) * scaling_factor_pr).permute(2, 1, 0) + adjustment_factor_pr.reshape(B, -1, 3)
    gt_scaled = (gt_pc.permute(2, 1, 0) * scaling_factor_gt).permute(2, 1, 0) + adjustment_factor_gt.reshape(B, -1, 3)
    return gt_scaled, pred_scaled


def train(loader, model, decoder_model, optimizer, EmdLoss, task='cls'):
    decoder_model.eval()
    model.train()

    def get_extra_param():
        return None

    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0

    train_fe_loss = 0.0
    train_cle_loss = 0.0

    time3 = time()
    for i, data_batch in enumerate(loader):
        time1 = time()
        pointcloud = data_batch['pointcloud']
        batch_size = pointcloud.shape[0] if torch.is_tensor(pointcloud) else len(pointcloud)
        mv_feature = data_batch['multiview']

        # MeshNet branch uses mesh data (centers/corners/normals/neighbors).
        if cfg.model_name == 'meshnet':
            mesh_data = build_mesh_data(data_batch)
            out, mesh_feature = model(mesh_data)
            pc_feature = mesh_feature
        else:
            # 原来的PointNet2分支
            out, pc_feature = model(data_batch['pointcloud'])
            mesh_feature = pc_feature  # 统一变量名
        # ================================

        loss = get_loss(task, 'smooth', data_batch, out)

        if not cfg.no_pointcmt:
            mv2pc_logits = model(mvf=mv_feature.to(DEVICE), fc_only=True)
            pc_dec_pc = decoder_model(pc_feature)
            mv_dec_pc = decoder_model(mv_feature)
            gt_scaled, pr_scaled = scale(mv_dec_pc, pc_dec_pc)

            cleloss = F.kl_div(out['logit'].softmax(dim=1).log(), (mv2pc_logits['logit']).softmax(dim=1),
                               reduction='sum')
            loss += 0.3 * cleloss

            fe_loss, _ = EmdLoss(pr_scaled, gt_scaled, 0.05, 3000)
            fe_loss = torch.sqrt(fe_loss).mean(1).mean()
            loss += 30 * fe_loss
        else:
            cleloss = torch.Tensor([0])
            fe_loss = torch.Tensor([0])

        train_fe_loss += fe_loss.item() * batch_size
        train_cle_loss += cleloss.item() * batch_size

        perf.update_all(data_batch=data_batch, out=out, loss=loss)
        time2 = time()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)

        if i % 100 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")

    print('Feature enhancement loss is ', train_fe_loss * 1.0 / 9840)
    print('Classifier enhencement loss is ', train_cle_loss * 1.0 / 9840)

    return perf.agg(), perf.agg_loss()


def save_checkpoint(id, epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    path = f"./checkpoints/{cfg.exp_name}/model_{id}.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state'])
            model = model.module

    optimizer.load_state_dict(checkpoint['optimizer_state'])

    # for backward compatibility with saved models
    if 'lr_sched_state' in checkpoint:
        lr_sched.load_state_dict(checkpoint['lr_sched_state'])
        if checkpoint['bnm_sched_state'] is not None:
            bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
    else:
        print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

    return model


def get_model(cfg):
    if cfg.model_name == 'pointnet2':
        model = models.PointNet2(num_class=cfg.num_class)
    elif cfg.model_name == 'meshnet':  # 新增分支
        if cfg.meshnet_cfg is None:
            mesh_cfg = {
                'structural_descriptor': {'num_kernel': 64, 'sigma': 0.2},
                'mesh_convolution': {'aggregation_method': 'Concat'},
                'mask_ratio': 0.95,
                'dropout': 0.5,
                'num_classes': cfg.num_class,
            }
        else:
            with open(cfg.meshnet_cfg, 'r') as f:
                y = yaml.safe_load(f)
            mesh_cfg = y.get('MeshNet', y)
            mesh_cfg = dict(mesh_cfg)
            mesh_cfg['num_classes'] = cfg.num_class

        model = models.MeshNetPointCMT(num_class=cfg.num_class, mesh_cfg=mesh_cfg)
    else:
        raise NotImplementedError

    return model

def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    else:
        assert False
    return metric


def get_optimizer(params):
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-2)
    lr_sched = lr_scheduler.CosineAnnealingLR(
        optimizer,
        1000,
        eta_min=0,
        last_epoch=-1)
    bnm_sched = None

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg):
    mesh_root = None
    if cfg.model_name == 'meshnet':
        if cfg.mesh_root is None:
            raise ValueError("mesh_root is required when model_name=meshnet")
        mesh_root = cfg.mesh_root

    dataset_train = ModelNet40_OfflineFeatures(cfg.data_root, split='train', mesh_root=mesh_root)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size,
                                               num_workers=8, shuffle=True, drop_last=True,
                                               pin_memory=(torch.cuda.is_available()))
    loader_test = torch.utils.data.DataLoader(
        ModelNet40(
            data_path=cfg.data_root,
            partition='test',
            mesh_root=mesh_root,
        ),
        num_workers=8,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    model = get_model(cfg)
    model.to(DEVICE)
    model = nn.DataParallel(model)

    decoder_model = importlib.import_module('models.cmpg')
    decoder_model = decoder_model.get_model().to(DEVICE)
    decoder_model = nn.DataParallel(decoder_model)

    params = list(model.parameters())
    optimizer, lr_sched, bnm_sched = get_optimizer(params)

    deccheckpoint = torch.load(cfg.cmpg_checkpoint, weights_only=False)
    decoder_model.load_state_dict(deccheckpoint['model_state'])
    decoder_model.eval()

    log_dir = f"./checkpoints/{cfg.exp_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    track_train = TrackTrain(early_stop_patience=1000)

    EmdLoss = emd_module.emdModule()
    print(str(model))
    for epoch in range(cfg.epochs):
        print(f'\nEpoch {epoch}')
        start = time()

        print('Training..')
        train_perf, train_loss = train(loader_train, model, decoder_model, optimizer, EmdLoss)
        pprint.pprint(train_perf, width=80)
        print('\nTesting..')
        test_perf = validate(loader_test, model)
        pprint.pprint(test_perf, width=80)
        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf('cls', train_perf, 'acc'),
            test_metric=get_metric_from_perf('cls', test_perf, 'acc'))

        if track_train.save_model(epoch, 'test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        if epoch % 25 == 0:
            save_checkpoint(f'{epoch}', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        lr_sched.step(epoch)

        end = time()
        last = end - start
        print('every epoch lasts for ', last)

    print('Saving the final model')
    save_checkpoint('final', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='pointnet2_pointcmt', help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data_root', type=str, default='dataset/ModelNet40/data/', help='Name of the data root')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of the model')
    parser.add_argument('--mesh_root', type=str, default=None,
                        help='MeshNet npz root with class/train|test subfolders.')
    parser.add_argument('--meshnet_cfg', type=str, default='config/train_config.yaml',
                        help='YAML path for MeshNet config (will read the `MeshNet:` section).')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=20, help='number of episode to train ')
    parser.add_argument('--cmpg_checkpoint', type=str, default="pretrained/modelnet40/cmpg.pth", help='decoder model of multiview')
    parser.add_argument('--num_class', type=int, default=40)
    parser.add_argument('--no_pointcmt', default=False, action='store_true')
    cfg = parser.parse_args()

    print(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    entry_train(cfg)
