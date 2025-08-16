import os
import random
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import distributed as dist

from models.HGPIFuNet_3d import HGPIFuNet_3d
from models.HGPIFuNet_xray import HGPIFuNet_xray
from recon_xray.dataset import ReconstructionDataset_Xray
from recon_xray.dataset_unlabeled import ReconstructionDataset_unlabeled
from utils import load_ckpt_safe, kaiming_normal_init_weight



class VecNorm(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
    
    def forward(self, input, target):
        vec = input - target
        norm = torch.norm(vec, p=self.p, dim=1) / input.shape[1]
        loss = torch.mean(norm)
        return loss


def create_loader(args, dst):
    def worker_init_fn(worker_id):
        worker_seed = (torch.initial_seed() + args.local_rank) % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    sampler = None
    if args.dist:
        sampler = torch.utils.data.distributed.DistributedSampler(dst)
    loader = DataLoader(
        dst,
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    return loader


def get_loss(pred_list, label, loss_func, w_scale=1.0):
    loss = 0.
    w_sum = 0.
    w = 1.0
    for pred in pred_list:
        loss += w * loss_func(pred, label)
        w_sum += w
        w *= w_scale
    loss /= w_sum
    return loss


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--name', type=str, default='semi+kd')
    parser.add_argument('--ct_name', type=str, default='semi')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--npoint', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--w_usp', type=float, default=0.5)
    parser.add_argument('--w_kd', type=float, default=0.1)
    parser.add_argument('--dist', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # dist initialization
    if args.local_rank == 0:
        save_dir = os.path.join('./recon_xray/logs', args.name)
        os.makedirs(save_dir, exist_ok=True)

    if args.dist:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    # data loaders
    unlabeled_dst = ReconstructionDataset_unlabeled(
        split='unlabeled'
    )
    unlabeled_loader = create_loader(args, unlabeled_dst)

    labeled_dst = ReconstructionDataset_Xray(
        split='train', 
        num_repeat=len(unlabeled_dst), 
    )
    labeled_loader = create_loader(args, labeled_dst)

    # models
    model_ct = HGPIFuNet_3d(in_ch=1, num_cls=5)
    ckpt_path = f'./recon_ct/logs/{args.ct_name}/latest.pth'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))['A']
    load_ckpt_safe(model_ct, ckpt)
    model_ct = model_ct.eval().cuda()

    model_xray = HGPIFuNet_xray(in_ch=2, num_cls=5)
    model_xray = kaiming_normal_init_weight(model_xray)
    model_xray = model_xray.cuda()

    if args.dist:
        model_ct = nn.parallel.DistributedDataParallel(model_ct, device_ids=[args.local_rank], find_unused_parameters=True)
        model_xray = nn.parallel.DistributedDataParallel(model_xray, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # loss functions, optimizer, lr scheduler
    weight = torch.FloatTensor(labeled_dst.weight).cuda()
    loss_func = nn.CrossEntropyLoss(weight)
    kd_loss = VecNorm(p=1)

    optimizer = torch.optim.SGD(
        model_xray.parameters(),
        lr=args.lr, 
        momentum=0.98, 
        weight_decay=3e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.epoch // 4,
        gamma=args.lr_decay
    )

    # training
    for epoch in range(args.epoch):
        if args.dist:
            unlabeled_loader.sampler.set_epoch(epoch)
            labeled_loader.sampler.set_epoch(epoch)
        
        loss_l_list = []
        loss_list = []

        model_xray.train()
        for batch_l, batch_u in zip(labeled_loader, unlabeled_loader):
            optimizer.zero_grad()

            # move to cuda
            batch_l['projs'] = batch_l['projs'].float().cuda()
            batch_l['points_proj'] = batch_l['points_proj'].float().cuda()
            batch_l['points_gt'] = batch_l['points_gt'].long().cuda()

            batch_u['ct'] = batch_u['ct'].float().cuda()
            batch_u['projs'] = batch_u['projs'].float().cuda()
            batch_u['points_proj'] = batch_u['points_proj'].float().cuda()
            batch_u['points_ct'] = batch_u['points_ct'].float().cuda()

            # labeled data
            outputs_l = model_xray(batch_l['projs'], batch_l['points_proj'])
            loss_l = get_loss(outputs_l['points_pred'], batch_l['points_gt'], loss_func, w_scale=1.5)

            # unlabeled data
            with torch.no_grad():
                outputs_ct = model_ct(batch_u['ct'], batch_u['points_ct'])
                pseudo_labels = torch.argmax(outputs_ct['points_pred'][-1], dim=1) # B, x, N => B, N
                encoding_feats = outputs_ct['encoding_feats'][-1]
            
            outputs_u = model_xray(batch_u['projs'], batch_u['points_proj'])
            loss_u = get_loss(outputs_u['points_pred'], pseudo_labels, loss_func, w_scale=1.5)
            loss_kd = get_loss(outputs_u['encoding_feats'], encoding_feats, kd_loss)

            # optimization
            loss = (1 - args.w_usp) * loss_l + args.w_usp * loss_u + args.w_kd * loss_kd
            loss.backward()
            optimizer.step()

            if args.dist:
                loss_l = reduce_mean(loss_l, dist.get_world_size())
                loss = reduce_mean(loss, dist.get_world_size())

            loss_l_list.append(loss_l.item())
            loss_list.append(loss.item())

        if args.local_rank == 0:
            print(f'  --- epoch: {epoch}, loss_l: {np.mean(loss_l_list)}, loss: {np.mean(loss_list)}')

            if args.dist:
                ckpt = model_xray.module.state_dict()
            else:
                ckpt = model_xray.state_dict()
            
            torch.save(ckpt, os.path.join(save_dir, 'latest.pth'))
            if epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0):
                torch.save(ckpt, os.path.join(save_dir, f'epoch_{epoch}.pth'))

        lr_scheduler.step()
