import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.unet_model import UNet
from dataset import SegmentationDataset
from utils import kaiming_normal_init_weight, xavier_normal_init_weight, get_current_consistency_weight



def create_model(args, init_func=None):
    model = UNet(n_channels=1, n_classes=2).cuda()
    
    if init_func is not None:
        model = init_func(model)

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.98, 
        weight_decay=3e-4
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.epoch // 4,
        gamma=args.lr_decay
    )

    return model, optimizer, lr_scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--cps_w', type=float, default=0.1)
    args = parser.parse_args()

    save_dir = os.path.join('./logs', args.name)
    os.makedirs(save_dir, exist_ok=True)

    unlabeled_dst = SegmentationDataset(
        split='unlabeled',
        unlabeled=True
    )
    labeled_dst = SegmentationDataset(
        split='train',
        repeat_num=len(unlabeled_dst)
    )

    unlabeled_loader = DataLoader(
        unlabeled_dst,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    labeled_loader = DataLoader(
        labeled_dst,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
    )

    model_A, optimizer_A, scheduler_A = create_model(args, kaiming_normal_init_weight)
    model_B, optimizer_B, scheduler_B = create_model(args, xavier_normal_init_weight)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        loss_sup_list = {
            'A': [],
            'B': []
        }
        loss_list = []
        model_A.train()
        model_B.train()

        cps_w = get_current_consistency_weight(args.cps_w, epoch, args.epoch)
        for batch_l, batch_u in zip(labeled_loader, unlabeled_loader):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            batch_l['image'] = batch_l['image'].cuda()
            batch_l['seg_mask'] = batch_l['seg_mask'].long().cuda()
            batch_u['image'] = batch_u['image'].cuda()

            # labeled part
            pred_A_l = model_A(batch_l['image'])
            pred_B_l = model_B(batch_l['image'])
            sup_A = loss_func(pred_A_l, batch_l['seg_mask'])
            sup_B = loss_func(pred_B_l, batch_l['seg_mask'])
            sup_loss = sup_A + sup_B
            sup_loss.backward()

            # unlabeled part
            pred_A_u = model_A(batch_u['image'])
            pred_B_u = model_B(batch_u['image'])
            mask_A = torch.argmax(pred_A_u.detach(), dim=1).long()
            mask_B = torch.argmax(pred_B_u.detach(), dim=1).long()
            cps_loss = loss_func(pred_A_u, mask_B) + loss_func(pred_B_u, mask_A)
            (cps_w * cps_loss).backward()
            
            optimizer_A.step()
            optimizer_B.step()

            loss = sup_loss + cps_w * cps_loss
            loss_list.append(loss.item())
            loss_sup_list['A'].append(sup_A.item())
            loss_sup_list['B'].append(sup_B.item())

        loss_sup_A = np.mean(loss_sup_list['A'])
        loss_sup_B = np.mean(loss_sup_list['B'])
        print(f'  --- epoch: {epoch}, cps_w: {cps_w}, loss_sup (A): {loss_sup_A}, loss_sup (B): {loss_sup_B}, loss: {np.mean(loss_list)}')

        torch.save({
                'A': model_A.state_dict(),
                'B': model_B.state_dict()
        }, os.path.join(save_dir, 'latest.pth'))

        if epoch % 100 == 0 or (epoch >= (args.epoch - 100) and epoch % 10 == 0):
            torch.save({
                'A': model_A.state_dict(),
                'B': model_B.state_dict()
            }, os.path.join(save_dir, f'epoch_{epoch}.pth'))

        scheduler_A.step()
        scheduler_B.step()

