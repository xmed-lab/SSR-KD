import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.HGPIFuNet_3d import HGPIFuNet_3d
from recon_ct.dataset import ReconstructionDataset_CT
from utils import kaiming_normal_init_weight, xavier_normal_init_weight



def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(weight, epoch, max_epoch):
    return weight * sigmoid_rampup(epoch, max_epoch)


def create_model(args, init_func=None):
    model = HGPIFuNet_3d(in_ch=1, num_cls=5).cuda()

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


def get_loss(pred_list, label, loss_func):
    loss = 0.
    for pred in pred_list:
        loss += loss_func(pred, label)
    loss /= len(pred_list)
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--name', type=str, default='semi')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--npoint', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--w_cps', type=float, default=0.1)

    # for ablation study
    parser.add_argument('--num_repeat', type=int, default=None)
    parser.add_argument('--num_labeled', type=int, default=None)
    parser.add_argument('--num_unlabeled', type=int, default=None)
    args = parser.parse_args()

    save_dir = os.path.join('./recon_ct/logs', args.name)
    os.makedirs(save_dir, exist_ok=True)

    unlabeled_dst = ReconstructionDataset_CT(split='unlabeled', npoint=args.npoint, unlabeled=True, num_repeat=args.num_repeat, num_data=args.num_unlabeled)
    unlabeled_loader = DataLoader(
        unlabeled_dst,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    labeled_dst = ReconstructionDataset_CT(split='train', npoint=args.npoint, num_repeat=len(unlabeled_dst), num_data=args.num_labeled)
    labeled_loader = DataLoader(
        labeled_dst,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True
    )

    model_A, optimizer_A, scheduler_A = create_model(args, kaiming_normal_init_weight)
    model_B, optimizer_B, scheduler_B = create_model(args, xavier_normal_init_weight)

    weight = torch.FloatTensor(labeled_dst.weight).cuda()
    loss_func = nn.CrossEntropyLoss(weight)

    for epoch in range(args.epoch):
        loss_sup_list = {
            'A': [],
            'B': []
        }
        loss_list = []
        model_A.train()
        model_B.train()

        w_cps = get_current_consistency_weight(args.w_cps, epoch, args.epoch)
        for batch_l, batch_u in zip(labeled_loader, unlabeled_loader):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            batch_l['ct'] = batch_l['ct'].cuda()
            batch_l['points'] = batch_l['points'].cuda()
            batch_l['labels'] = batch_l['labels'].long().cuda()
            
            batch_u['ct'] = batch_u['ct'].cuda()
            batch_u['points'] = batch_u['points'].cuda()

            '''--- labeled part ---'''
            output_A_l = model_A(batch_l['ct'], batch_l['points'])
            output_B_l = model_B(batch_l['ct'], batch_l['points'])

            # supervise
            sup_loss_A = get_loss(output_A_l['preds'], batch_l['labels'], loss_func)
            sup_loss_B = get_loss(output_B_l['preds'], batch_l['labels'], loss_func)
            sup_loss = sup_loss_A + sup_loss_B
            loss_sup_list['A'].append(sup_loss_A.item())
            loss_sup_list['B'].append(sup_loss_B.item())
            
            # cross supervise
            pred_A = torch.argmax(output_A_l['preds'][-1].detach(), dim=1)
            pred_B = torch.argmax(output_B_l['preds'][-1].detach(), dim=1)
            usp_loss = get_loss(output_A_l['preds'], pred_B, loss_func) + \
                get_loss(output_B_l['preds'], pred_A, loss_func)
            
            loss_labeled = sup_loss + w_cps * usp_loss / 2.0
            loss_labeled.backward()

            '''--- unlabeled part ---'''
            output_A_u = model_A(batch_u['ct'], batch_u['points'])
            output_B_u = model_B(batch_u['ct'], batch_u['points'])

            # cross supervise
            pred_A = torch.argmax(output_A_u['preds'][-1].detach(), dim=1)
            pred_B = torch.argmax(output_B_u['preds'][-1].detach(), dim=1)
            usp_loss = get_loss(output_A_u['preds'], pred_B, loss_func) + \
                get_loss(output_B_u['preds'], pred_A, loss_func)
            
            loss_unlabeled = w_cps * usp_loss / 2.0
            loss_unlabeled.backward()

            '''--- optimize ---'''
            loss = loss_labeled + loss_unlabeled
            loss_list.append(loss.item())

            optimizer_A.step()
            optimizer_B.step()

        loss_sup_A = np.mean(loss_sup_list['A'])
        loss_sup_B = np.mean(loss_sup_list['B'])
        print(f'  --- epoch: {epoch}, w_cps: {w_cps}, loss_sup (A): {loss_sup_A}, loss_sup (B): {loss_sup_B}, loss: {np.mean(loss_list)}')

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
