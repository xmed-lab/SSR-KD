import torch
import torch.nn as nn

from models.VNet import VNet
from models.SurfaceClassifier import SurfaceClassifier


def index_3d(feat, uv):
    '''
    :param feat: [B, C, H, W, D] image features
    :param uv: [B, 3, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 3]
    uv = uv.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]; 5-d case
    feat = feat.transpose(2, 4)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1, 1]
    return samples[:, :, :, 0, 0]  # [B, C, N]


class HGPIFuNet_3d(nn.Module):
    def __init__(self, in_ch=1, num_cls=5):
        super().__init__()

        self.im_filter = VNet(n_channels=in_ch)

        mlp_dim = [64, 128, 256, 128, 64]
        self.surface_classifier = SurfaceClassifier(
            filter_channels=mlp_dim + [num_cls],
            no_residual=False,
            last_op=None
        )

    def encode_3d(self, image):
        self.feat_list = self.im_filter(image)

    def query(self, points):
        self.point_feat_list = []
        self.pred_list = []
        for ct_feats in self.feat_list:
            point_feats = index_3d(ct_feats, points)
            point_preds = self.surface_classifier(point_feats)
            self.pred_list.append(point_preds)
            self.point_feat_list.append(point_feats)
    
    def get_pred(self, part=None):
        pred = self.pred_list[-1]
        pred = torch.argmax(pred, dim=1)
        if part is None:
            pred = (pred > 0).float()
        else:
            pred = (pred == part).float()
        
        pred = pred - 0.5
        return pred

    def forward(self, ct, points):
        self.encode_3d(ct)
        self.query(points)
        return {
            'points_pred': self.pred_list,
            'encoding_feats': self.point_feat_list
        }
