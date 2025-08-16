import torch
import torch.nn as nn

from models.HGFilters import HGFilter
from models.SurfaceClassifier import SurfaceClassifier



def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]


class HGPIFuNet_xray(nn.Module):
    def __init__(self, in_ch=2, num_cls=5, kd_dim=64):
        super().__init__()

        self.n_stack = 4
        filter_out = 256
        self.frontal_filter = HGFilter(in_ch=in_ch, n_stack=self.n_stack, out_ch=filter_out)
        self.lateral_filter = HGFilter(in_ch=in_ch, n_stack=self.n_stack, out_ch=filter_out)
        
        p_in = 2 * filter_out
        mlp_dim = [p_in, 512, 128, num_cls]
        self.surface_classifier = SurfaceClassifier(
            filter_channels=mlp_dim,
            no_residual=False
        )

        self.use_kd = False
        if kd_dim is not None:
            self.use_kd = True
            self.projector = nn.Conv1d(mlp_dim[0], kd_dim, kernel_size=1, bias=False)

    def encode_2d(self, images):
        frontal_feats, _ = self.frontal_filter(images[:, 0, ...])
        lateral_feats, _ = self.lateral_filter(images[:, 1, ...])
        self.image_feats = [frontal_feats, lateral_feats]
        
        if not self.training:
            for i in range(2):
                self.image_feats[i] = [self.image_feats[i][-1]]

    def query(self, points_proj):
        if self.use_kd:
            self.point_feat_list = []

        self.pred_list = []
        for i in range(self.n_stack if self.training else 1):
            feat_list = []
            for k in range(2):
                view_feats = self.image_feats[k][i]
                xy = points_proj[:, k, ...]
                feat_list.append(index_2d(view_feats, xy))

            point_feats = torch.cat(feat_list, dim=1)
            point_preds = self.surface_classifier(point_feats)
            self.pred_list.append(point_preds)

            if self.use_kd:
                self.point_feat_list.append(self.projector(point_feats))

    def get_pred(self, part=None):
        pred = self.pred_list[-1]
        pred = torch.argmax(pred[:, :5, :], dim=1) # B, N
        if part is None:
            pred = (pred > 0).float()
        else:
            pred = (pred == part).float()
        
        pred = pred - 0.5
        return pred

    def forward(self, xrays, points_proj):
        # xray: [B, M, 2, H, W]
        # points_proj: [B, M, N, 2]
        self.encode_2d(xrays)
        self.query(points_proj)
        return {
            'points_pred': self.pred_list,
            'encoding_feats': self.point_feat_list if self.use_kd else None
        }
