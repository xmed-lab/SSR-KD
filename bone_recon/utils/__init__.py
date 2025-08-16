import numpy as np
from torch import nn


def load_ckpt_safe(model, ckpt):
    model_dict = model.state_dict()
    ckpt = {k : v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(ckpt)
    model.load_state_dict(model_dict)
    return model


def random_select(arr, size=-1, choice=None):
    if choice is None:
        if size > len(arr):
            size = len(arr)
        if choice is None:
            choice = np.random.choice(len(arr), size=size, replace=False)
    return arr[choice], choice # copy


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
    return model
