import math
import glob
import re
import warnings
import torch
import torch.nn as nn
from pathlib import Path


def norm_cdf(x):
    return (1. + math.erf(x / math.sqrt(2.))) / 2.


def no_grad_trunc_normal_(tensor, mean, std, a, b):
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return no_grad_trunc_normal_(tensor, mean, std, a, b)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def iou(label, pred, lis, smooth=1e-9):
    segment = torch.sigmoid(pred)
    segment = torch.where(segment > 0.5, 1, 0)
    for i, seg in enumerate(segment):
        intersaction = torch.sum((seg > 0) & (label[i] > 0))
        union = torch.sum((seg > 0) | (label[i] > 0))
        x = intersaction / (union + smooth)
        lis.append(x.item())
    return lis


def dice(label, pred, lis, smooth=1e-9):
    segment = torch.sigmoid(pred)
    segment = torch.where(segment > 0.5, 1, 0)
    for i, seg in enumerate(segment):
        intersaction = torch.sum((seg > 0) & (label[i] > 0))
        union = torch.sum((seg > 0) | (label[i] > 0))
        x = 2 * intersaction / (intersaction + union + smooth)
        lis.append(x.item())
    return lis


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


if __name__ == '__main__':
    label = torch.ones(1, 1, 256, 256)
    pred = torch.zeros(1, 1, 256, 256)
    pred[:, :, 128:192, 128:192] = 1
    lis = []
    dice(label, pred, lis)
    print(lis)
    lis = []
    iou(label, pred, lis)
    # iou()
    print(lis)
