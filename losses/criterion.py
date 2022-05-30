import torch.nn as nn
import torch.nn.functional as f
from pywick import losses as ls


class GiangPolyCriterion(nn.Module):
    def __init__(self, weights=[0.5, 0.5]):
        super(GiangPolyCriterion, self).__init__()
        self.bce = ls.BCELoss2d()
        self.tversky = ls.TverskyLoss(alpha=0.4, beta=0.6)
        self.weights = weights

    def forward(self, logits, targets):
        return self.weights[0] * self.bce(logits, targets), \
               self.weights[1] * self.tversky(logits, targets.long())
