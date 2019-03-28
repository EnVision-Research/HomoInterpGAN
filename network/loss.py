import torch
from torch import nn
from util.logger import logger
import torch.nn.functional as F
from network import base_network

log = logger()


def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    loss = F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    return loss


def classification_loss_list(logit, target):
    '''
    compute the classification loss when logit and target are lists, e.g., [[1,2,3],[4],[5,6]]
    '''
    loss = 0
    for logit_now, target_now in zip(logit, target):
        loss += classification_loss(logit_now, target_now)
    return loss

class perceptural_loss(nn.Module):
    def __init__(self):
        super(perceptural_loss, self).__init__()
        self.vgg = base_network.VGG(pretrained=True).cuda()
        self.vgg = nn.DataParallel(self.vgg)
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        feat_x, _, _ = self.vgg(x)
        # feat_x = feat_x[0]
        feat_y, _, _ = self.vgg(y)
        # feat_y = feat_y[0]
        loss = self.mse(feat_x, feat_y.detach())
        return loss