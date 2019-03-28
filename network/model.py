'''
define the models for the homomorphic interpolation model
'''
import torch
from torch import nn
from collections import OrderedDict
from network import base_network

class encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(encoder, self).__init__()
        self.model = base_network.VGG(pretrained=pretrained)

    def forward(self, x):
        y = self.model(x)
        y = y[-1]
        return y


class _interp_branch(nn.Module):
    '''
    one branch of the interpolator network
    '''

    def __init__(self, in_channels, out_channels):
        super(_interp_branch, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
    def forward(self, x):
        return self.model(x)

class interp_net(nn.Module):
    def __init__(self, n_branch, channels=512):
        '''
        the multi-branch interpolator network
        :param channels: channels of the latent space.
        :param n_branch: number of branches. each branch deals with an attribtue
        '''
        super(interp_net, self).__init__()
        self.n_branch = n_branch
        branch = []
        branch_fn = _interp_branch
        for i in range(n_branch):
            branch += [branch_fn(channels, channels)]
        self.branch = nn.ModuleList(branch)

    def forward(self, feat1, feat2, selective_vector, **kwargs):
        y = feat2 - feat1
        selective_tensor = selective_vector.unsqueeze(2).unsqueeze(3)
        selective_tensor = selective_tensor.expand((-1, -1, y.size(2), y.size(3)))
        z = []
        for i in range(self.n_branch):
            tmp = self.branch[i](y)
            tmp = tmp * selective_tensor[:, i:i + 1, :, :]
            z += [tmp]
        z = feat1 + sum(z)
        return z


class decoder(nn.Module):
    def __init__(self, upsample_mode='nearest'):
        super(decoder, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        ])
    def forward(self, x, img=None):
        y = self.model(x)
        return y



class discrim(nn.Module):
    '''
    attr is the input attribute list that is consistent with data/attreibuteDataset/Dataset_attr_merged_v2,
    e.g., Moustache@#No_Beard@Goatee,Smile,Young,Bangs
    '''

    def __init__(self, attr):
        super(discrim, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 4, padding=1, stride=2),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 4, padding=1, stride=2),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 2)
        )
        # self.model = nn.Sequential(
        #     nn.Conv2d(512, 256, 1),
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 256, 14),
        # )
        self.ifReal = nn.Conv2d(256, 256, 1)
        # self.attribute = nn.Conv2d(256, n_attributes, 1)
        attr_branches = []
        attr = attr.split(',')
        for i in range(len(attr)):
            attr_now = attr[i].split('@')
            branch_now = nn.Conv2d(256, len(attr_now), 1)
            attr_branches += [branch_now]
        attr_branches = nn.ModuleList(attr_branches)
        self.attr_branches = attr_branches
        self.model = self.model
        self.ifReal = self.ifReal

    def forward(self, x):
        y = self.model(x)
        ifReal = self.ifReal(y)
        attributes = []
        for branch_now in self.attr_branches:
            attribute_now = branch_now(y).squeeze(2).squeeze(2)
            attributes += [attribute_now]
        return ifReal, attributes
