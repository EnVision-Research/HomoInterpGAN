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
                                   # nn.LeakyReLU(1e-1, True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                   nn.ReLU(True),
                                   # nn.LeakyReLU(1e-1, True),
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
    def __init__(self, upsample_mode='nearest', pretrained=True):
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
        if pretrained:
            model_path = 'https://www.dropbox.com/s/8lwmwfs42w5oioi/homo-decoder-8a84d0ce.pth?dl=1'
            state_dict = torch.utils.model_zoo.load_url(model_path, 'checkpoints/vgg')
            self.load_state_dict(state_dict)

    def forward(self, x):
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


class decoder2(nn.Module):

    def __init__(self, res=False, pretrained=True):
        super(decoder2, self).__init__()
        self.res = res
        self.model = base_network.Vgg_recon_noskip()
        self.model = nn.DataParallel(self.model).cuda()
        if pretrained:
            print('load decoder_noskip')
            state_dict = torch.load('checkpoints/decoder_noskip.pth')
            self.load_state_dict(state_dict)

    def forward(self, x, image=None):
        y = self.model(x)
        if self.res:
            y += image
        return y


class Vgg_recon_noskip(nn.Module):
    def __init__(self, drop_rate=0, norm='batch'):
        super(Vgg_recon_noskip, self).__init__()

        self.recon5 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate, norm=norm)
        self.upool4 = _TransitionUp(512, 512, norm=norm)
        self.upsample4 = _Upsample(512, 512, norm=norm)
        # self.recon4 = _PoolingBlock(3, 1024, 512, drop_rate = drop_rate)
        self.recon4 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate, norm=norm)
        self.upool3 = _TransitionUp(512, 256, norm=norm)
        self.upsample3 = _Upsample(512, 256, norm=norm)
        self.recon3 = _PoolingBlock(3, 256, 256, drop_rate=drop_rate, norm=norm)
        self.upool2 = _TransitionUp(256, 128, norm=norm)
        self.upsample2 = _Upsample(256, 128, norm=norm)
        self.recon2 = _PoolingBlock(2, 128, 128, drop_rate=drop_rate, norm=norm)
        self.upool1 = _TransitionUp(128, 64, norm=norm)
        self.upsample1 = _Upsample(128, 64, norm=norm)
        self.recon1 = _PoolingBlock(1, 64, 64, drop_rate=drop_rate, norm=norm)
        self.recon0 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, features_3):
        # print('fy', len(fy))

        recon5 = self.recon5(features_3)
        recon5 = nn.functional.upsample(recon5, scale_factor=2, mode='bilinear')
        upool4 = self.upsample4(recon5)

        recon4 = self.recon4(upool4)
        recon4 = nn.functional.upsample(recon4, scale_factor=2, mode='bilinear')
        upool3 = self.upsample3(recon4)

        recon3 = self.recon3(upool3)
        recon3 = nn.functional.upsample(recon3, scale_factor=2, mode='bilinear')
        upool2 = self.upsample2(recon3)

        recon2 = self.recon2(upool2)
        recon2 = nn.functional.upsample(recon2, scale_factor=2, mode='bilinear')
        upool1 = self.upsample1(recon2)

        recon1 = self.recon1(upool1)
        recon0 = self.recon0(recon1)
        return recon0


class model_deploy_container(nn.Module):

    def __init__(self, encoder, interp_net, decoder):
        super(model_deploy_container, self).__init__()
        self.encoder = encoder
        self.interp_net = interp_net
        self.decoder = decoder

    def forward(self, x1, x2, v):
        '''
        :param x1: the testing image
        :param x2: the reference image
        :param v: the control vector
        :return: the output image
        '''
        f1, f2 = self.encoder(x1), self.encoder(x2)
        fout = self.interp_net(f1, f2, v)
        xout = self.decoder(fout)
        return xout


class model_deploy(model_deploy_container):
    ''' the model used for testing '''

    def __init__(self, n_branch, model_path, label='latest', parallel=False):
        nn.Module.__init__(self)
        self.encoder = encoder()
        self.interp_net = interp_net(n_branch)
        self.decoder = decoder(pretrained=False)
        self.encoder = nn.DataParallel(self.encoder)
        self.interp_net = nn.DataParallel(self.interp_net)
        self.decoder = nn.DataParallel(self.decoder)

        self.encoder.load_state_dict(torch.load('{}/encoder-{}.pth'.format(model_path, label)))
        self.interp_net.load_state_dict(torch.load('{}/interp_net-{}.pth'.format(model_path, label)))
        self.decoder.load_state_dict(torch.load('{}/decoder-{}.pth'.format(model_path, label)))
