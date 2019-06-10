'''
the trainer for the homomorphic interpolation model
'''
from __future__ import print_function
import torch
from torch import nn
from optimizer.base_optimizer import base_optimizer
from network import base_network
from network.loss import classification_loss_list, perceptural_loss
from util import util, logger, opt, curves
import os
import util.tensorWriter as vs
from collections import OrderedDict

log = logger.logger()


class optimizer(base_optimizer):

    def __init__(self, model, option=opt.opt()):
        super(optimizer, self).__init__()
        self._default_opt()
        self.opt.merge_opt(option)
        self._get_model(model)
        self._get_aux_nets()
        self._define_optim()
        self.writer = curves.writer(log_dir=self.opt.save_dir + '/log')
        if self.opt.continue_train:
            self.load()

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.save_dir = '/checkpoints/default'
        self.opt.n_discrim = 5

    def set_input(self, input):
        self.image, self.attribute = input
        self.image = util.toVariable(self.image).cuda()
        self.attribute = [att.cuda() for att in self.attribute]

    def zero_grad(self):
        self.encoder.zero_grad()
        self.interp_net.zero_grad()
        self.decoder.zero_grad()
        self.discrim.zero_grad()
        self.KGTransform.zero_grad()

    def _get_aux_nets(self):
        self.vgg_teacher = nn.DataParallel(base_network.VGG(pretrained=True))
        self.perceptural_loss = perceptural_loss()
        self.KGTransform = nn.DataParallel(nn.Conv2d(512, 512, 1))

    def _get_model(self, model):
        encoder, interp_net, decoder, discrim = model
        self.encoder = encoder.cuda()
        self.interp_net = interp_net.cuda()
        self.decoder = decoder.cuda()
        self.discrim = discrim.cuda()
        with open(self.opt.save_dir + '/encoder.txt', 'w') as f:
            print(encoder, file=f)
        with open(self.opt.save_dir + '/interp_net.txt', 'w') as f:
            print(interp_net, file=f)
        with open(self.opt.save_dir + '/decoder.txt', 'w') as f:
            print(decoder, file=f)
        with open(self.opt.save_dir + '/discrim.txt', 'w') as f:
            print(discrim, file=f)

    def _define_optim(self):
        self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_interp = torch.optim.Adam(self.interp_net.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_discrim = torch.optim.Adam(self.discrim.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_KGTransform = torch.optim.Adam(self.KGTransform.parameters(), lr=1e-4, betas=[0.5, 0.999])

    def load(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        # self._check_and_load(self.encoder, save_dir.format('encoder', label))
        # self._check_and_load(self.interp_net, save_dir.format('interp_net', label))
        self._check_and_load(self.decoder, save_dir.format('decoder', label))
        # self._check_and_load(self.discrim, save_dir.format('discrim', label))
        # self._check_and_load(self.KGTransform, save_dir.format('KGTransform', label))

    def save(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        # torch.save(self.encoder.state_dict(), save_dir.format('encoder', label))
        # torch.save(self.interp_net.state_dict(), save_dir.format('interp_net', label))
        torch.save(self.decoder.state_dict(), save_dir.format('decoder', label))
        # torch.save(self.discrim.state_dict(), save_dir.format('discrim', label))
        # torch.save(self.KGTransform.state_dict(), save_dir.format('KGTransform', label))

    def optimize_parameters(self, global_step):
        self.encoder.train()
        self.interp_net.train()
        self.discrim.train()
        self.decoder.train()
        self.KGTransform.train()
        self.loss = OrderedDict()
        ''' define v '''
        self.v = self.generate_select_vector()
        self.rand_idx = torch.randperm(self.image.size(0)).cuda()
        self.image_permute = self.image[self.rand_idx]
        self.attr_permute = []
        for att in self.attribute:
            self.attr_permute += [att[self.rand_idx]]
        ''' compute the target attributes '''
        self.attr_interp = []
        for i, (att, attp) in enumerate(zip(self.attribute, self.attr_permute)):
            self.attr_interp += [att + self.v[:, i:i + 1] * (attp - att)]
        ''' pre-computed variables '''
        self.feat = self.encoder(self.image)
        self.feat_permute = self.feat[self.rand_idx]
        self.feat_interp = self.interp_net(self.feat, self.feat_permute, self.v)

        self.zero_grad()
        self.compute_dec_loss().backward(retain_graph=True)
        self.optim_decoder.step()

        # self.zero_grad()
        # self.compute_discrim_loss().backward(retain_graph=True)
        # self.optim_discrim.step()
        #
        # self.zero_grad()
        # self.compute_KGTransform_loss().backward(retain_graph=True)
        # self.optim_KGTransform.step()
        #
        # if global_step % self.opt.n_discrim == 0:
        #     self.zero_grad()
        #     self.compute_enc_int_loss().backward()
        #     self.optim_encoder.step()
        #     self.optim_interp.step()

    def compute_dec_loss(self):
        self.loss['dec'] = 0
        im_out = self.decoder(self.feat, self.image)
        self.loss['dec_per'] = self.perceptural_loss(im_out, self.image)
        self.loss['dec'] += self.loss['dec_per']
        self.loss['dec_mse'] = nn.MSELoss()(im_out, self.image.detach())
        self.loss['dec'] += self.loss['dec_mse']
        return self.loss['dec']

    def compute_discrim_loss(self):
        self.loss['discrim'] = 0
        discrim_real, real_attr = self.discrim(self.feat.detach())
        discrim_interp, interp_attr = self.discrim(self.feat_interp.detach())
        ''' gradient penality '''
        gp_interpolate = self.random_interpolate(self.feat.data, self.feat_interp.data)
        gp_interpolate = util.toVariable(gp_interpolate, requires_grad=True)
        discrim_gp_interpolated, _ = self.discrim(gp_interpolate)
        self.loss['discrim_gp'] = util.gradient_penalty(gp_interpolate, discrim_gp_interpolated) * 100.
        self.loss['discrim'] += self.loss['discrim_gp']
        ''' the GAN loss '''
        self.loss['discrim_gan'] = discrim_interp.mean() - discrim_real.mean()
        self.loss['discrim'] += self.loss['discrim_gan']
        ''' the attribute classification loss '''
        att_detach = [att.detach() for att in self.attribute]
        self.loss['discrim_cls'] = classification_loss_list(interp_attr, att_detach)
        self.loss['discrim'] += self.loss['discrim_cls']
        return self.loss['discrim']

    def compute_KGTransform_loss(self):
        feat_T1 = self.vgg_teacher(self.image)[-1]
        feat_T2 = self.KGTransform(self.feat)
        self.loss['KGTransform'] = nn.MSELoss()(feat_T2, feat_T1.detach())
        return self.loss['KGTransform']

    def compute_enc_int_loss(self):
        self.loss['enc_int'] = 0
        # discrim_real, out_attr = self.discrim(self.feat)
        discrim_interp, interp_attr = self.discrim(self.feat_interp)
        ''' GAN loss '''
        self.loss['enc_int_gan'] = -discrim_interp.mean()
        self.loss['enc_int'] += self.loss['enc_int_gan']
        ''' classification loss '''
        interp_detach = [att.detach() for att in self.attr_interp]
        self.loss['enc_int_cls'] = classification_loss_list(interp_attr, interp_detach)
        self.loss['enc_int'] += self.loss['enc_int_cls']
        ''' interp all loss '''
        feat_interp_all = self.interp_net(self.feat.detach(), self.feat_permute.detach(),
                                          self.generate_select_vector(type='select_all'))
        self.loss['enc_int_all'] = nn.MSELoss()(feat_interp_all, self.feat_permute.detach())
        self.loss['enc_int'] += self.loss['enc_int_all']
        ''' reconstruction loss '''
        im_out = self.decoder(self.feat)
        self.loss['enc_int_mse'] = nn.MSELoss()(im_out, self.image.detach())
        self.loss['enc_int'] += self.loss['enc_int_mse']
        self.loss['enc_int_per'] = self.perceptural_loss(im_out, self.image.detach())
        self.loss['enc_int'] += self.loss['enc_int_per']
        ''' knowledge guidance loss '''
        feat_T1 = self.vgg_teacher(self.image)[-1]
        feat_T2 = self.KGTransform(self.feat)
        self.loss['enc_int_KG'] = nn.MSELoss()(feat_T2, feat_T1.detach())
        self.loss['enc_int'] += self.loss['enc_int_KG']
        return self.loss['enc_int']

    def get_current_errors(self):
        return self.loss

    def interp_test(self, img1, img2):
        '''
        testing the interpolation effect.
        :param type: "single" and "accumulate"
        :return: a torch image that combines the interpolation results
        '''
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        result_map = []
        n_branches = self.interp_net.n_branch
        for attr_idx in range(n_branches):
            result_row = [img1.data.cpu()]
            for strength in [0, 0.5, 1]:
                attr_vec = torch.zeros(1, n_branches + 1)
                attr_vec[:, attr_idx] = strength
                attr_vec = util.toVariable(attr_vec).cuda()
                interp_feat = self.interp_net(feat1, feat2, attr_vec)
                out_tmp = self.decoder(interp_feat, img1)
                result_row += [out_tmp.data.cpu()]
            result_row += [img2.data.cpu()]
            result_row = torch.cat(result_row, dim=3)
            result_map += [result_row]
        result_row = [img1.data.cpu()]
        # interpolate all the attributes
        for strength in [0, 0.5, 1]:
            attr_vec = torch.ones(1, n_branches) * strength
            attr_vec = util.toVariable(attr_vec).cuda()
            interp_feat = self.interp_net(feat1, feat2, attr_vec)
            out_tmp = self.decoder(interp_feat, img1)
            result_row += [out_tmp.data.cpu()]
        result_row += [img2.data.cpu()]
        result_row = torch.cat(result_row, dim=3)
        result_map += [result_row]

        result_map = torch.cat(result_map, dim=2)
        return result_map

    def save_samples(self, global_step=0):
        self.encoder.eval()
        self.interp_net.eval()
        self.discrim.eval()
        self.decoder.eval()
        n_branches = self.interp_net.n_branch
        save_path = os.path.join(self.opt.save_dir, 'interp_curve')
        util.mkdir(save_path)
        im_out = [self.image.data.cpu()]
        v = torch.zeros(self.image.size(0), n_branches)
        v = util.toVariable(v).cuda()
        feat = self.interp_net(self.feat, self.feat_permute, v)
        out_now = self.decoder(feat, self.image)
        im_out += [out_now.data.cpu()]

        im_out = [util.toVariable(tmp) for tmp in im_out]
        im_out = torch.cat(im_out, dim=0)
        im_out = vs.untransformTensor(im_out.data.cpu())
        vs.writeTensor('%s/%d.jpg' % (save_path, global_step), im_out, nRow=self.image.size(0))

    def _generate_select_vector(self, n_branches, type='uniform'):
        '''
        generate the select vector to select the interpolation curve
        type:

        :return: nSample x selct_dims, which indicates which attribute to be transferred.
        '''

        if type == 'one_attr_randsample':  # each sample has one random selected attribute
            selected_vector = []
            for i in range(self.image.size(0)):
                tmp = torch.randperm(n_branches)[0]  # randomly select one attribute
                # log('generate_select_vector: tmp:', tmp)
                one_hot_vec = torch.zeros(1, n_branches)
                one_hot_vec[:, tmp] = 1
                # log('generate_select_vector: one_hot_vec:', one_hot_vec)
                selected_vector += [one_hot_vec]
            selected_vector = torch.cat(selected_vector, dim=0)
            # log('one-attr-randsample', selected_vector)
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'one_attr_batch':  # each batch has one common selected attribute
            raise NotImplemented
        elif type == 'uniform':
            selected_vector = torch.rand(self.image.size(0), n_branches)
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'uniform_binarize':
            selected_vector = torch.rand(self.image.size(0), n_branches)
            selected_vector = (selected_vector > 0.5).float() * 1.
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'select_all':
            selected_vector = torch.ones(self.image.size(0), n_branches)
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'select_none':
            selected_vector = torch.zeros(self.image.size(0), n_branches)
            # log('selective_none', torch.sum(selected_vector))
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        else:
            raise NotImplemented

    def generate_select_vector(self, type='uniform'):
        n_branches = self.interp_net.n_branch
        return self._generate_select_vector(n_branches, type)

    def random_interpolate(self, gt, pred):
        batch_size = gt.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).cuda()
        # alpha = alpha.expand(gt.size()).cuda()
        interpolated = gt * alpha + pred * (1 - alpha)
        return interpolated