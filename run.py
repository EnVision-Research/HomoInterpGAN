from optimizer import optim_homoInterp
import glob
from network import model
import argparse
import os
from data import attributeDataset
from util import util
from torch.utils.data import DataLoader
from util import tensorWriter
import torch
from torch import nn


class Engine(object):

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        shared_parser = argparse.ArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(dest='command')
        ''' shared arguments '''
        # parser.add_argument('mode', choices=['train', 'test'], default='train', help='training/testing')
        shared_parser.add_argument('-gpu', default='0', help='the gpu index to run')
        shared_parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
        shared_parser.add_argument('--data_dir',
                                   default='/mnt/backup/project/ycchen/datasets/face/images/celeba_aligned',
                                   help='celeba_aligned dataset dir')
        shared_parser.add_argument('--attr',
                                   default='Mouth_Slightly_Open@Smiling,Male@No_Beard@Mustache@Goatee@Sideburns,Black_Hair@Blond_Hair@Brown_Hair@Gray_Hair,Bald@Receding_Hairline@Bangs,Young',
                                   help='Target attributes. We use \"@\" to split single attributes, and use \",\" to split grouped attributes')
        shared_parser.add_argument('--dec_type', default='v1')
        ''' arguments for training '''
        parser_train = subparsers.add_parser('train', parents=[shared_parser])
        parser_train.add_argument('-sp', '--save_dir', default='checkpoints/default2', help='model save path')
        parser_train.add_argument('-ct', '--continue_train', action='store_true',
                                  help='if true, then load model stored in the save_dir')
        parser_train.add_argument('-e', '--epoch', type=int, default=10000, help='the training epoch')

        ''' arguments for testing '''
        shared_parser_test = argparse.ArgumentParser(add_help=False)
        shared_parser_test.add_argument('-mp', '--model_path', help='this path stores *.pth files')
        shared_parser_test.add_argument('--pth_label', default='latest', help='The label of the pth files. ')
        shared_parser_test.add_argument('-sp', '--save_dir', help='path for saving output images')
        parser_test_selected_curve = subparsers.add_parser('test_selected_curve',
                                                           parents=[shared_parser, shared_parser_test])
        parser_test_selected_curve.add_argument('-bl', '--branch_list', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                                                help='The order indicates the interpolation curve. ')

        parser_attr_manipulation = subparsers.add_parser('attribute_manipulation',
                                                         parents=[shared_parser, shared_parser_test])
        parser_attr_manipulation.add_argument('--filter_target_attr', required=True,
                                              help='To filter the target attribute.')
        parser_attr_manipulation.add_argument('-s', '--strength', type=float, default=1,
                                              help='The edit strength. It is suggested to be larger than 0 and smaller than 2')
        parser_attr_manipulation.add_argument('--branch_idx', type=int, required=True, help='Specify the used  branch')
        parser_attr_manipulation.add_argument('--n_ref', type=int, default=1, help='The number of reference images')
        parser_attr_manipulation.add_argument('--test_folder', default=None, help='The folder of testing image. If not specified, we randomly use the testing set of celeba')
        parser_attr_manipulation.add_argument('--n_test', type=int, default=10, help='The maximum number of testing images.')
        return parser

    def basic_setting(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        util.mkdir(self.args.save_dir)

    def define_model(self):
        encoder = model.encoder()
        n_branch = len(self.args.attr.split(','))
        interp = model.interp_net(
            n_branch=n_branch + 1)  # +1 because it needs at least one residual branch to model other attributes
        if self.args.dec_type == 'v1':
            decoder = model.decoder()
        elif self.args.dec_type == 'v2':
            decoder = model.decoder2()
        else:
            raise NotImplementedError
        discrim = model.discrim(self.args.attr)
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        interp = nn.DataParallel(interp)
        # discrim = nn.DataParallel(discrim)

        return encoder, interp, decoder, discrim

    def define_optim(self, model):
        optim = optim_homoInterp.optimizer(model, self.args)
        return optim

    def load_dataset(self):
        with open('info/celeba-train-flip.txt', 'r') as f:
            train_list = [os.path.join(self.args.data_dir, tmp.rstrip()) for tmp in f]
        with open('info/celeba-test-flip.txt', 'r') as f:
            test_list = [os.path.join(self.args.data_dir, tmp.rstrip()) for tmp in f]
        train_dataset = attributeDataset.GrouppedAttrDataset(image_list=train_list, attributes=self.args.attr,
                                                             csv_path='info/celeba-with-orientation.csv')
        test_dataset = attributeDataset.GrouppedAttrDataset(image_list=test_list, attributes=self.args.attr,
                                                            csv_path='info/celeba-with-orientation.csv')
        return train_dataset, test_dataset

    def train(self):
        model = self.define_model()
        optim = self.define_optim(model)
        train_dataset, test_dataset = self.load_dataset()
        optim.test_dataset = test_dataset
        from util import training_framework
        TrainEngine = training_framework.TrainEngine(dataset=train_dataset, optimizer=optim,
                                                     batch_size=self.args.batch_size)
        TrainEngine.run(self.args.epoch)

    def test_selected_curve(self):
        '''
        given a training and a reference image, synthesize the intermediate results with the selected interpolation curve.
        :return:
        '''
        util.mkdir(self.args.save_dir)
        n_branch = len(self.args.attr.split(',')) + 1
        test_model = model.model_deploy(n_branch=n_branch, model_path=self.args.model_path,
                                        label=self.args.pth_label).eval().cuda()
        _, test_dataset = self.load_dataset()
        loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        for i, data in enumerate(loader):
            img, _ = data
            img = util.toVariable(img).cuda()
            idx = torch.randperm(img.size(0)).cuda()
            img_ref = img[idx]
            img_out = [img]
            v = torch.zeros(img.size(0), n_branch)
            for j in self.args.branch_list:
                v[:, j:j + 1] = 1
                v = util.toVariable(v).cuda()
                out_now = test_model(img, img_ref, v)
                img_out += [out_now]
            img_out += [img_ref]
            img_out = torch.cat(img_out)
            img_out = tensorWriter.untransformTensor(img_out.data.cpu())
            tensorWriter.writeTensor('{}/{}.jpg'.format(self.args.save_dir, i), img_out, nRow=img.size(0))

    def attribute_manipulation(self):
        '''
        perform attribute manipulation
        :return:
        '''
        from data.attributeDataset import Dataset_testing_filtered, Dataset_testing
        util.mkdir(self.args.save_dir)

        n_branch = len(self.args.attr.split(',')) + 1  # n groupped attribute + 1 residual attribute.
        test_model = model.model_deploy(n_branch=n_branch, model_path=self.args.model_path,
                                        label=self.args.pth_label).eval().cuda()
        if self.args.test_folder is None:
            _, test_dataset = self.load_dataset()
        else:
            image_list = glob.glob(self.args.test_folder+'/*.jpg')
            test_dataset = Dataset_testing(image_list)
        ref_dataset = Dataset_testing_filtered(self.args.data_dir, self.args.filter_target_attr, n_samples=self.args.n_ref)
        loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True)
        ref_loader = DataLoader(ref_dataset, batch_size=1, shuffle=True)
        # for data, ref_data in zip(loader, ref_loader):
        img_out = [tmp for tmp in ref_loader]
        img_out = torch.cat(img_out)
        img_out = tensorWriter.untransformTensor(img_out.data.cpu())
        tensorWriter.writeTensor('{}/reference.jpg'.format(self.args.save_dir), img_out, nRow=1)

        for i, data in enumerate(loader):
            img, _ = data
            img = util.toVariable(img).cuda()
            img_out = [img]
            for ref_data in ref_loader:
                print('proceesing the {}-th batch'.format(i))
                img_ref = ref_data
                img_ref = util.toVariable(img_ref).cuda()
                v = torch.zeros(img.size(0), n_branch)
                v[:, self.args.branch_idx:self.args.branch_idx + 1] = self.args.strength
                v = util.toVariable(v).cuda()
                img_ref_now = img_ref.expand_as(img)
                out_now = test_model(img, img_ref_now, v)
                img_out += [out_now]
                # img_out += [img_ref]
            img_out = torch.cat(img_out)
            img_out = tensorWriter.untransformTensor(img_out.data.cpu())
            tensorWriter.writeTensor('{}/{}.jpg'.format(self.args.save_dir, i), img_out, nRow=img.size(0))
            i += 1
            if i > self.args.n_test:
                break

    def run(self):
        parser = self.parse_args()
        self.args = parser.parse_args()
        self.basic_setting()
        exec ('self.{}()'.format(self.args.command))


if __name__ == '__main__':
    engine = Engine()
    engine.run()
