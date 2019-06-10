from optimizer import optim_homoInterp
from network import model
import argparse
import os
from data import attributeDataset
from util import util


class Engine(object):

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('mode', choices=['train', 'test'], default='train', help='training/testing')
        parser.add_argument('-gpu', default='0', help='the gpu index to run')
        parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('-sp', '--save_dir', default='checkpoints/default2', help='model save path')
        parser.add_argument('-ct', '--continue_train', action='store_true',
                            help='if true, then load model stored in the save_dir')
        parser.add_argument('-e', '--epoch', type=int, default=10000, help='the training epoch')

        parser.add_argument('--data_dir', help='celeba_aligned dataset dir')
        parser.add_argument('--attr',
                            default='Mouth_Slightly_Open@Smiling,Male@No_Beard@Mustache@Goatee@Sideburns,Black_Hair@Blond_Hair@Brown_Hair@Gray_Hair,Bald@Receding_Hairline@Bangs,Young',
                            help='Target attributes. We use \"@\" to split single attributes, and use \",\" to split grouped attributes')
        return parser

    def basic_setting(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        util.mkdir(self.args.save_dir)

    def define_model(self):
        encoder = model.encoder()
        n_branch = len(self.args.attr.split(','))
        interp = model.interp_net(n_branch=n_branch+1)  # +1 because it needs at least one residual branch to model other attributes
        decoder = model.decoder()
        discrim = model.discrim(self.args.attr)
        return encoder, interp, decoder, discrim

    def define_optim(self, model):
        from optimizer import train_decoder
        optim = train_decoder.optimizer(model, self.args)
        return optim

    def load_dataset(self):
        with open('info/celeba-train.txt', 'r') as f:
            train_list = [os.path.join(self.args.data_dir, tmp.rstrip()) for tmp in f]
        with open('info/celeba-test.txt', 'r') as f:
            test_list = [os.path.join(self.args.data_dir, tmp.rstrip()) for tmp in f]
        train_dataset = attributeDataset.GrouppedAttrDataset(image_list=train_list, attributes=self.args.attr, csv_path='info/celeba.csv')
        test_dataset = attributeDataset.GrouppedAttrDataset(image_list=test_list, attributes=self.args.attr, csv_path='info/celeba.csv')
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

    def test(self):
        pass

    def run(self):
        parser = self.parse_args()
        self.args = parser.parse_args()
        self.basic_setting()
        if self.args.mode == 'train':
            self.train()
        elif self.args.mode == 'test':
            self.test()
        else:
            raise NotImplementedError


if __name__ == '__main__':
    engine = Engine()
    engine.run()
