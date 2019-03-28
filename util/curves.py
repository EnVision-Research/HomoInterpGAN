'''
the training, testing and accuracy curves
'''
import numpy as np
import os
from collections import OrderedDict
import tensorboardX
import matplotlib.pyplot as plt
from . import logger
from . import util
log = logger.logger()
plt.switch_backend('agg')


class writer(object):

    def __init__(self, log_dir, tensorboard=True):
        super(writer, self).__init__()
        self.tensorboard = tensorboard
        if tensorboard:
            self.tbwriter = tensorboardX.SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        self.curve_dict = {}
        util.mkdir(log_dir)

    def add_scalar(self, tag, scalar_value, global_step=0, overwrite=False):
        scalar_value = float(scalar_value)
        if self.tensorboard:
            self.tbwriter.add_scalar(tag, scalar_value, global_step)
        save_path = os.path.join(self.log_dir, tag) + '.txt'
        if not os.path.exists(save_path):
            mode = 'w'
        elif overwrite:
            mode = 'w'
        else:
            mode = 'a'
        with open(save_path, mode) as f:
            f.write('{},{}\n'.format(scalar_value, global_step))
        if tag not in self.curve_dict:
            value_list, global_step_list = self._read_file(save_path)
            self.curve_dict[tag] = []
            for v, g in zip(value_list, global_step_list):
                self.curve_dict[tag] += [g, v]
        if mode == 'w':
            self.curve_dict[tag] = [(global_step, scalar_value)]
        else:
            self.curve_dict[tag] += [(global_step, scalar_value)]

        self.plot(tag, True)

    def _read_file(self, save_path):
        value_list = []
        global_step_list = []
        with open(save_path, 'r') as f:
            for data in f:
                value, global_step = data.split(',')
                value_list += [float(value)]
                global_step_list += [float(global_step)]
        return value_list, global_step_list

    def _curve_dict_to_list(self, tag):
        global_step_list = []
        value_list = []
        try:
            for g, v in self.curve_dict[tag]:
                global_step_list += [g]
                value_list += [v]
        except:
            global_step_list = [-1]
            value_list = [-1]
        return value_list, global_step_list

    def maxmin_value(self, tag, fn=max):
        value_list, global_step_list = self._curve_dict_to_list(tag)
        value = fn(value_list)
        global_step = global_step_list[value_list.index(value)]
        return value, global_step

    def plot(self, tag, if_save=True):
        ''' plot the curve by reading the tag '''
        ''' read the file from tag '''
        save_path = os.path.join(self.log_dir, tag) + '.txt'
        if tag in self.curve_dict:
            value_list, global_step_list = self._curve_dict_to_list(tag)
        else:
            value_list, global_step_list = self._read_file(save_path)
        # log(value_list)
        plt.plot(np.asarray(global_step_list), np.asarray(value_list))
        plt.xlabel('evaluation step')
        plt.ylabel(tag)
        plt.tight_layout()
        plt.legend(loc='upper right')
        if if_save:
            plt.savefig(save_path[:-4] + '.jpg')
            plt.clf()

    def multiplot(self, tag_list):
        ''' plot multiple curve in a file list '''
        plt.clf()
        for tag in tag_list:
            self.plot(tag, if_save=False)
        plt.legend(tuple(tag_list))
        name = '_'.join(tag_list)
        save_path = os.path.join(self.log_dir, name + '.jpg')
        plt.savefig(save_path)
        plt.clf()
