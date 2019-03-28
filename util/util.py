from __future__ import print_function
import os
import glob
import sys
import fnmatch
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import tempfile
import cv2
import imageio
import math

try:
    from requests.utils import urlparse
    import requests.get as urlopen

    requests_available = True
except ImportError:
    requests_available = False
    if sys.version_info[0] == 2:
        from urlparse import urlparse  # noqa f811
        from urllib2 import urlopen  # noqa f811
    else:
        from urllib.request import urlopen
        from urllib.parse import urlparse


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def npImg2torchVar(img, requires_grad=False, type='cv'):
    '''
    from numpy image to pytorch variable
    :param img: the np image, which has shape height x width x channels
    :param type: type of reading the image, cv is BGR, and IO is RGB
    :return: a variable with shape 1 x channels x height x width
    '''
    if type == 'cv':
        img = img[:, :, [2, 1, 0]]
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = img.transpose((0, 3, 1, 2))  # from batch x height x width x channel to batch x channel x height x width
    img = toVariable(img, requires_grad=requires_grad)
    return img


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


# implement expand_dims for pytorch tensor x
def expand_dims(x, axis):
    shape = list(x.size())
    assert len(shape) >= axis, 'expand_dims error'
    shape.insert(axis, 1)
    y = x.view(shape)
    return y


def filter_not_exist(file_list):
    '''
    filter non-exist file
    :param file_list: list of file path
    :return: file path that is verified exists
    '''
    return [f for f in file_list if os.path.exists(f)]


# convert a unknown object (could be variable) to tensor
def toTensor(obj):
    if isinstance(obj,Variable):
        y = obj.data
    elif isinstance(obj,np.ndarray):
        y = torch.from_numpy(obj)
    elif isinstance(obj, torch.FloatTensor) or isinstance(obj, torch.cuda.FloatTensor):
        y = obj
    elif isinstance(obj, torch.nn.Parameter):
        y = obj.data
    else:
        assert 0, 'type: %s is not supported yet' % type(obj)
    return y


# convert a unknown object (could be variable) to tensor
def toVariable(obj, requires_grad=False):
    if isinstance(obj, Variable):
        y = Variable(obj.data, requires_grad=requires_grad)
    elif type(obj) == np.ndarray:
        y = torch.from_numpy(obj.astype(np.float32))
        y = Variable(y, requires_grad=requires_grad)
    elif isinstance(obj, torch.FloatTensor) or isinstance(obj,torch.cuda.FloatTensor):
        y = Variable(obj, requires_grad=requires_grad)
    elif type(obj) == list or type(obj) == tuple:
        y = []
        for item in obj:
            y += [toVariable(item, requires_grad=requires_grad)]
    else:
        assert 0, 'type: %s is not supported yet' % type(obj)
    return y



def print_network(net, filepath=None):
    if filepath is None:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)
    else:
        num_params = 0
        with open(filepath + '/network.txt', 'w') as f:
            for param in net.parameters():
                num_params += param.numel()
            print(net, file=f)
            f.write('Total number of parameters: %d' % num_params)


from .visualization import writeTensor


# -------------------------- General ---------------------------------#

def readRGB(uri):
    img = cv2.imread(uri)
    return img[:, :, [2, 1, 0]]


def np_onehot_1d(label, codeLen):
    '''
    do onehot encoding wih numpy
    :param label: one dimensional vector (np.array)
    :param codeLen: the length of the one hot code
    :return:
    '''
    return np.eye(codeLen)[label.astype(np.uint16)].astype(np.float32)


def list_reshape(l, nRow):
    '''
    reshape a 1-d list to 2d
    :param l: 1d list
    :param nRow:
    :return: 2d list
    '''
    nRow = int(nRow)
    assert len(l) % nRow == 0, 'size mismatch, len(l)=%d, nRow=%d' % (len(l), nRow)
    nCol = int(len(l) / nRow)
    result = []
    for i in range(nRow):
        result += [l[i * nCol:(i + 1) * nCol]]
    return result


def _download_url_to_file(url, dst):
    u = urlopen(url)
    if requests_available:
        file_size = int(u.headers["Content-Length"])
        u = u.raw
    else:
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta.get_all("Content-Length")[0])

    f = tempfile.NamedTemporaryFile(delete=False)
    with tqdm(total=file_size) as pbar:
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
            pbar.update(len(buffer))
    f.close()
    shutil.move(f.name, dst)


def load_from_url(url, save_dir='facelet_bank'):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(save_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_url_to_file(url, cached_file)
    return torch.load(cached_file)


def center_crop(img, target_size, bias=(0,0)):
    '''
    center crop on numpy data.
    :param img: H x W x C
    :param target_size: h x w
    :param bias: the bias to the center,
    :return: h x w x C
    '''
    diff_x = img.shape[0] - target_size[0]
    diff_y = img.shape[1] - target_size[1]
    start_x = int(diff_x // 2) + bias[0]
    start_y = int(diff_y // 2) + bias[1]
    if len(img.shape) > 2:
        img2 = img[start_x:start_x + target_size[0], start_y:start_y + target_size[1], :]
    else:
        img2 = img[start_x:start_x + target_size[0], start_y:start_y + target_size[1]]
    return img2

def center_paste(img_cropped, img_src, bias):
    '''
    paste img_cropped in the center of img_src
    '''
    nb = img_src.shape[0]
    na = img_cropped.shape[0]
    lower_0 = (nb // 2) - (na // 2) + bias[0]
    upper_0 = (nb // 2) + (na // 2) + bias[0]
    nb = img_src.shape[1]
    na = img_cropped.shape[1]
    lower_1 = (nb // 2) - (na // 2) + bias[1]
    upper_1 = (nb // 2) + (na // 2) + bias[1]
    img_src = img_src.copy()
    if len(img_src.shape) > 2:
        img_src[lower_0:upper_0, lower_1:upper_1, :] = img_cropped
    else:
        img_src[lower_0:upper_0, lower_1:upper_1] = img_cropped
    return img_src

def remove_format_name(filename):
    filename = filename.split('.')
    filename = '.'.join(filename[:-1])
    return filename


def check_exist(file_list):
    for file in file_list:
        if not os.path.exists(file):
            print('file not exit: ' + file)
            return False
    return True


def str2numlist(str_in, type=int):
    dc = []
    dc_str = str_in.split(',')
    for d in dc_str:
        dc += [type(d)]
    return dc


def mkdir(path):
    if not os.path.exists(path):
        print('mkdir %s' % path)
        os.makedirs(path)




def globall(path, pattern):
    '''
    glob all data based on the pattern
    :param path: the root path
    :param pattern: the pattern to filter
    :return: all files that matches the pattern
    '''
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def script_name():
    name = os.path.basename(sys.argv[0])
    name = name[:-3]
    return name


def print_args(ckpt_dir, args):
    '''
    print all args generated from the argparse
    :param ckpt_dir: the save dir
    :param args: the args
    :return:
    '''
    args_dict = vars(args)
    with open(ckpt_dir + '/options.txt', 'w') as f:
        for k, v in args_dict.items():
            f.write('%s: %s\n' % (k, v))


def print_args_to_screen(args):
    '''
    print all args generated from the argparse
    :param ckpt_dir: the save dir
    :param args: the args
    :return:
    '''
    args_dict = vars(args)
    for k, v in args_dict.items():
        print('%s: %s' % (k, v))
    print('\n')


def featmap2feature(featmap):
    '''
    convert the 4D feature map to a 2D vector
    :param featmap: Batch x nChannel x Height x Wdith
    :return: Batch x num_features
    '''
    shape = featmap.size()
    feature = featmap.view(shape[0], shape[1] * shape[2] * shape[3])
    return feature

def gradient_penalty(xin, yout, mask=None):
    gradients = torch.autograd.grad(yout, xin, create_graph=True,
                                    grad_outputs=torch.ones(yout.size()).cuda(), retain_graph=True, only_inputs=True)[0]
    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1.) ** 2).mean()
    return gp