'''
dataset that outputs both image and the attributes.
'''

from . import base_dataset
import scipy.misc
import imageio
import numpy as np
import torch
import pandas as pd
import os
import torchvision as tv
from util import util
from util.logger import logger
import cv2
import random
from tqdm import tqdm

log = logger()
mean = torch.Tensor((0.485, 0.456, 0.406))
stdv = torch.Tensor((0.229, 0.224, 0.225))
# mean = torch.Tensor((0.5, 0.5, 0.5))
# stdv = torch.Tensor((0.5, 0.5, 0.5))
# print('attributeDataset warning: mean and stdv are 0.5')
forward_transform = tv.transforms.Compose(
    [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])


# class Dataset(base_dataset.BaseDataset):
#     def __init__(self, image_list, transform=forward_transform, scale=(128, 128), crop_size=(160, 160),
#                  bias=(0, 15),
#                  csv_path='analysis/list_attr_celeba.csv', sep=' ', scale_attribute=True):
#         super(Dataset, self).__init__()
#         self.files = []
#         supported_format = ['jpg', 'png', 'jpeg']
#         for image_now in image_list:  # filter out files that are not image
#             format = image_now.split('.')[-1]
#             format = format.lower()
#             is_image = False
#             for sf in supported_format:
#                 if format == sf:
#                     is_image = True
#                     break
#             if is_image:
#                 self.files += [image_now]
#
#         print('* Total Images: {}'.format(len(self.files)))
#         self.transform = transform
#         self.scale = scale
#         self.bias = bias
#         self.frame = pd.read_csv(csv_path, sep=sep)
#         self.crop_size = crop_size
#         self.scale_attribute = scale_attribute
#
#     def __getitem__(self, index):
#         # print(self.files[index])
#         img = util.readRGB(self.files[index]).astype(np.float32)
#         if self.crop_size[0] > 0:
#             img = util.center_crop(img, self.crop_size, self.bias)
#         shape = img.shape
#         if self.scale[0] > 0:
#             img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
#         img = self.transform(img)
#         image_name = os.path.basename(self.files[index])
#         attribute = self.frame[self.frame['name'] == image_name]
#         attribute = attribute.values[0][1:].astype(np.float32)
#         if self.scale_attribute:
#             attribute = (attribute + 1) / 2  # scale the attribute to range form {0,1}
#         # log(img)
#         return img, attribute, self.files[index], shape
#
#     def __len__(self):
#         return len(self.files)
#
#
# class Dataset_testing(Dataset):
#     def __getitem__(self, index):
#         img = util.readRGB(self.files[index]).astype(np.float32)
#         if self.crop_size[0] > 0:
#             img = util.center_crop(img, self.crop_size, self.bias)
#         shape = img.shape
#         if self.scale[0] > 0:
#             img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
#         img = self.transform(img)
#         return img, self.files[index], shape


class Dataset_testing(base_dataset.BaseDataset):
    '''
    compared with Datset, now we can use the attributes to filter images
    '''

    def __init__(self, image_list,
                 transform=forward_transform, scale=(128, 128), crop_size=(160, 160), bias=(0, 15)):
        super(Dataset_testing, self).__init__()
        self.files = image_list
        # print(self.files[:5])
        self.transform = transform
        self.scale = scale
        self.crop_size = crop_size
        self.bias = bias

        from util import faceflip
        self.faceflip = faceflip

    def __getitem__(self, index):
        print(self.files[index])
        img = util.readRGB(self.files[index]).astype(np.float32)
        orientation = self.faceflip.get_orientation(img.astype(np.uint8))
        image_name = os.path.basename(self.files[index])
        if orientation == 'right':
            img = cv2.flip(img, 1)
        elif orientation == 'left':
            pass
        if self.crop_size[0] > 0:
            img = util.center_crop(img, self.crop_size, self.bias)
        if self.scale[0] > 0:
            img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
        img = self.transform(img)
        return img, image_name

    def __len__(self):
        return len(self.files)

class Dataset_testing_filtered(base_dataset.BaseDataset):
    '''
    compared with Datset, now we can use the attributes to filter images
    '''

    def __init__(self, base_dir, attributes=None,
                 transform=forward_transform, scale=(128, 128), crop_size=(160, 160), bias=(0, 15),
                 csv_path='info/celeba-with-orientation.csv', n_samples=10):
        super(Dataset_testing_filtered, self).__init__()
        frame = pd.read_csv(csv_path)
        frame = frame[frame['orientation'] != 'unknown']
        self.orientation = pd.concat((frame['name'], frame['orientation']), axis=1)
        self.orientation = self.orientation.set_index('name').to_dict()['orientation']
        if attributes is not None:
            attributes = attributes.split(',')
            for attr in attributes:
                if attr[:3] == 'NOT':
                    frame = frame[frame[attr[3:]] <= 0]
                else:
                    frame = frame[frame[attr] == 1]
        # filter out the image list by frame
        filtered_names = frame['name'].tolist()
        # image_list = [tmp for tmp in image_list if os.path.basename(tmp) in filtered_names]
        # image_list = sorted(image_list)
        self.files = [os.path.join(base_dir, tmp) for tmp in filtered_names]
        random.shuffle(self.files)
        self.files = self.files[:n_samples]
        # print(self.files[:5])
        self.transform = transform
        self.scale = scale
        self.crop_size = crop_size
        self.bias = bias

    def __getitem__(self, index):
        print(self.files[index])
        img = util.readRGB(self.files[index]).astype(np.float32)
        image_name = os.path.basename(self.files[index])
        orientation = self.orientation[image_name]
        if orientation == 'right':
            img = cv2.flip(img, 1)
        elif orientation == 'left':
            pass
        if self.crop_size[0] > 0:
            img = util.center_crop(img, self.crop_size, self.bias)
        if self.scale[0] > 0:
            img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)


class GrouppedAttrDataset(base_dataset.BaseDataset):
    '''
    the attribute is generated as [nparray[att1, att2], nparray[att3], nparray[att4]], where each inside list corresponds
    to one branch.

    the input also has the format like:
    Moustache@No_Beard@Goatee,Smile,Young,Bangs
    however, it does not treat the "@" as "or", but concate them.
    '''

    def __init__(self, image_list, attributes, transform=forward_transform, scale=(128, 128), crop_size=(160, 160),
                 bias=(0, 15),
                 csv_path='info/celeba-with-orientation.csv', csv_split=',', random_crop_bias=0):
        super(GrouppedAttrDataset, self).__init__()
        self.files = []
        supported_format = ['jpg', 'png', 'jpeg']
        for image_now in image_list:  # filter out files that are not image
            format = image_now.split('.')[-1]
            format = format.lower()
            is_image = False
            for sf in supported_format:
                if format == sf:
                    is_image = True
                    break
            if is_image:
                self.files += [image_now]

        print('* Total Images: {}'.format(len(self.files)))
        self.transform = transform
        self.scale = scale
        self.bias = bias
        self.frame = pd.read_csv(csv_path, sep=csv_split)
        self.orientation = pd.concat((self.frame['name'], self.frame['orientation']), axis=1)
        self.orientation = self.orientation.set_index('name').to_dict()['orientation']
        self.crop_size = crop_size
        self.random_crop_bias = random_crop_bias

        # parse the "attribtues"
        attributes = attributes.split(',')  # each ',' separates a branch.
        f2 = self.frame['name'].to_frame()
        f3 = self.frame.replace(-1, 0)

        for attrs in attributes:
            attrs_split = attrs.split('@')  # '@' separates attributes inside each branch.`
            for i, att in enumerate(attrs_split):
                if att[0] == '#':
                    col_now = ~f3[att[1:]]
                else:
                    col_now = f3[att]
                if i == 0:
                    attr_value = pd.DataFrame(col_now)
                else:
                    attr_value = pd.concat([attr_value, col_now], axis=1)
            # frame_now = pd.DataFrame(atts,attr_value)
            attr_value = list(attr_value.values.astype(np.float32))
            # log(attr_value)
            # log(attrs)
            f2[attrs] = attr_value
        self.frame = f2

    def __getitem__(self, index):
        try:
            # print(self.files[index])
            img = util.readRGB(self.files[index]).astype(np.float32)
            image_name = os.path.basename(self.files[index])
            orientation = self.orientation[image_name]
            if orientation == 'right':
                img = cv2.flip(img, 1)
            elif orientation == 'left':
                pass
            else:
                raise RuntimeError

            if self.crop_size[0] > 0:
                if self.random_crop_bias > 0:
                    img = util.random_crop(img, self.crop_size, (self.random_crop_bias, self.random_crop_bias))
                else:
                    img = util.center_crop(img, self.crop_size, self.bias)
            if self.scale[0] > 0:
                img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
            # img = img[self.crop_size[0]:shape[0] - self.crop_size[0], self.crop_size[1]:shape[1] - self.crop_size[1], :]
            img = self.transform(img)
            attribute = self.frame[self.frame['name'] == image_name]
            attribute = attribute.values[0][1:]
            attribute = tuple(attribute)
            return img, attribute
        except:
            rd_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rd_idx)

    def __len__(self):
        return len(self.files)

