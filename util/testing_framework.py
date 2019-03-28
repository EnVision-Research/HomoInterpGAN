'''
this will generate a universarial framework for model testing.
'''
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.testData import untransform
from .logger import logger
from tqdm import tqdm

log = logger(True)


class TestingFramework(object):
    def __init__(self, dataset, forward_fn, post_transform=untransform, save_fn=None):
        super(TestingFramework, self).__init__()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        self.dataloader_it = iter(self.dataloader)
        self.forward_fn = forward_fn
        self.post_transform = post_transform
        self.save_fn = save_fn

    def run_save(self):
        log('TestingFramework.next')
        for data in self.dataloader:
            result = self.forward_fn(data)
            result = self.post_transform(result.data[0].cpu())
            if self.save_fn is not None:
                self.save_fn(result, data)
            else:
                raise NotImplemented

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        try:
            data = next(self.dataloader_it)
        except:
            raise StopIteration
        result = self.forward_fn(data)
        result = self.post_transform(result.data[0].cpu())
        if self.save_fn is not None:
            self.save_fn(result, data)
        return result

    def __len__(self):
        return len(self.dataset)


class TestingFrameworkParams(TestingFramework):
    '''
    the param_iteractor generates the parameters.
    '''

    def __init__(self, dataset, forward_fn, post_transform=untransform, save_fn=None, param_iteractor=None):
        super(TestingFrameworkParams, self).__init__(dataset=dataset,
                                                     forward_fn=forward_fn,
                                                     post_transform=post_transform,
                                                     save_fn=save_fn)
        self.param_iteractor = param_iteractor

    def next(self):
        try:
            data = next(self.dataloader_it)
        except:
            raise StopIteration
        result_all = []
        data_all = []
        for param in tqdm(self.param_iteractor):
            result = self.forward_fn(data, param)
            result = self.post_transform(result.data[0].cpu())
            result_all += [result]
            data_all += [data]
        if self.save_fn is not None:
            self.save_fn(result_all, data_all)
        return result_all
