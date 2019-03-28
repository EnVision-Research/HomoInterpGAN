'''
enhance the print function
'''
from __future__ import print_function
import sys
from colored import fg, bg, attr
import sys


class logger(object):
    def __init__(self, valid=True, file=None, if_print=True):
        super(logger, self).__init__()
        self.valid = valid
        self.file = file
        self.if_print = if_print

    def __call__(self, *args):
        if self.valid:
            if self.file is not None:
                with open(self.file, 'a') as f:
                    print(*args, file=f)
            args = tuple([fg(2)] + list(args) + [attr('reset')])
            if self.if_print:
                print(*args)


import torch
log = logger()

class layer_logger(torch.nn.Module):

    def __init__(self, prefix, log_type='size'):
        super(layer_logger, self).__init__()
        self.prefix = prefix
        self.log_type = log_type
    def forward(self, x):
        if self.log_type == 'size':
            log(self.prefix, x.size())
        return x


if __name__ == '__main__':
    print_ = logger(True, file='temp.txt')
    print_('dafa', 3, 4, 5)
    print_((3, 4, 5))
    print(5, 5)
