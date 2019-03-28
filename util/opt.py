'''
set the default option.
'''
import collections
import yaml

class opt(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def merge_dict(self, d):
        # update the option attribute with dict.
        self.__dict__.update(d)
        return self

    def merge_opt(self, o):
        # update the option attribute with another class
        d = vars(o)
        self.__dict__.update(d)
        return self

    def __str__(self):
        args_dict = vars(self)
        print_str = ''
        for k, v in args_dict.items():
            print_str += '%s: %s \n' % (k, v)
        return print_str

    def load(self, path):
        with open(path, 'r') as f:
            opt_now = yaml.load(f)
            print(opt_now)
        for k, v in opt_now.items():
            setattr(self, k, v)

def merge_dict(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


if __name__ == '__main__':
    a = opt()
    a.load('checkpoints/flatSpace_v3/with_featall/options.txt')
    print(a)
    # a = opt()
    # b = opt()
    # a.a = 1
    # a.b = 1
    # b.b = 2
    # b.c = 2
    # from util import util

    # util.print_args_to_screen(a)
    # util.print_args_to_screen(b)
    # a.merge_opt(b)
    # util.print_args_to_screen(a)
