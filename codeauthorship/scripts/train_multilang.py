import argparse
import os
import json
import random
import sys


from codeauthorship.dataset.reading import *


class Model(object):
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg


def get_argument_parser():

    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--json_result', action='store_true')
    parser.add_argument('--name', default=None, type=str)
    # args
    parser.add_argument('--path_py', default=None, type=str)
    parser.add_argument('--path_c', default=None, type=str)
    parser.add_argument('--path_cpp', default=None, type=str)
    parser.add_argument('--preset_py', default='none', choices=('none', 'small', 'medium'))
    parser.add_argument('--preset_c', default='none', choices=('none', 'small', 'medium'))
    parser.add_argument('--preset_cpp', default='none', choices=('none', 'small', 'medium'))
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cutoff', default=9, type=int)
    
    return parser
    

def parse_args(parser):
    options = parser.parse_args()

    if options.preset_py != 'none':
        preset_py = dict(small='~/Downloads/gcj-py-small.jsonl')
        options.path_py = os.path.expanduser(preset_py[options.preset_py])

    if options.preset_c != 'none':
        preset_c = dict(small='~/Downloads/gcj-c-small.jsonl')
        options.path_c = os.path.expanduser(preset_c[options.preset_c])

    if options.preset_cpp != 'none':
        preset_cpp = dict(small='~/Downloads/gcj-cpp-small.jsonl')
        options.path_cpp = os.path.expanduser(preset_cpp[options.preset_cpp])

    # Random seed.
    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    return options


def run(options):
    dataset = DatasetReader(options).read()


if __name__ == '__main__':
    parser = get_argument_parser()
    options = parse_args(parser)
    run(options)
