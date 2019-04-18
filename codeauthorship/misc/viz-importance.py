import argparse
import os
import json
import random
import sys

# for obfuscation
import keyword
import builtins

from collections import Counter

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms

from codeauthorship.scripts.train_baseline import get_dataset, get_reserved_words
from codeauthorship.scripts.train_baseline import get_argument_parser, parse_args


def init_rescaling_funcs(fig):
    r = fig.canvas.get_renderer()

    # Draw a blank. Initializes ax.
    plt.text(0, 0, ' ')
    ax = fig.get_axes()[0]

    # Steps used to normalize future measurements.
    ## 1. Draw an invisible rectangle.
    size = 0.1
    p = patches.Rectangle((0, 0), size, size, fill=False, linewidth=0)
    ax.add_patch(p)
    ## 2. Get scaling constant.
    p_width = p.get_window_extent(renderer=r).width
    fig_width = fig.get_window_extent(renderer=r).width
    rescale_width_constant = size/p_width*fig_width
    p_height = p.get_window_extent(renderer=r).height
    fig_height = fig.get_window_extent(renderer=r).height
    rescale_height_constant = size/p_height*fig_height
    ## 3. Create scaling function.
    def rescale_width(x):
        return x/fig_width*rescale_width_constant
    def rescale_height(x):
        return x/fig_height*rescale_height_constant
    return rescale_width, rescale_height


def run(options):
    random.seed(options.seed)
    np.random.seed(options.seed)
    dataset = get_dataset(options.path_in, options)

    print('dataset-size = {}'.format(dataset['metadata']['dataset_size']))
    print('vocab-size = {}'.format(dataset['metadata']['vocab_size']))
    print('# of classes = {}'.format(dataset['metadata']['n_classes']))

    label2idx = dataset['metadata']['label2idx']
    idx2label = {v: k for k, v in label2idx.items()}
    labels = dataset['secondary']['labels']

    contents = [' '.join(x) for x in dataset['primary']]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contents).toarray()
    Y = np.array(labels)

    Xargsort = np.argsort(X, axis=0)
    Xsort = np.take_along_axis(X, Xargsort, axis=0)

    reserved_words = get_reserved_words()
    word2idx = vectorizer.vocabulary_
    idx2word = {v: k for k, v in word2idx.items()}

    print(dataset['primary'][0])


    # Viz.
    contents = [
        ['def', ' ', 'square', '(', 'x', ')', '{'],
        ['\t', 'y', ' ', '=', ' ', 'x', '*', 'x'],
        ['\t', 'return', ' ', 'y'],
        ['}'],
    ]

    # fig = plt.figure()
    # rescale_width, rescale_height = init_rescaling_funcs(fig)
    # r = fig.canvas.get_renderer()
    # ax = fig.get_axes()[0]

    # char = plt.text(0, 0, 'X', alpha=0)
    # char_width = rescale_width(char.get_window_extent(renderer=r).width) + 0.01
    # char_height = rescale_height(char.get_window_extent(renderer=r).width) + 0.03

    # init_xoffset = 0.1
    # init_yoffset = 0.9

    # for i, line in enumerate(contents):
    #     if i == 0:
    #         xoffset = init_xoffset
    #         yoffset = init_yoffset
    #     else:
    #         xoffset = init_xoffset

    #     for w in line:
    #         w = w.replace('\t', '  ')
    #         # if w == '\t':
    #         #     w = ' ' * 4
    #         for c in w:
    #             # 1. Draw box.
    #             p = patches.Rectangle((xoffset, yoffset), char_width, char_height, fill=True, color='tab:blue')
    #             ax.add_patch(p)

    #             # 2. Draw character.
    #             t = plt.text(xoffset + char_width/2, yoffset + char_height/2, c,
    #                          horizontalalignment='center', verticalalignment='center',
    #                          bbox=dict(pad=0, fill=False, linewidth=0))
    #             xoffset += char_width
        
    #     yoffset -= char_height

    # plt.axis('off')
    # plt.savefig('imgs/code.png', bbox_inches='tight')

    seen = 0

    for example_id, seq in enumerate(dataset['primary'][:1000]):
        # Format code.
        contents = []
        buf = []
        for w in seq:
            if w == '\n':
                if len(buf) > 0:
                    contents.append(buf)
                    buf = []
                continue
            buf.append(w)
        if len(buf) > 0:
            contents.append(buf)
        buf = None

        if len(contents) > 10:
            continue

        # Draw.
        fig = plt.figure()
        rescale_width, rescale_height = init_rescaling_funcs(fig)
        r = fig.canvas.get_renderer()
        ax = fig.get_axes()[0]

        char = plt.text(0, 0, 'X', alpha=0)
        char_width = rescale_width(char.get_window_extent(renderer=r).width) + 0.01
        char_height = rescale_height(char.get_window_extent(renderer=r).width) + 0.03

        init_xoffset = 0.
        init_yoffset = 1.-char_height

        # Turn off clipping
        # ax.set_clip_on(False) 
        # artists = [] 
        # artists.extend(ax.collections) 
        # artists.extend(ax.patches) 
        # artists.extend(ax.lines) 
        # artists.extend(ax.texts) 
        # artists.extend(ax.artists) 
        # for a in artists: 
        #     a.set_clip_on(False)

        # Draw text and rects.
        for i, line in enumerate(contents):
            if i == 0:
                xoffset = init_xoffset
                yoffset = init_yoffset
            else:
                xoffset = init_xoffset

            for w in line:
                w = w.replace('\t', '  ')
                # if w == '\t':
                #     w = ' ' * 4
                for c in w:
                    # 1. Draw box.
                    p = patches.Rectangle((xoffset, yoffset), char_width, char_height, fill=True, color='tab:blue', clip_on=False)
                    ax.add_patch(p)

                    # 2. Draw character.
                    t = plt.text(xoffset + char_width/2, yoffset + char_height/2, c,
                                 horizontalalignment='center', verticalalignment='center',
                                 bbox=dict(pad=0, fill=False, linewidth=0))
                    xoffset += char_width
            
            yoffset -= char_height

        plt.axis('off')
        plt.savefig('imgs/code-{:06}.png'.format(example_id), bbox_inches='tight')
        # plt.savefig('imgs/code-{:06}.png'.format(example_id))
        seen += 1

        if seen == 10:
            break



if __name__ == '__main__':

    parser = get_argument_parser()
    parser.add_argument('--topk', default=100, type=int)
    options = parse_args(parser)

    print(json.dumps(options.__dict__, sort_keys=True))

    run(options)
