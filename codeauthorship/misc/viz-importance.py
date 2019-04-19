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

from codeauthorship.scripts.train_baseline import run_train
from codeauthorship.scripts.train_baseline import get_dataset, get_reserved_words
from codeauthorship.scripts.train_baseline import get_argument_parser, parse_args


reserved_words = get_reserved_words()

tFONTSIZE = 6


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


def get_color(tt, separate_NAME=False, reserved=False):
    if tt == 'NAME':
        if separate_NAME:
            if reserved:
                color = 'tab:blue'
            else:
                color = 'tab:orange'
        else:
            color = 'tab:blue'
    elif tt == 'ENCODING':
        color = 'yellow'
    elif tt == 'OP':
        color = 'green'
    elif tt in ('INDENT', 'DEDENT', 'COMMENT', 'NEWLINE', 'NL'):
        color = 'gray'
    elif tt in ('NUMBER', 'STRING'):
        color = 'purple'
    elif tt in ('ENDMARKER'):
        color = 'red'
    else:
        raise Exception('type={}'.format(tt))
    return color


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

    word2idx = vectorizer.vocabulary_
    idx2word = {v: k for k, v in word2idx.items()}

    print(dataset['primary'][0])
    print(dataset['secondary']['seq_types'][0])

    # Shuffle.
    index = np.arange(Y.shape[0])
    random.shuffle(index)
    X = X[index]
    Y = Y[index]

    # Filter to classes with at least 9 instances (and balance labels).

    ## First record 9 instances from each class (ignore classes with less than 9 instances).
    index = np.arange(Y.shape[0])
    index_to_keep = []
    label_set = set(labels)
    for label in label_set:
        mask = Y == label
        if mask.sum() < options.cutoff:
            continue
        # TODO: Should we take all of the instances?
        index_to_keep += index[mask].tolist()[:options.cutoff]
    index_to_keep = np.array(index_to_keep)

    ## Then filter accordingly.
    X = X[index_to_keep]
    Y = Y[index_to_keep]

    # Get importance.
    cross_validation_splitter = StratifiedKFold(n_splits=9)

    for i, (train_index, test_index) in enumerate(cross_validation_splitter.split(X, Y)):
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = Y[train_index], Y[test_index]
        train_results = run_train(trainX, trainY)
        model = train_results['model']

        break

    fi_lst = model.feature_importances_.tolist()
    feature_importance = {}
    for i in range(model.feature_importances_.shape[0]):
        feature_importance[idx2word[i]] = fi_lst[i]
    maximportance = max(list(feature_importance.values()))

    # Viz.

    seen = 0
    seen_limit = 100

    index = np.arange(len(dataset['primary']))
    random.shuffle(index)

    for idx in index.tolist()[:1000]:

        seq = dataset['primary'][idx]
        token_types = dataset['secondary']['seq_types'][idx]
        rand_tt = random.choice(['NAME', 'ENCODING', 'OP', 'INDENT', 'NUMBER', 'ENDMARKER'])

        # Format code.
        content_types = []
        contents = []
        buf = []
        buf_types = []
        for w, tt in zip(seq, token_types):
            if w == '\n':
                if len(buf) > 0:
                    contents.append(buf)
                    content_types.append(buf_types)
                    buf = []
                    buf_types = []
                continue
            buf.append(w)
            buf_types.append(tt)
        if len(buf) > 0:
            contents.append(buf)
            content_types.append(buf_types)
        buf = None
        buf_types = None

        if options.maxchar is not None:
            max_characters = max([len(''.join(line)) for line in contents])
            if max_characters > options.maxchar:
                continue

        # Draw.
        fig = plt.figure()
        rescale_width, rescale_height = init_rescaling_funcs(fig)
        r = fig.canvas.get_renderer()
        ax = fig.get_axes()[0]

        char = plt.text(0, 0, 'X', alpha=0, fontsize=tFONTSIZE)
        char_width = rescale_width(char.get_window_extent(renderer=r).width) + 0.01
        char_height = rescale_height(char.get_window_extent(renderer=r).width) + 0.03

        init_xoffset = 0.
        init_yoffset = 1.-char_height

        # Draw text and rects.
        for i, line in enumerate(contents):
            line_types = content_types[i]

            if i == 0:
                xoffset = init_xoffset
                yoffset = init_yoffset
            else:
                xoffset = init_xoffset

            for w, tt in zip(line, line_types):
                importance = feature_importance.get(w, 0)
                ibuf = maximportance / 10
                alpha = (importance+ibuf) / (maximportance+ibuf)
                color = get_color(tt, separate_NAME=options.separate_NAME, reserved=w in reserved_words)

                if options.random_type:
                    alpha = 1.
                    if tt != rand_tt:
                        alpha = 0

                w = w.replace('\t', '  ')
                for c in w:

                    # 1. Draw box.
                    p = patches.Rectangle((xoffset, yoffset),
                        char_width, char_height,
                        fill=True,
                        color=color,
                        alpha=alpha,
                        clip_on=False)
                    ax.add_patch(p)

                    # 2. Draw character.
                    t = plt.text(xoffset + char_width/2, yoffset + char_height/2, c,
                                 horizontalalignment='center', verticalalignment='center',
                                 fontsize=tFONTSIZE,
                                 bbox=dict(pad=0, fill=False, linewidth=0))
                    xoffset += char_width
            
            yoffset -= char_height

        plt.axis('off')
        plt.savefig('{}/code-{:06}.png'.format(options.imgdir, idx), bbox_inches='tight')
        plt.close()
        seen += 1

        if seen == seen_limit:
            break



if __name__ == '__main__':

    parser = get_argument_parser()
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--maxchar', default=None, type=int)
    parser.add_argument('--separate_NAME', action='store_true')
    parser.add_argument('--random_type', action='store_true')
    parser.add_argument('--imgdir', default='imgs', type=str)
    options = parse_args(parser)

    if options.separate_NAME:
        options.imgdir = 'imgs-separate'
    if options.random_type:
        options.imgdir = 'imgs-randtype'

    print(json.dumps(options.__dict__, sort_keys=True))

    run(options)
