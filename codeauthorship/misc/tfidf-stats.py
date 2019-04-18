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

from codeauthorship.scripts.train_baseline import get_dataset, get_reserved_words
from codeauthorship.scripts.train_baseline import get_argument_parser, parse_args


def run_top_ranking(options):
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
    X = vectorizer.fit_transform(contents)
    Y = np.array(labels)

    reserved_words = get_reserved_words()
    word2idx = vectorizer.vocabulary_
    idx2word = {v: k for k, v in word2idx.items()}

    token_counter = Counter()

    for seq in dataset['primary']:
        token_counter.update(seq)

    most_common_tokens = token_counter.most_common()

    print('tfidf-vocab-size={}'.format(len(word2idx)))
    print(list(word2idx.keys())[:10])
    print('token-vocab-size={}'.format(len(token_counter)))

    rank = 0
    skipped = 0
    topk = options.topk
    ndocs = 10

    for x in most_common_tokens:
        token, count = x
        if token not in word2idx:
            skipped += 1
            continue
        idx = word2idx[token]
        isreserved = 1 if token in reserved_words else 0
        tfidf = [X[j, idx] for j in random.sample(range(X.shape[0]), ndocs)]
        print(','.join(map(str, [rank, token, count, isreserved] + tfidf)))

        rank += 1
        if rank >= topk:
            break

    print('skipped', skipped)


def run_top_udf(options):
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

    token_counter = Counter()

    for seq in dataset['primary']:
        token_counter.update(seq)

    most_common_tokens = token_counter.most_common()

    print('tfidf-vocab-size={}'.format(len(word2idx)))
    print(list(word2idx.keys())[:10])
    print('token-vocab-size={}'.format(len(token_counter)))

    rank = 0
    skipped = 0
    topk = options.topk
    ndocs = 10

    for x in most_common_tokens:
        token, count = x
        if token not in word2idx or token in reserved_words:
            skipped += 1
            continue
        idx = word2idx[token]
        isreserved = 1 if token in reserved_words else 0
        tfidf = Xsort[-10:, idx].reshape(-1).tolist()[::-1]
        print(','.join(map(str, [rank, token, count, isreserved] + tfidf)))

        rank += 1
        if rank >= topk:
            break

    print('skipped', skipped)


if __name__ == '__main__':

    parser = get_argument_parser()
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--top_ranking', action='store_true')
    parser.add_argument('--top_userdefined', action='store_true')
    options = parse_args(parser)

    print(json.dumps(options.__dict__, sort_keys=True))

    if options.top_ranking:
        run_top_ranking(options)

    if options.top_userdefined:
        run_top_udf(options)
