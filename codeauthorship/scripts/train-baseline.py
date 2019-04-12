import argparse
import os
import json

import collections
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


# def get_dataset_from_file(file_name):
#     files, labels = [], []
#     dataset = collections.defaultdict(list)
#     with open(file_name, 'r') as f:
#         for i, line in enumerate(f):
#             username, tokens = line.strip().split("\t", 1)
#             files.append(tokens), labels.append(username)
#             if i == 300:
#                 break
#     label_vocab = {label:i for i, label in enumerate(sorted(set(labels)))}
#     converted_labels = [label_vocab[label] for label in labels]
#     return files, np.array(converted_labels), label_vocab


def indexify(value2idx, lst):
    def func():
        for x in lst:
            if isinstance(x, (list, tuple)):
                yield [value2idx[xx] for xx in x]
            else:
                yield value2idx[x]
    return list(func())


def get_dataset(path):
    dataset = {}

    # Primary data.
    seq = []

    # Secondary data. len(seq) == len(extra[key])
    extra = {}
    labels = []
    example_ids = []

    # Metadata. Information about the dataset.
    metadata = {}

    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            seq.append(ex['tokens'])
            labels.append(ex['username'])
            example_ids.append(ex['example_id'])

    # Build vocab.

    ## Labels.
    label_vocab = set(labels)
    label2idx = {k: i for i, k in enumerate(sorted(label_vocab))}

    ## Tokens.
    token_vocab = set()
    for x in seq:
        token_vocab.update(x)
    token2idx = {k: i for i, k in enumerate(sorted(token_vocab))}

    # Indexify.
    labels = indexify(label2idx, labels)
    # seq = indexify(token2idx, seq)

    # Record everything.
    extra['example_ids'] = example_ids
    extra['labels'] = labels
    metadata['dataset_size'] = len(example_ids)
    metadata['label2idx'] = label2idx
    metadata['n_classes'] = len(label2idx)
    metadata['token2idx'] = token2idx
    metadata['vocab_size'] = len(token2idx)


    dataset['primary'] = seq
    dataset['secondary'] = extra
    dataset['metadata'] = metadata

    return dataset


def run(options):
    dataset = get_dataset(options.path_in)

    print('dataset-size = {}'.format(dataset['metadata']['dataset_size']))
    print('vocab-size = {}'.format(dataset['metadata']['vocab_size']))
    print('# of classes = {}'.format(dataset['metadata']['n_classes']))

    contents = [' '.join(x) for x in dataset['primary']]
    labels = dataset['secondary']['labels']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contents)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X, labels)

    predictions = clf.predict(X)
    accuracy = clf.score(X, labels)
    print(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj2008.csv.jsonl', type=str)
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)

    run(options)
