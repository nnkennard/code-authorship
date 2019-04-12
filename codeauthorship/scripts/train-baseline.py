import argparse
import os
import json
import random
import sys

from collections import Counter

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
    random.seed(11)
    dataset = get_dataset(options.path_in)

    print('dataset-size = {}'.format(dataset['metadata']['dataset_size']))
    print('vocab-size = {}'.format(dataset['metadata']['vocab_size']))
    print('# of classes = {}'.format(dataset['metadata']['n_classes']))

    label2idx = dataset['metadata']['label2idx']
    idx2label = {v: k for k, v in label2idx.items()}
    labels = dataset['secondary']['labels']

    contents = [' '.join(x) for x in dataset['primary']]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(contents)

    # Create data split.

    ## Shuffle.
    index = list(range(len(labels)))
    random.shuffle(index)
    author2idx = {}
    for idx in index:
        label = labels[idx]
        author2idx.setdefault(label, []).append(idx)

    ## Create split.
    cutoff = 4
    train_idx = []
    test_idx = []
    for k in author2idx.keys():
        if len(author2idx[k]) <= cutoff:
            continue
        test_idx += author2idx[k][:cutoff]
        train_idx += author2idx[k][cutoff:]

    # Create data.
    trainX = X[train_idx]
    trainY = [labels[idx] for idx in train_idx]
    testX = X[test_idx]
    testY = [labels[idx] for idx in test_idx]

    # Get more detail about class distribution.

    class_dist = Counter(trainY)

    print('hi-freq class info:')
    for i, x in enumerate(class_dist.most_common()[:10]):
        label = idx2label[x[0]]
        count = x[1]
        percent = count / len(trainY)

        print('{:3}. {:10} {:.3f} ({}/{})'.format(
            i, label, percent, count, len(trainY)))
    print('lo-freq class info:')
    for i, x in enumerate(class_dist.most_common()[-10:]):
        label = idx2label[x[0]]
        count = x[1]
        percent = count / len(trainY)

        print('{:3}. {:10} {:.3f} ({}/{})'.format(
            i, label, percent, count, len(trainY)))

    class_freq_dist = Counter(class_dist.values())

    print('freq-info (# of files, # of authors with # of files):')
    for k in sorted(class_freq_dist.keys()):
        print(k, class_freq_dist[k])

    # 

    clf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=0)
    clf.fit(trainX, trainY)

    freq2metrics = {}

    predictions = clf.predict(trainX)

    for yhat, y in zip(predictions, testY):
        true_freq = class_dist[y]
        false_freq = class_dist[yhat]

        for freq in [true_freq, false_freq]:
            if freq not in freq2metrics:
                freq2metrics[freq] = dict(false_pos=0, true_pos=0, false_neg=0)

        if yhat == y:
            freq2metrics[true_freq]['true_pos'] += 1
        else:
            freq2metrics[true_freq]['false_neg'] += 1
            freq2metrics[false_freq]['false_pos'] += 1

    f1_lst = []

    for k in sorted(freq2metrics.keys()):
        true_pos = freq2metrics[k]['true_pos']
        false_pos = freq2metrics[k]['false_pos']
        false_neg = freq2metrics[k]['false_neg']

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if  (true_pos + false_neg) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print('freq={} precision={:.3f}, recall={:.3f}, f1={:.3f}'.format(
            k, precision, recall, f1))

        f1_lst.append(f1)

    true_pos = sum([x['true_pos'] for x in freq2metrics.values()])
    accuracy = true_pos / len(testY)
    average_f1 = np.mean(f1_lst)

    print('average-f1={:.3f} accuracy={:.3f}'.format(average_f1, accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj2008.csv.jsonl', type=str)
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)

    run(options)
