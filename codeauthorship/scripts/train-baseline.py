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


class ObfuscationAlgorithm(object):
    def __init__(self, rand_vocab_size=100000):
        super(ObfuscationAlgorithm, self).__init__()

        reserved_words = []
        reserved_words += keyword.kwlist
        reserved_words += dir(builtins)

        self.reserved_words = set(reserved_words)
        self.rand_vocab = ['rand_vocab_{}'.format(i) for i in range(rand_vocab_size)]
        self.rand_vocab_size = rand_vocab_size

    def obfuscate(self, tokens):
        # Get all NAME tokens that are not in the reserved vocab.
        local_vocab = set([x['val'] for x in tokens if x['type'] == 'NAME' and x['val'] not in self.reserved_words])
        local_vocab_size = len(local_vocab)

        # Create a mapping from the local vocab to our randomized vocab.
        index = random.sample(range(self.rand_vocab_size), local_vocab_size)
        local2rand = {local: self.rand_vocab[index[i]] for i, local in enumerate(local_vocab)}

        # Map the tokens.
        def map_tokens(tokens, mapping):
            for x in tokens:
                if x['type'] == 'NAME' and x['val'] not in self.reserved_words:
                    new_x = {'val': mapping[x['val']], 'type': x['type']}
                    yield new_x
                    continue
                yield x
        new_tokens = map_tokens(tokens, mapping=local2rand)

        return new_tokens


def indexify(value2idx, lst):
    def func():
        for x in lst:
            if isinstance(x, (list, tuple)):
                yield [value2idx[xx] for xx in x]
            else:
                yield value2idx[x]
    return list(func())


def get_dataset(path):
    print('reading = {}'.format(path))

    # Local variables.
    obfuscate = ObfuscationAlgorithm()

    #

    dataset = {}

    # Primary data.
    seq = []

    # Secondary data. len(seq) == len(extra[key])
    extra = {}
    labels = []
    example_ids = []

    # Metadata. Information about the dataset.
    metadata = {}

    # Read data.
    type_counter = Counter()

    tokens_to_ignore = []

    if options.nocomment:
        tokens_to_ignore.append('COMMENT')
    if options.nostring:
        tokens_to_ignore.append('STRING')
    if options.nonewline:
        tokens_to_ignore.append('NEWLINE')
    if options.nonumber:
        tokens_to_ignore.append('NUMBER')
    if options.noindent:
        tokens_to_ignore.append('INDENT')
    if options.nodedent:
        tokens_to_ignore.append('DEDENT')
    if options.noencoding:
        tokens_to_ignore.append('ENCODING')
    if options.noendmarker:
        tokens_to_ignore.append('ENDMARKER')
    if options.noerrortoken:
        tokens_to_ignore.append('ERRORTOKEN')
    if options.nonl:
        tokens_to_ignore.append('NL')
    if options.noname:
        tokens_to_ignore.append('NAME')
    if options.noop:
        tokens_to_ignore.append('OP')

    def get_tokens(tokens):
        xs = [x for x in tokens if x['type'] not in tokens_to_ignore]
        # There should always be at least two tokens.
        while len(xs) <= 1:
            xs.append({'val': 'FILLER', 'type': 'FILLER'})
        # Optionally, obfuscate the tokens.
        if options.obfuscate_names:
            xs = obfuscate.obfuscate(xs)
        return xs

    with open(path) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            tokens = get_tokens(ex['tokens'])
            token_vals = [x['val'] for x in tokens]
            token_types = [x['type'] for x in tokens]
            type_counter.update(token_types)

            seq.append(token_vals)
            labels.append(ex['username'])
            example_ids.append(ex['example_id'])

    lengths = [len(x) for x in seq]
    print('average-length = {}'.format(np.mean(lengths)))
    for k, v in type_counter.items():
        print(k, v)

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


def run_train(X, Y):
    model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=0)
    model.fit(X, Y)
    results = {}
    results['model'] = model
    return results


def run_evaluation(model, X, Y):
    predictions = model.predict(X)
    acc = np.mean(predictions == Y)
    results = {}
    results['acc'] = acc
    return results


def run_experiment(trainX, trainY, testX, testY):
    label2freq = Counter(trainY)

    # Train and Predict
    clf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=0)
    clf.fit(trainX, trainY)
    predictions = clf.predict(testX)

    # Get F1 by frequency (Note: This doesn't help anymore since everything is same freq).
    freq2metrics = {}
    for yhat, y in zip(predictions, testY):
        true_freq = label2freq[y]
        false_freq = label2freq[yhat]

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
        # print('freq={} precision={:.3f}, recall={:.3f}, f1={:.3f}'.format(
        #     k, precision, recall, f1))

        f1_lst.append(f1)

    true_pos = sum([x['true_pos'] for x in freq2metrics.values()])
    accuracy = true_pos / len(testY)
    average_f1 = np.mean(f1_lst)

    print('average-f1={:.3f} accuracy={:.3f}'.format(average_f1, accuracy))


def run(options):
    random.seed(options.seed)
    np.random.seed(options.seed)
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
    Y = np.array(labels)

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

    # Run k-fold cross validation.

    acc_lst = []

    cross_validation_splitter = StratifiedKFold(n_splits=9)

    for i, (train_index, test_index) in enumerate(cross_validation_splitter.split(X, Y)):
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = Y[train_index], Y[test_index]
        train_results = run_train(trainX, trainY)
        model = train_results['model']
        eval_results = run_evaluation(model, testX, testY)
        acc = eval_results['acc']

        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        print('fold={} train-size={} test-size={} acc={:.3f} '.format(
            i, train_size, test_size, acc))

        acc_lst.append(acc)

    average_acc = np.mean(acc_lst)

    print('average-acc={:.3f}'.format(average_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj-small.jsonl', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cutoff', default=9, type=int)
    # tokens to ignore
    parser.add_argument('--nocomment', action='store_true')
    parser.add_argument('--nostring', action='store_true')
    parser.add_argument('--nonewline', action='store_true')
    parser.add_argument('--nonumber', action='store_true')
    parser.add_argument('--noindent', action='store_true')
    parser.add_argument('--nodedent', action='store_true')
    parser.add_argument('--noencoding', action='store_true')
    parser.add_argument('--noendmarker', action='store_true')
    parser.add_argument('--noerrortoken', action='store_true')
    parser.add_argument('--nonl', action='store_true')
    parser.add_argument('--noname', action='store_true')
    parser.add_argument('--noop', action='store_true')
    # data manipulation
    parser.add_argument('--obfuscate_names', action='store_true')
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    print(json.dumps(options.__dict__, sort_keys=True))

    run(options)
