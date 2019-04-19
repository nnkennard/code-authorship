import argparse
import os
import json
import random
import sys

from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from codeauthorship.dataset.reading import *
from codeauthorship.dataset.manager import *


def run_train(X, Y):
    model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=0)
    model.fit(X, Y)
    results = {}
    results['model'] = model
    return results


def run_evaluation(model, X, Y):
    predictions = model.predict(X)

    # Accuracy
    acc = np.mean(predictions == Y)

    results = {}
    results['acc'] = acc
    return results


def run_cv(X, Y):
    models = []
    metrics = {}
    metrics['acc'] = []

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

        # Record for later.
        metrics['acc'].append(acc)
        models.append(model)

    results = {}
    results['metrics'] = metrics
    results['models'] = models

    return results


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
    # settings
    parser.add_argument('--max_features', default=None, type=int)
    
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
    random.seed(options.seed)
    np.random.seed(options.seed)
    raw_datasets = DatasetReader(options).read()

    # TODO: Use language as a feature?
    X, Y, languages = DatasetManager(options).build(raw_datasets)

    print('language-counter={}'.format(Counter(languages)))

    results = run_cv(X, Y)

    acc_mean = np.mean(results['metrics']['acc'])
    acc_std = np.std(results['metrics']['acc'])
    acc_max = np.max(results['metrics']['acc'])

    print('acc-mean={:.3f} acc-std={:.3f} acc-max={:.3f}'.format(
        acc_mean, acc_std, acc_max))


if __name__ == '__main__':
    parser = get_argument_parser()
    options = parse_args(parser)
    run(options)
