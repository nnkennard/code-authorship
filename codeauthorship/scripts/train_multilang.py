"""

TODO List

- Build a larger dataset.
- Record accuracy per language.
- Get examples of files that we were not able to classify correctly.
- Are there authors we are especially bad at?
- Is there duplication between the files?
- Re-run existing experiments with larger data.
    - Feature Ablation - Keep 1 or 2.
    - Feature Ablation - Remove 1 or 2.
    - Feature Importance code images.
    - Frequency of NAME tokens with TFIDF scores.
    - Performance by number of features.
    - Performance by usage.

- New experiments.
    - Python and C

- Things to double check.
    - How many authors are we using?
    - How much code from each language are we using?
        - Venn Diagram with # of tokens
        - Venn Diagram with # of files
    - How often does duplication appear?

"""


import argparse
import os
import json
import random
import sys

from collections import Counter

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from codeauthorship.dataset.reading import *
from codeauthorship.dataset.manager import *
from codeauthorship.utils.logging import *


def get_leaf_node_count(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return is_leaves.sum()


def run_train(options, X, Y):
    logger = get_logger()

    max_leaf_nodes = None
    if options.max_leaf_nodes_scale is not None:
        max_leaf_nodes = n_classes * max_leaf_nodes_scale
    model = RandomForestClassifier(verbose=3 if options.verbose else 0,
        n_estimators=options.n_estimators,
        max_depth=options.max_depth,
        n_jobs=options.n_jobs,
        max_leaf_nodes=max_leaf_nodes,
        random_state=0,
        )
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


def run_evaluation_topk(model, X, Y, k=10):
    class2idx = {k: i for i, k in enumerate(model.classes_.tolist())}
    y_idx = [class2idx[k] for k in Y.tolist()]
    # Reshape so it is one class per row.
    y_idx = np.array(y_idx).repeat(k).reshape(-1, k)

    prob = model.predict_proba(X)
    pred = np.argsort(prob, axis=1)[:, ::-1][:, :k]

    # Compute accuracy at k.
    acck = {}
    for kk in range(1, k+1):
        pred_topk = pred[:, :kk]
        y_idx_k = y_idx[:, :kk]
        acck[kk] = np.sum(pred_topk == y_idx_k, axis=1).mean()

    results = {}
    results['acck'] = acck

    return results


def run_cv(options, X, Y):
    logger = get_logger()

    n_classes = len(set(Y))
    logger.info('n-classes={}'.format(n_classes))

    metadata = {}
    metadata['n_classes'] = n_classes
    metrics = {}
    metrics['acc'] = []
    metrics['acck'] = {}
    acck = {}

    n_splits = 9
    cross_validation_splitter = StratifiedKFold(n_splits=n_splits)
    splits = cross_validation_splitter.split(X, Y)

    for i in range(n_splits):
        logger.info('fold {}'.format(i))
        train_index, test_index = next(splits)
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = Y[train_index], Y[test_index]

        # Train
        logger.info('train')
        train_results = run_train(options, trainX, trainY)
        model = train_results['model']

        depths = [a.tree_.max_depth for a in model.estimators_]
        logger.info('depths = {}'.format(depths))

        leaf_node_counts = [get_leaf_node_count(a) for a in model.estimators_]
        logger.info('leaf-node-counts = {}'.format(depths))

        # Eval
        logger.info('eval')
        eval_results = run_evaluation(model, testX, testY)
        acc = eval_results['acc']
        logger.info('eval-acc={:.3f}'.format(acc))

        eval_results = run_evaluation_topk(model, testX, testY)
        for k, v in eval_results['acck'].items():
            logger.info('k={} eval-acc={:.3f}'.format(k, v))
            acck.setdefault(k, []).append(v)

        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        # Record for later.
        metrics['acc'].append(acc)

        del model

    for k, v in acck.items():
        metrics['acck'][k] = np.mean(v)
    results = {}
    results['metrics'] = metrics
    results['metadata'] = metadata

    return results


def get_argument_parser():

    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--json_result', action='store_true')
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--verbose', action='store_true')
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
    parser.add_argument('--exact', action='store_true')
    # data
    parser.add_argument('--max_features', default=None, type=int)
    parser.add_argument('--max_classes', default=None, type=int)
    # rfc
    parser.add_argument('--n_jobs', default=-1, type=int)
    parser.add_argument('--n_estimators', default=100, type=int)
    parser.add_argument('--max_depth', default=None, type=int)
    parser.add_argument('--max_leaf_nodes_scale', default=None, type=int)
    
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
    logger = configure_logger()

    logger.info('start')

    random.seed(options.seed)
    np.random.seed(options.seed)
    raw_datasets = DatasetReader(options).read()

    # TODO: Use language as a feature?
    X, Y, languages = DatasetManager(options).build(raw_datasets)

    logger.info('language-counter={}'.format(Counter(languages)))
    logger.info('X.shape={} Y.shape={}'.format(X.shape, Y.shape))

    results = run_cv(options, X, Y)

    acc_mean = np.mean(results['metrics']['acc'])
    acc_std = np.std(results['metrics']['acc'])
    acc_max = np.max(results['metrics']['acc'])
    logger.info('acc-mean={:.3f} acc-std={:.3f} acc-max={:.3f}'.format(
        acc_mean, acc_std, acc_max))

    for k, v in results['metrics']['acck'].items():
        logger.info('k={} acc-mean={:.3f}'.format(k, v))

    if options.json_result:
        json_result = {}
        json_result['options'] = options.__dict__
        json_result['metadata'] = results['metadata']
        json_result['metrics'] = {}
        json_result['metrics']['acc_mean'] = acc_mean
        json_result['metrics']['acc_std'] = acc_std
        json_result['metrics']['acc_max'] = acc_max
        json_result['metrics']['acc_k'] = results['metrics']['acck']
        print(json.dumps(json_result, sort_keys=True))


if __name__ == '__main__':
    parser = get_argument_parser()
    options = parse_args(parser)
    run(options)
