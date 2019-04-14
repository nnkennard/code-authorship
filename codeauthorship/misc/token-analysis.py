import argparse
import os
import json

import keyword
import builtins

from collections import OrderedDict, Counter

import numpy as np


def get_dataset(path):
    def read_data():
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                label = data['username']
                tokens = data['tokens']

                # Only look at specific token types.
                tokens = [x for x in tokens if x['type'] == 'NAME']

                ex = {}
                ex['label'] = label
                ex['tokens'] = [x['val'] for x in tokens]

                yield ex
    records = list(read_data())

    labels = [x['label'] for x in records]
    tokens = [x['tokens'] for x in records]

    dataset = {}
    dataset['labels'] = labels
    dataset['tokens'] = tokens

    return dataset


def get_reserved_words():
    reserved_words = []
    reserved_words += keyword.kwlist
    reserved_words += dir(builtins)

    return set(reserved_words)


def run_show_reserved(options):
    reserved_words = get_reserved_words()
    dataset = get_dataset(options.path_in)

    author2token_counter = {}
    token_counter = Counter()

    for x, y in zip(dataset['tokens'], dataset['labels']):
        token_counter.update(x)
        if y not in author2token_counter:
            author2token_counter[y] = Counter()
        author2token_counter[y].update(x)

    total_tokens = sum(token_counter.values())
    total_reserved = sum([token_counter[x] for x in reserved_words])

    # About 25% of the data is for reserved tokens. What are the others?
    # print('reserved={:.3f} ({}/{})'.format(
    #     total_reserved/total_tokens, total_reserved, total_tokens))

    # Look at top frequency tokens.
    # Questions:
    # - How are these tokens distributed among authors?

    def count_token_author_usage(token, author2token_counter):
        return sum([1 if counter[token] > 0 else 0 for counter in author2token_counter.values()])

    reserved = list(reserved_words)
    counts = [token_counter[x] for x in reserved]
    index = np.argsort(counts)[::-1]

    for i, idx in enumerate(index):
        val = reserved[idx]
        count = counts[idx]
        usage = count_token_author_usage(val, author2token_counter)
        print('{},{},{},{}'.format(i, val, count, usage))


def run_show_unreserved(options):
    reserved_words = get_reserved_words()
    dataset = get_dataset(options.path_in)

    author2token_counter = {}
    token_counter = Counter()

    for x, y in zip(dataset['tokens'], dataset['labels']):
        token_counter.update(x)
        if y not in author2token_counter:
            author2token_counter[y] = Counter()
        author2token_counter[y].update(x)

    total_tokens = sum(token_counter.values())
    total_reserved = sum([token_counter[x] for x in reserved_words])

    # About 25% of the data is for reserved tokens. What are the others?
    # print('reserved={:.3f} ({}/{})'.format(
    #     total_reserved/total_tokens, total_reserved, total_tokens))

    # Look at top frequency tokens.
    # Questions:
    # - How are these tokens distributed among authors?

    def count_token_author_usage(token, author2token_counter):
        return sum([1 if counter[token] > 0 else 0 for counter in author2token_counter.values()])

    seen = 0
    tosee = 100
    for x in token_counter.most_common():
        val, count = x
        if val in reserved_words:
            continue
        usage = count_token_author_usage(val, author2token_counter)
        print('{},{},{},{}'.format(seen, val, count, usage))
        seen += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj-small.jsonl', type=str)
    parser.add_argument('--show_reserved', action='store_true')
    parser.add_argument('--show_unreserved', action='store_true')
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)

    if options.show_unreserved:
        run_show_unreserved(options)
    if options.show_reserved:
        run_show_reserved(options)
