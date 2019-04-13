import argparse
import os
import json

from collections import OrderedDict, Counter


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


def run_freq_tokens(options):

    dataset = get_dataset(options.path_in)

    token_counter = Counter()

    for x, y in zip(dataset['tokens'], dataset['labels']):
        token_counter.update(x)

    for x in token_counter.most_common(100):
        print(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj.jsonl', type=str)
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)

    run_freq_tokens(options)
