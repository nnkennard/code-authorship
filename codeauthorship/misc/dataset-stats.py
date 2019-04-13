import argparse
import os
import json

from collections import OrderedDict, Counter


def run(options):
    metrics = OrderedDict()
    metrics['count'] = 0
    metrics['max-length'] = 0
    metrics['vocab-size'] = 0

    vocab = Counter()

    with open(options.path_in) as f:
        for line in f:
            data = json.loads(line)
            tokens = data['tokens']
            metrics['count'] += 1
            metrics['max-length'] = max(metrics['max-length'], len(tokens))
            vocab.update(tokens)

    metrics['vocab-size'] = len(vocab)

    print('metrics:')
    for k, v in metrics.items():
        print(k, v)
    print()

    freq = vocab.most_common()

    print('hi-freq-tokens:')
    for i, x in enumerate(freq[:10]):
        print(i, x)
    print()


    print('lo-freq-tokens:')
    for i, x in enumerate(freq[-10:]):
        print(i, x)
    print()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj2008.csv.jsonl', type=str)
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)

    run(options)
