"""
Ideas:
    - Randomly truncate sequence (optionally use information from multiple instances).
    - Weight cross-entropy by code similarity. One-shot setting?
"""

import argparse
import os
import json
import logging
import random

from collections import Counter

from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.elmo import _ElmoCharacterEncoder as CharacterEncoder
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from tqdm import tqdm


formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('code-authorship')
logger.setLevel(logging.INFO)

# Log to console.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# HACK: Weird fix that counteracts other libraries (i.e. allennlp) modifying
# the global logger.
if len(logger.parent.handlers) > 0:
    logger.parent.handlers.pop()


class Dataset(object):
    def __init__(self, path):
        super(Dataset, self).__init__()
        self.path = path

    def read(self):
        tokens = []
        example_ids = []
        labels = []

        # Read file.
        with open(self.path) as f:
            for i, line in tqdm(enumerate(f), desc='read'):
                data = json.loads(line)
                example_ids.append(i)
                tokens.append(data['tokens'])
                labels.append(data['username'])

        # Indexify labels.
        label2idx = {}
        for x in tqdm(labels, desc='indexify-label'):
            if x not in label2idx:
                label2idx[x] = len(label2idx)
        labels = [label2idx[x] for x in labels]

        # TODO: Are we supposed to filter based on number of times seen?
        #c = Counter()
        #for x in labels:
        #    c[x] += 1
        #total = sum(c.values())
        #for k, v in sorted(c.items(), key=lambda x: x[1]):
        #    logger.info('{} {} ({:.3f})'.format(k, v, v/total))

        # Result.
        extra = dict(labels=labels, example_ids=example_ids)
        metadata = dict(label2idx=label2idx)

        return {
            "tokens": tokens,
            "extra": extra,
            "metadata": metadata,
        }


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item

    def __len__(self):
        return len(self.dataset)


def make_collate_fn(extra):
    def collate_fn(batch):
        index, tokens = zip(*batch)

        batch_map = {}
        batch_map['index'] = index
        batch_map['tokens'] = tokens

        #char = char[:, :64] # Truncate input. # TODO: Randomly select slice (or subsequence).
        tokens = [x[:64] for x in tokens]

        batch_map['char'] = batch_to_ids(tokens)

        for k, v in extra.items():
            batch_map[k] = [v[idx] for idx in index]

        return batch_map
    return collate_fn


class BatchSampler(torch.utils.data.Sampler):

    def __init__(self, datasource, batch_size, include_partial=False):
        self.batch_size = batch_size
        self.datasource = datasource
        self.include_partial = include_partial

        assert include_partial == False

    def reset(self):
        dataset_size = len(self.datasource)

        order = list(range(dataset_size))
        random.shuffle(order)

        self.order = order
        self.index = 0
        self.n_batches = dataset_size // self.batch_size

    def get_next_batch(self):
        start = self.index * self.batch_size
        batch_index = self.order[start:start+self.batch_size]
        self.index += 1
        return batch_index

    def __iter__(self):
        self.reset()
        for _ in range(len(self)):
            yield self.get_next_batch()

    def __len__(self):
        return self.n_batches


class BatchIterator(object):
    def __init__(self, dataset, config):
        super(BatchIterator, self).__init__()
        self.dataset = dataset
        self.config = config

    def get_iterator(self):
        dataset = self.dataset
        batch_size = self.config['batch_size']
        dataset_size = len(dataset['tokens'])
        n_batches = dataset_size // batch_size

        logger.info('# of example = {}'.format(dataset_size))
        logger.info('batch size = {}'.format(batch_size))
        logger.info('# of batches = {}'.format(n_batches))

        sampler = BatchSampler(dataset['tokens'], batch_size, include_partial=False)

        loader = torch.utils.data.DataLoader(SimpleDataset(dataset['tokens']), batch_sampler=sampler,
                shuffle=(sampler is None), num_workers=8, collate_fn=make_collate_fn(dataset['extra']))

        def my_iterator():
            for batch_map in loader:
                yield batch_map

        return my_iterator()


class Net(nn.Module):
    def __init__(self, n_classes=2):
        super(Net, self).__init__()

        self.n_classes = n_classes
        self.token_embed_size = 128 # Output of char_encoder.
        self.classifier_hidden_dim = 32 # TODO: Should be an option to modify.

        self.classify = nn.Sequential(
            nn.Linear(self.token_embed_size, self.classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.classifier_hidden_dim, n_classes),
            )

        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'

        self.char_encoder = CharacterEncoder(options_file=options_file, weight_file=weight_file, requires_grad=True)

        transformer_config = dict(
            input_dim=self.token_embed_size,
            hidden_dim=self.token_embed_size,
            projection_dim=self.token_embed_size,
            feedforward_hidden_dim=self.token_embed_size,
            num_layers=1,
            num_attention_heads=1,
            use_positional_encoding=True,
            dropout_prob=0,
            residual_dropout_prob=0,
            attention_dropout_prob=0,
        )

        self.transformer = StackedSelfAttentionEncoder(**transformer_config)

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            param.data.normal_()

    def forward(self, char):
        model_input = self.char_encoder(char)
        mask = model_input['mask']
        token_embedding = model_input['token_embedding']
        h_seq = self.transformer(token_embedding, mask) # TODO: What is the mask for?
        h = torch.max(h_seq, dim=1)[0]
        yhat = self.classify(h)
        return yhat


def run(options):
    random.seed(options.seed)

    logger.info(json.dumps(options.__dict__, sort_keys=True))

    logger.info('Initializing dataset.')
    dataset = Dataset(options.data_path).read()
    batch_iterator = BatchIterator(dataset, dict(batch_size=128))

    logger.info('Initializing model.')
    net = Net(n_classes=len(dataset['metadata']['label2idx']))
    opt = optim.Adam(net.parameters())

    if options.cuda:
        net.cuda()

    logger.info('Training.')

    step = 0

    for batch_map in batch_iterator.get_iterator():
        # Predict.
        logger.info('[model] predict')
        #tokens = batch_map['tokens']
        char = batch_map['char']
        if options.cuda:
            logger.info('cuda')
            char = char.cuda()
        yhat = net(char)

        # Compute loss.
        logger.info('[model] loss')
        labels = torch.LongTensor(batch_map['labels'])
        if options.cuda:
            labels = labels.cuda()
        loss = nn.CrossEntropyLoss()(yhat, labels)

        # Compute acc.
        correct = torch.sum(yhat.argmax(dim=1) == labels).item()
        total = yhat.shape[0]
        #acc = correct / total

        # Gradient step.
        logger.info('[model] update')
        opt.zero_grad()
        loss.backward()
        params = [p for p in net.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        # Logging.
        logger.info('step={} loss={:.5f} correct/total={}/{}'.format(step, loss.item(), correct, total))

        step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='~/Downloads/gcj2008.csv.jsonl', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cuda', action='store_true')
    options = parser.parse_args()

    options.data_path = os.path.expanduser(options.data_path)

    if options.seed is None:
        options.seed = random.randint(0, 1e7)

    return options


if __name__ == '__main__':
    options = parse_args()
    run(options)
