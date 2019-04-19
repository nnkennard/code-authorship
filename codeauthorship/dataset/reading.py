import json


def indexify(value2idx, lst):
    def func():
        for x in lst:
            if isinstance(x, (list, tuple)):
                yield [value2idx[xx] for xx in x]
            else:
                yield value2idx[x]
    return list(func())


class DatasetReader(object):
    def __init__(self, options):
        super(DatasetReader, self).__init__()
        self.path_py = options.path_py
        self.path_c = options.path_c
        self.path_cpp = options.path_cpp

        self.dataset_py = PyDataset(options)
        self.dataset_c = CDataset(options)
        self.dataset_cpp = CPPDataset(options)

    def readfile(self, path):
        def func():
            with open(path) as f:
                for line in f:
                    yield json.loads(line)
        return list(func())

    def read(self):
        datasets = []

        if self.path_py is not None:
            records = self.readfile(self.path_py)
            datasets.append(self.dataset_py.build(records))
        if self.path_c is not None:
            records = self.readfile(self.path_c)
            datasets.append(self.dataset_c.build(records))
        if self.path_cpp is not None:
            records = self.readfile(self.path_cpp)
            datasets.append(self.dataset_cpp.build(records))

        dataset = ConsolidateDatasets().build(datasets)


class ConsolidateDatasets(object):
    def build(self, datasets):
        """
        - Align labels between multiple datasets.
        - Creates one large dataset.
        - Adds a new secondary option for language.
        """
        pass


class Dataset(object):
    language = None

    def __init__(self, options):
        super(Dataset, self).__init__()
        self.options = options

    def build(self, records):
        dataset = {}

        # Primary data.
        seq = []

        # Secondary data. len(seq) == len(extra[key])
        extra = {}
        seq_types = []
        labels = []
        example_ids = []

        # Metadata. Information about the dataset.
        metadata = {}
        
        for i, ex in enumerate(records):
            tokens = ex['tokens']
            seq.append([x['val'].lower() for x in tokens]) # NOTE: Case is ignored.
            seq_types.append([x['type'] for x in tokens])
            labels.append(ex['username'])
            example_ids.append(ex['example_id'])

        # Indexify if needed.
        labels, label2idx = self.build_label_vocab(labels)

        # Record everything.
        extra['example_ids'] = example_ids
        extra['labels'] = labels
        extra['seq_types'] = seq_types

        metadata['label2idx'] = label2idx
        metadata['language'] = self.language

        dataset['primary'] = seq
        dataset['secondary'] = extra
        dataset['metadata'] = metadata

        return dataset

    def build_label_vocab(self, labels):
        label_vocab = set(labels)
        label2idx = {k: i for i, k in enumerate(sorted(label_vocab))}
        labels = indexify(label2idx, labels)
        return labels, label2idx


class PyDataset(Dataset):
    language = 'python'


class CDataset(Dataset):
    language = 'c'


class CPPDataset(Dataset):
    language = 'cpp'
        