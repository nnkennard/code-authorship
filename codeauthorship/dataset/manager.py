import random

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from codeauthorship.utils.logging import *


class DatasetManager(object):
    def __init__(self, options):
        self.options = options

    def balance_data(self, datasets):
        """
        # TODO:
        - Select exactly 9 files for each label.
        - Optionally balance data across languages.
        """

        logger = get_logger()

        files_per_author = 9

        # 1. Accumulate all data.
        all_text_data = []
        all_labels = []
        all_languages = []

        for dset in datasets:
            all_text_data += dset['primary']
            all_labels += dset['secondary']['labels']
            all_languages += dset['secondary']['lang']

        # 2. Shuffle the data.
        N = len(all_labels)
        rindex = np.arange(N)
        random.shuffle(rindex)
        all_text_data = [all_text_data[i] for i in rindex]
        all_labels = [all_labels[i] for i in rindex]
        all_languages = [all_languages[i] for i in rindex]

        Y = np.array(all_labels)

        # 3. Record an index matching our criteria.

        ## First record 9 instances from each class (ignore classes with less than 9 instances).
        index = np.arange(N)
        index_to_keep = []
        label_lst = list(set(all_labels))

        found = 0
        for label in label_lst:
            mask = Y == label
            if mask.sum() < files_per_author:
                continue
            # TODO: Should we take all of the instances?
            index_to_keep += index[mask].tolist()[:files_per_author]
            found += 1

        ## Optionally, downsample eligible classes.
        assert len(index_to_keep) == files_per_author * found
        logger.info('found {} eligible classes'.format(found))
        if self.options.max_classes is not None:
            index_to_keep = index_to_keep[:self.options.max_classes*files_per_author]
            logger.info('downsampled to {} classes'.format(len(index_to_keep) // files_per_author))

        index_to_keep = np.array(index_to_keep)

        # 4. Filter the data accordingly.
        text_data = [all_text_data[i] for i in index_to_keep]
        labels = [all_labels[i] for i in index_to_keep]
        languages = [all_languages[i] for i in index_to_keep]

        return text_data, labels, languages

    def build(self, raw_datasets):
        logger = get_logger()

        # Configuration.
        max_features = self.options.max_features

        # Balance data.
        logger.info('balancing data')
        raw_text_data, labels, languages = self.balance_data(raw_datasets)

        logger.info('joining strings')
        contents = []
        for x in raw_text_data:
            contents.append(' '.join(x))
            del x

        logger.info('tfidf data')
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(contents)
        Y = np.array(labels)

        return X, Y, languages
