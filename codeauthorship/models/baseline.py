import collections
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

def get_dataset_from_file(file_name):
  files, labels = [], []
  dataset = collections.defaultdict(list)
  with open(file_name, 'r') as f:
    for i, line in enumerate(f):
      username, tokens = line.strip().split("\t", 1)
      files.append(tokens), labels.append(username)
  label_vocab = {label:i for i, label in enumerate(sorted(set(labels)))}
  converted_labels = [label_vocab[label] for label in labels]
  return files, np.array(converted_labels), label_vocab


def main():
  input_file = sys.argv[1]

  files, labels, label_vocab = get_dataset_from_file(input_file)
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(files)
  skf = StratifiedKFold(n_splits=9)
  #TODO(neha): add a flag or something for n_splits
  for train_index, test_index in skf.split(X, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
      random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)


if __name__ == "__main__":
  main()
