import collections
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def get_dataset_from_file(file_name):
  files, labels = [], []
  dataset = collections.defaultdict(list)
  with open(file_name, 'r') as f:
    for i, line in enumerate(f):
      username, tokens = line.strip().split("\t", 1)
      files.append(tokens), labels.append(username)
      if i == 300:
        break
  label_vocab = {label:i for i, label in enumerate(sorted(set(labels)))}
  converted_labels = [label_vocab[label] for label in labels]
  return files, np.array(converted_labels), label_vocab


def main():
  input_file = sys.argv[1]

  files, labels, label_vocab = get_dataset_from_file(input_file)
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(files)
  clf = RandomForestClassifier(n_estimators=100, max_depth=2,
      random_state=0)
  clf.fit(X, labels)

  predictions = clf.predict(X)
  accuracy = clf.score(X, labels)
  print(accuracy)


if __name__ == "__main__":
  main()
