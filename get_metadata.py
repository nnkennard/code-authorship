import argparse
import os
from collections import Counter

from tqdm import tqdm

import codeauth_lib


default_path = '/iesl/canvas/nnayak/codeauth_data/gcj-dataset'
default_files = [os.path.join(default_path, 'gcj{}.csv'.format(year)) for year in range(2008, 2019)]


class Example(object):
  def __init__(self, author, code, language=None):
    self.author = author
    self.code = code
    self.language = language

  @staticmethod
  def from_row(row):
    return Example(author=row['username'], code=row['flines'],
      language=os.path.splitext(row['file'])[1])


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--files', default=','.join(default_files), type=str)
  parser.add_argument('--quiet', action='store_true')
  parser.add_argument('--limit', default=9, type=int)
  options = parser.parse_args()

  options.files = [os.path.expanduser(fn) for fn in options.files.split(',')]

  examples = []
  for filename in options.files:
    reader = codeauth_lib.file_getter(filename)
    for row in tqdm(reader, disable=options.quiet, desc=filename):
      examples.append(Example.from_row(row))

  author_counter = Counter()
  language_counter = Counter()

  for ex in tqdm(examples, disable=options.quiet):
    if ex.language.lower().strip() in ('.cpp',):
      author_counter[ex.author.lower()] += 1
      language_counter[ex.language] += 1

      # print('PREVIEW')
      # print(ex.author)
      # print(ex.language)
      # print(ex.code)
      # print()

  authors = list(author_counter.keys())

  for a in authors:
    if author_counter[a] < options.limit:
      del author_counter[a]

  print('# of authors', len(author_counter))
  print('# of files', sum(author_counter.values()))
  print(language_counter)

if __name__ == "__main__":
  main()
