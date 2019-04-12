"""
header:
,year,round,username,task,solution,file,full_path,flines
"""

import argparse
import os
import json

# data_reader.py
import csv
import io
import sys
import tempfile
import tokenize

csv.field_size_limit(sys.maxsize)

def file_getter(filename):
    def func(f):
        for line in f:
            yield line.replace('\0', '')
    with open(filename, 'r') as f:
        reader = csv.DictReader(list(func(f)))
    return reader
# data_reader.py, end


def convert_file(path_in, path_out):
    reader = file_getter(path_in)

    with open(path_out, 'w') as f:
        for i, row in enumerate(reader):
            username = row['username']

            # Only python for now.
            if not row['file'].endswith('.py'):
                print('Skipping {}'.format(row['file']))
                continue

            # Write code to temporary file.
            tempf = tempfile.NamedTemporaryFile(mode='w')
            tempf.write(row['flines'])
            tempf.flush()
            tempfname = tempf.name

            # Tokenize code.
            try:
                tokenizer = tokenize.tokenize(open(tempfname, 'rb').readline)
                tokens = []
                for x in tokenizer:
                    tokens.append(x.string)

                f.write('{}\n'.format(json.dumps(dict(username=username, tokens=tokens))))
            except:
                print('Failed {}'.format(row['file']))

            # Cleanup.
            tempf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='~/Downloads/gcj2008.csv', type=str)
    parser.add_argument('--path_out', default='~/Downloads/gcj2008.csv.jsonl', type=str)
    options = parser.parse_args()

    options.path_in = os.path.expanduser(options.path_in)
    options.path_out = os.path.expanduser(options.path_out)

    convert_file(options.path_in, options.path_out)
