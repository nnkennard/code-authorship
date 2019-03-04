import csv
import io
import sys

DATA_PATH = "/iesl/canvas/nnayak/data/codeauth_data/gcj-dataset"
csv.field_size_limit(sys.maxsize)

def file_getter(year):
  assert year in range(2008, 2019)
  filename = DATA_PATH + "/gcj" + str(year) + ".csv"

  with open(filename, 'r') as f:
    data = f.read().replace("\0", "___NULL")
    reader = csv.DictReader(data.splitlines())
    return reader
