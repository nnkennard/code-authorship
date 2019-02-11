import csv
import io
import sys

csv.field_size_limit(sys.maxsize)

def file_getter(filename):
  def func(f):
    for line in f:
      yield line.replace('\0', '')
  with open(filename, 'r') as f:
    reader = csv.DictReader(list(func(f)))
  return reader
