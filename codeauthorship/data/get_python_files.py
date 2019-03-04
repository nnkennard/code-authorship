from io import BytesIO
import io
import os
import tokenize
import codeauthorship.codeauth_lib as codeauth_lib
import argparse


def main(options):

  # Assign files according to command line args.
  filenames = []
  if options.yearstart <= 0:
    filenames.append(options.data_path)
  else:
    if options.yearend <= 0:
      filenames.append(os.path.join(options.data_path, 'gcj{}.csv'.format(options.yearstart)))
    else:
      for year in range(options.yearstart, options.yearend):
        filenames.append(os.path.join(options.data_path, 'gcj{}.csv'.format(year)))

  # Read files.
  rows = []
  for filename in filenames:
    reader = codeauth_lib.file_getter(filename)
    for i, row in enumerate(reader):
      num_chars = str(len(str(row["flines"])))
      file_type = row["full_path"].split(".")[-1]
      file_text = row["flines"]
      if file_type == "py":
        if file_text.startswith("#"):
          # These are really hard to tokenize
          continue
        try:
          g = tokenize.tokenize(BytesIO(file_text.encode('utf-8')).readline)
          tokens = [tok for _, tok, _, _, _ in g]
          print(row["username"] + "\t" + " ".join(tokens))
        except:
          continue


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', default='/iesl/canvas/nnayak/data/codeauth_data/gcj-dataset', type=str)
  parser.add_argument('--yearstart', default=2008, type=int)
  parser.add_argument('--yearend', default=2017, type=int)
  options = parser.parse_args()

  main(options)
