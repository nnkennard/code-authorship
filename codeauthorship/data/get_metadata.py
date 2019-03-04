import argparse
import os
import codeauthorship.codeauth_lib as codeauth_lib

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
    for row in reader:
      num_chars = str(len(str(row["flines"])))
      row_list = (row["year"], row["round"], row["username"], row["solution"],
          row["full_path"].split(".")[-1], num_chars)
      rows.append(row_list)

  for row in rows:
    print("\t".join(row))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', default='/iesl/canvas/nnayak/data/codeauth_data/gcj-dataset', type=str)
  parser.add_argument('--yearstart', default=2008, type=int)
  parser.add_argument('--yearend', default=2017, type=int)
  options = parser.parse_args()

  main(options)
