import collections
import sys

def main():
  filename, num_files_str = sys.argv[1:3]
  num_files = int(num_files_str)
  user_list = []

  with open(filename, 'r') as f:
    for line in f:
      user_list.append(line.split()[0])

  user_counts = collections.Counter(user_list)
  users_to_keep = []
  for user, count in user_counts.items():
    if count == num_files:
      users_to_keep.append(user)

  output_file = filename.replace(".tsv", "_" + num_files_str + "files.tsv")
  with open(filename, "r") as f:
    with open(output_file, 'w') as g:
      for line in f:
        if line.split()[0] in users_to_keep:
          g.write(line)

if __name__ == "__main__":
  main()
