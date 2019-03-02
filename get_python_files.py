from io import BytesIO
import io
from tokenize import tokenize
import codeauth_lib

def main():

  rows = []
  for year in range(2008, 2017):
    reader = codeauth_lib.file_getter(year)
    for row in reader:
      num_chars = str(len(str(row["flines"])))
      file_type = row["full_path"].split(".")[-1]
      print(file_type)
      file_text = row["flines"]
      if file_type == "py":
        g = tokenize(BytesIO(file_text.encode('utf-8')).readline)
      row_list = (row["year"], row["round"], row["username"], row["solution"],
          num_chars)
      rows.append(row_list)

  for row in rows:
    print("\t".join(row))

if __name__ == "__main__":
  main()
