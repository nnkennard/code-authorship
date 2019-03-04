from io import BytesIO
import io
import tokenize
import codeauth_lib

def main():

  rows = []
  for year in range(2008, 2017):
    reader = codeauth_lib.file_getter(year)
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
  main()
