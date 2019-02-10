import codeauth_lib

def main():

  rows = []
  for year in range(2008, 2018):
    reader = codeauth_lib.file_getter(year)
    for row in reader:
      row_list = (row["year"], row["round"], row["username"], row["solution"],
          str(len(str(row["flines"]))))
      rows.append(row_list)

  for row in rows:
    print("\t".join(row))

if __name__ == "__main__":
  main()
