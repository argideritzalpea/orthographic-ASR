#!/usr/bin/env python

import csv
import os
import sys

extensions = ('.tsv')
unique_char = set()

def process(input_folder):

    for subdir, dirs, files in os.walk(input_folder):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in extensions:
                input_csv = os.path.join(subdir, file)
                f = open(input_csv, encoding='utf-8')
                next(f)
                reader = csv.reader(f, delimiter='\t')

                with open(input_folder + "/" + "alphabet.txt", 'w', encoding='utf-8') as o:
                    for row in reader:
                        for c in row[2]:
                            if not c in unique_char:
                                unique_char.add(c)
                    o.write('\n'.join(unique_char))
                    o.write('\n# The last (non-comment) line needs to end with a newline.\n')

def main():
    process(sys.argv[1])


if __name__ == "__main__":
    # execute only if run as a script
    main()
