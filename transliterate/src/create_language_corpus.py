import argparse
import os
import re
import pandas as pd

parser = argparse.ArgumentParser(description='Take in series of .tsvs and create a language corpus file')
parser.add_argument('--files', type=argparse.FileType('r'), nargs='+')
parser.add_argument('--output', '-x', type=str, default="", help='output file')
args = parser.parse_args()

output_file = args.output

with open(output_file, "w+") as out:
    for f in args.files:
        df = pd.read_csv(f, sep='\t')
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        for (idx, row) in df_shuffled.iterrows():
            out.write(row.sentence+"\n")
