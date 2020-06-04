import argparse
import os
import re
import yaml

parser = argparse.ArgumentParser(description='Interact with essentials database on mlab')
parser.add_argument('--orthography', '-o', choices=['tifinagh_ahaggar', 'tifinagh_ahaggar_lig', 'tifinagh_ircam', 'arabic'], type=str)
parser.add_argument('--input', '-i', type=str, default="", help='input tsv')
parser.add_argument('--output', '-x', type=str, default="", help='output folder')
args = parser.parse_args()

orthography = args.orthography
input_file = args.input
output_file = args.output

print(input_file)

input_data = yaml.load(open(input_file, encoding='utf-8'), Loader=yaml.FullLoader)

unique_chars = set()
for x in input_data:
    for j in input_data["tokens"]:
        for char in j:
            unique_chars.add(char)

with open(output_file, 'w', encoding='utf-8') as o:
    o.write('\n'.join(sorted(unique_chars)))
    o.write('\n# The last (non-comment) line needs to end with a newline.\n')
