from graphtransliterator import GraphTransliterator
import argparse
import pandas as pd
import os
import re

parser = argparse.ArgumentParser(description='Interact with essentials database on mlab')
parser.add_argument('--orthography', '-o', choices=['tifinagh_ahaggar', 'tifinagh_ahaggar_lig', 'tifinagh_ircam', 'arabic', 'latin_norm'], type=str)
parser.add_argument('--input', '-i', type=str, default="", help='input tsv')
parser.add_argument('--output', '-x', type=str, default="", help='output folder')
args = parser.parse_args()

orthography = args.orthography
input_file = args.input
output_folder = args.output

transliterators_path = "transliterate/transliterators/"
paths = {"arabic": transliterators_path + "/Arabic_script.yml",
        "tifinagh_ahaggar": transliterators_path + "/tifinagh_Ahaggar.yml",
        "tifinagh_ahaggar_lig": transliterators_path + "/tifinagh_Ahaggar_lig.yml",
        "tifinagh_ircam": transliterators_path + "/tifinagh_ircam.yml"}

"""
example = "yesbeɣ-asent-tent"
gt = GraphTransliterator.from_yaml_file(paths["tifinagh_ahaggar_lig"])
gt.transliterate(example)
"""

normalization_dict = {
    "'": "",
    "γ": "ɣ",
    "ԑ": "ɛ",
    "ε": "ɛ",
    "€": "ɛ",
    "ğ": "ǧ",
    "é": "e",
    "è": "e",
    "ṉ": "n",
    "â": "a",
    "ĥ": "ḥ",
    "_": "-",
    "‑": "-",
    "σ": "ɛ",
    "ă": "a",
    "!": "",
    "ï": "i",
    "ç": "s",
    "̣": "",
    "“": "",
    "”": ""
}

def normalize(string, orthography):
    str = string.lower()
    norm_string = re.sub(r"- ", "-", str)
    if orthography != "latin_norm":
        norm_string = re.sub(r'[‟“–<>:"/\\|?*«».,!;”“…()ʷ&’]', "", norm_string)
    else:
        norm_string = re.sub(r'[‟“–<>:"/\\|?*«».,!;”“…()ʷ&’]', "", norm_string)
    norm_string = re.sub(r"ḏ", "ḍ", norm_string)
    norm_string = re.sub(r"ĉ", "č", norm_string)
    norm_string = re.sub(r"ĝ", "ǧ", norm_string)
    norm_string = re.sub(r"ǰ", "j", norm_string)
    norm_string = re.sub(r"ŵ", "w", norm_string)
    norm_string = re.sub(r"ṛ", "ṛ", norm_string)
    norm_string = re.sub(r"ḍ", "ḍ", norm_string)
    norm_string = re.sub(r"ḥ", "ḥ", norm_string)
    norm_string = re.sub(r"ẓ", "ẓ", norm_string)
    norm_string = re.sub(r"f̣", "f", norm_string)
    norm_string = re.sub(r"ṣ", "ṣ", norm_string)
    norm_string = re.sub(r"ṭ", "ṭ", norm_string)
    translated = norm_string.translate(norm_string.maketrans(normalization_dict))
    norm_string = re.sub(r"-", " ", translated)
    norm_string = re.sub(r'( ){2,}', " ", norm_string)
    return norm_string



df = pd.read_csv(input_file, sep='\t')
df_augmented = df.copy()
df_augmented["normalized"] = df["sentence"] = df.apply(lambda row : normalize(row['sentence'], orthography), axis = 1)
if orthography in ['tifinagh_ahaggar', 'tifinagh_ahaggar_lig', 'tifinagh_ircam', 'arabic']:
    gt = GraphTransliterator.from_yaml_file(paths[orthography])
    df_augmented['transliteration'] = df["sentence"] = df_augmented.apply(lambda row : gt.transliterate(row['normalized']), axis = 1)
df.to_csv(output_folder + '/' + os.path.basename(input_file),sep='\t',index=False,header=True)

input_base = os.path.splitext(os.path.basename(input_file))[0]
with open(output_folder + '/' + input_base + "_compare.txt", "w+") as f:
    for (idx, row) in df_augmented.iterrows():
        f.write(row.sentence+"\n")
        f.write("\t"+row.normalized+"\n")
        if orthography in ['tifinagh_ahaggar', 'tifinagh_ahaggar_lig', 'tifinagh_ircam', 'arabic']:
            f.write("\t"+"\t"+ row.transliteration+"\n")
