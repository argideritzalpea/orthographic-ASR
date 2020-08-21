import lingpy
import lingpy as lp
from lingpy import *
from lingpy.settings import rc
from tqdm import tqdm
import pandas as pd

from collections import Counter

lingpy.data.derive.compile_model("sca_tif", "~/lingpy/lingpy/data/models/")
sca_tif = lingpy.data.model.Model("sca_tif")
rcParams['model'] = sca_tif

sca_model
rc(merge_vowels=False)
sca_model.converter['w']

sca_model = rc('sca')
sca_model.converter['j'] = 'I'
sca_model.converter['w'] = 'Y'

rc(sca=sca_tif)

import re

#geminates = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in original]
# pairwise
#out = lingpy.align.pairwise.nw_align(original[0], original[1])


class ConfusionDictionary:

    def __init__(self):
        self.gold2pred = {}
        self.pred2gold = {}

    def add(self, aligned_with_spaces):
        for x in aligned_with_spaces:
            gold, pred = x[0], x[1]

            self.gold2pred.setdefault(gold, Counter())
            self.gold2pred[gold][pred] += 1

            self.pred2gold.setdefault(pred, Counter())
            self.pred2gold[pred][gold] += 1

    def getGolds(self):
        return self.gold2pred

    def getPreds(self):
        return self.pred2gold
def alignTranscripts(compare_tuple, mergeGeminates=True, spacedGeminates=True):
    """
    Initial alignment of transcripts using Lingpy's 'prog_align' algorithm.
    """
    if spacedGeminates == True:
        compare_tuple = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in compare_tuple]
    condensed = [re.sub(" ", "", x) for x in compare_tuple]
    msa = Multiple(condensed, merge_geminates=mergeGeminates)
    msa.prog_align()
    aligned = [x.split("\t") for x in msa.__unicode__().split("\n")]
    aligned = list(zip(aligned[0],aligned[1]))
    return aligned
def addBackSpaces(aligned, compare_tuple):
    a_id = 0
    o_id = 0
    final = []
    original = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in compare_tuple]
    while o_id < len(original[0]):
        #print(aligned[a_id], original[0][o_id])
        if aligned[a_id][0] == "-":
            final.append(aligned[a_id])
            a_id += 1
            continue
        elif aligned[a_id][0] == original[0][o_id]:
            final.append(aligned[a_id])
            a_id += 1
        elif o_id < len(original[0])-1:
            if aligned[a_id][0] == original[0][o_id] + original[0][o_id+1]:
                final.append(aligned[a_id])
                a_id += 1
                o_id += 1
            elif aligned[a_id][0].endswith(":") and original[0][o_id] != " ":
                if aligned[a_id][0] == original[0][o_id] + original[0][o_id+2]:
                    final.append(aligned[a_id])
                    a_id += 1
                    o_id += 2
            else:
                #print(aligned[a_id][0], original[0][o_id], original[0][o_id+1])
                final.append((" ", "*"))
        o_id += 1

    a_id = 0
    o_id = 0
    final2 = []
    while o_id < len(original[1]):
        if a_id >= len(final):
            final2.append(("-", original[1][o_id]))
            #print("appending: ",("-", original[1][o_id]))
            o_id += 1
            continue
        #print("final:", final[a_id])
        #print("comparing: ", final[a_id][1], original[1][o_id])
        #print("a_id", a_id, "o_id", o_id)
        if final[a_id][1] == "-":
            final2.append(final[a_id])
            #print("appending: ",final[a_id])
            a_id += 1
            #print("1")
            continue
        elif final[a_id][1] == "*" and original[1][o_id] == " ":
            final2.append((" ", " "))
            #print("appending: ", (" ", " "))
            a_id += 1
            o_id += 1
            #print("2")
            continue
        elif final[a_id][1] == " " and original[1][o_id] == " ":
            final2.append((" ", " "))
            #print("appending: ", (" ", " "))
            a_id += 1
            o_id += 1
            #print("2")
            continue
        elif final[a_id][1] == "*":
            final2.append(final[a_id])
            #print("appending: ", final[a_id])
            a_id += 1
            if final[a_id-1][1] not in ["*", "-"]:
                o_id += 1
            #print("3")
            continue
        elif final[a_id][1][0] == original[1][o_id][0]:
            #print("equal, appending: ",final[a_id])
            final2.append(final[a_id])
            a_id += 1
            # Tally up for geminates
            if o_id < len(original[1])-1:
                if original[1][o_id] == original[1][o_id+1]:
                    o_id += 1
                if o_id < len(original[1])-2:
                    if original[1][o_id+2] == ":":
                        o_id += 2
        elif o_id < len(original[1])-1:
            if final[a_id][1] == original[1][o_id] + original[1][o_id+1]:
                final2.append(final[a_id])
                #print("appending: ", final[a_id])
                #print("4")
                a_id += 1
                o_id += 1
            else:
                #print("5")
                #print(final2)
                if final2[-1] != ("*", " "):
                    final2.append(("*", " "))
                    #print("appending: ", ("*", " "))
        o_id += 1

    for i in range(len(final) - a_id):
        #print(final[a_id+i])
        final2.append(final[a_id+i])

    return final2
def alignTranscriptsNoGemination(compare_tuple, mergeGeminates=False):
    """
    Initial alignment of transcripts using Lingpy's 'prog_align' algorithm.
    """
    #spaced_geminates = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in compare_tuple]
    condensed = [re.sub(" ", "", x) for x in compare_tuple]
    msa = Multiple(condensed, merge_geminates=mergeGeminates)
    msa.prog_align()
    aligned = [x.split("\t") for x in msa.__unicode__().split("\n")]
    aligned = list(zip(aligned[0],aligned[1]))
    return aligned
def addBackSpacesNoGemination(aligned, compare_tuple):
    a_id = 0
    o_id = 0
    final = []
    original = compare_tuple
    while o_id < len(original[0]):
        #print(aligned[a_id], original[0][o_id])
        if aligned[a_id][0] == "-" and original[0][o_id] != " ":
            final.append(aligned[a_id])
            a_id += 1
            continue
        elif aligned[a_id][0] == original[0][o_id]:
            final.append(aligned[a_id])
            a_id += 1
        elif o_id < len(original[0])-1:
            final.append((" ", "*"))
        o_id += 1

    a_id = 0
    o_id = 0
    final2 = []

    while o_id < len(original[1]):
        if a_id >= len(final):
            final2.append(("-", original[1][o_id]))
            #print("appending: ",("-", original[1][o_id]))
            o_id += 1
            continue
        #print("final:", final[a_id])
        #print("comparing: ", final[a_id][1], original[1][o_id])
        #print("a_id", a_id, "o_id", o_id)
        if final[a_id][1] == "-":
            final2.append(final[a_id])
            #print("appending: ",final[a_id])
            a_id += 1
            #print("1")
            continue
        elif final[a_id][1] == "*" and original[1][o_id] == " ":
            final2.append((" ", " "))
            #print("appending: ", (" ", " "))
            a_id += 1
            o_id += 1
            #print("2")
            continue
        elif final[a_id][1] == " " and original[1][o_id] == " ":
            final2.append((" ", " "))
            #print("appending: ", (" ", " "))
            a_id += 1
            o_id += 1
            #print("2")
            continue
        elif final[a_id][1] == "*":
            final2.append(final[a_id])
            #print("appending: ", final[a_id])
            a_id += 1
            if final[a_id-1][1] not in ["*", "-"]:
                o_id += 1
            #print("3")
            continue
        elif final[a_id][1][0] == original[1][o_id][0]:
            #print("equal, appending: ",final[a_id])
            final2.append(final[a_id])
            a_id += 1
        elif o_id < len(original[1])-1:
            #print("5")
            if final2[-1] != ("*", " "):
                final2.append(("*", " "))
                #print("appending: ", ("*", " "))
        o_id += 1

    for i in range(len(final) - a_id):
        #print(final[a_id+i])
        final2.append(final[a_id+i])

    return final2


## ConfusionMatrix
confusion_matrix = ConfusionDictionary()
possibilities = set(x for x in confusion_matrix.getPreds())
for x in confusion_matrix.getGolds():
    possibilities.add(x)
import json
from graphtransliterator import GraphTransliterator
gt = GraphTransliterator.from_yaml_file("/Users/mosaix/orthographic-ASR/transliterate/transliterators/latin_prealignment.yml")
tf = GraphTransliterator.from_yaml_file("/Users/mosaix/orthographic-ASR/transliterate/transliterators/tifinagh_to_latin.yml")
no_lm_store = {}


gold_aligned = []
pred_aligned = []
with open('transliterate/output/latin_norm/no_lm/inferences.json') as f:
    data = json.load(f)
with open('transliterate/output/latin_norm/no_lm/alignments.txt', "w+") as l:
    for i in data:
        try:
            wavfile = i['wav_filename'].split('/')[-1]
            compare_tuple = (gt.transliterate(i['src']), gt.transliterate(i['res']))
            no_lm_store.setdefault(wavfile, {})
            no_lm_store[wavfile]['latin_gold'] = gt.transliterate(i['src'])
            no_lm_store[wavfile]['latin_pred'] = gt.transliterate(i['res'])
            aligned = alignTranscripts(compare_tuple)
            aligned_with_spaces = addBackSpaces(aligned, compare_tuple)
            aligned_out = list(map(list, zip(*aligned_with_spaces)))
            l.write(str(i['src'] + "     :     " + i['res']))
            l.write("\n")
            l.write("gold: " + str(aligned_out[0]))
            l.write("\n")
            l.write("pred: " + str(aligned_out[1]))
            l.write("\n")
            l.write("\n")
            gold_aligned.append(aligned_out[0])
            pred_aligned.append(aligned_out[1])
            confusion_matrix.add(aligned_with_spaces)
        except:
            print("EXCEPTION")



gold_aligned = []
pred_aligned = []
with open('transliterate/output/tifinagh/no_lm/inferences.json') as f:
    data = json.load(f)
with open('transliterate/output/tifinagh/no_lm/alignments.txt', "w+") as l:
    for i in data:
        try:
            wavfile = i['wav_filename'].split('/')[-1]
            no_lm_store.setdefault(wavfile, {})
            no_lm_store[wavfile]['tif_gold'] = gt.transliterate(tf.transliterate(i['src']))
            no_lm_store[wavfile]['tif_pred'] = gt.transliterate(tf.transliterate(i['res']))
            compare_tuple = (gt.transliterate(tf.transliterate(i['src'])), gt.transliterate(tf.transliterate(i['res'])))
            aligned = alignTranscriptsNoGemination(compare_tuple)
            aligned_with_spaces = addBackSpacesNoGemination(aligned, compare_tuple)
            aligned_out = list(map(list, zip(*aligned_with_spaces)))
            l.write(str(i['src'] + "     :     " + i['res']))
            l.write("\n")
            l.write("gold: " + str(aligned_out[0]))
            l.write("\n")
            l.write("pred: " + str(aligned_out[1]))
            l.write("\n")
            l.write("\n")
            gold_aligned.append(aligned_out[0])
            pred_aligned.append(aligned_out[1])
            confusion_matrix.add(aligned_with_spaces)
        except:
            print("EXCEPTION")
            print(i)

gold_aligned = []
pred_aligned = []
with open('transliterate/output/latin2tif/no_lm/inferences.json') as f:
    data = json.load(f)
with open('transliterate/output/latin2tif/no_lm/alignments.txt', "w+") as l:
    for i in data:
        try:
            wavfile = i['wav_filename'].split('/')[-1]
            no_lm_store.setdefault(wavfile, {})
            no_lm_store[wavfile]['l2t_pred'] = gt.transliterate(tf.transliterate(i['res']))
            compare_tuple = (gt.transliterate(tf.transliterate(i['src'])), gt.transliterate(tf.transliterate(i['res'])))
            aligned = alignTranscriptsNoGemination(compare_tuple)
            aligned_with_spaces = addBackSpacesNoGemination(aligned, compare_tuple)
            aligned_out = list(map(list, zip(*aligned_with_spaces)))
            l.write(str(i['src'] + "     :     " + i['res']))
            l.write("\n")
            l.write("gold: " + str(aligned_out[0]))
            l.write("\n")
            l.write("pred: " + str(aligned_out[1]))
            l.write("\n")
            l.write("\n")
            gold_aligned.append(aligned_out[0])
            pred_aligned.append(aligned_out[1])
            confusion_matrix.add(aligned_with_spaces)
        except:
            print("EXCEPTION")
            print(i)

gold_aligned_l2t = []
pred_aligned_l2t = []
with open('transliterate/output/latin2tif/no_lm/inferences.json') as f:
    data = json.load(f)
with open('transliterate/output/latin2tif/no_lm/alignments_LatinGold.txt', "w+") as l:
    for z, i in enumerate(data):
        wavfile = i['wav_filename'].split('/')[-1]
        if 'l2t_pred' in no_lm_store[wavfile] and 'latin_gold' in no_lm_store[wavfile] and no_lm_store[wavfile]['l2t_pred'] != "":
            compare_tuple = (no_lm_store[wavfile]['latin_gold'], no_lm_store[wavfile]['l2t_pred'])
            aligned = alignTranscriptsNoGemination(compare_tuple)
            aligned_with_spaces = addBackSpacesNoGemination(aligned, compare_tuple)
            aligned_out = list(map(list, zip(*aligned_with_spaces)))
            l.write(str(no_lm_store[wavfile]['latin_gold'] + "     :     " + no_lm_store[wavfile]['l2t_pred']))
            l.write("\n")
            l.write("gold: " + str(aligned_out[0]))
            l.write("\n")
            l.write("pred: " + str(aligned_out[1]))
            l.write("\n")
            l.write("\n")
            gold_aligned_l2t.append(aligned_out[0])
            pred_aligned_l2t.append(aligned_out[1])
            confusion_matrix.add(aligned_with_spaces)

gold_aligned_tif = []
pred_aligned_tif = []
with open('transliterate/output/tifinagh/no_lm/inferences.json') as f:
    data = json.load(f)
with open('transliterate/output/tifinagh/no_lm/alignments_LatinGold.txt', "w+") as l:
    for z, i in enumerate(data):
        wavfile = i['wav_filename'].split('/')[-1]
        if 'tif_pred' in no_lm_store[wavfile] and 'latin_gold' in no_lm_store[wavfile] and no_lm_store[wavfile]['tif_pred'] != "":
            compare_tuple = (no_lm_store[wavfile]['latin_gold'], no_lm_store[wavfile]['tif_pred'])
            aligned = alignTranscriptsNoGemination(compare_tuple)
            aligned_with_spaces = addBackSpacesNoGemination(aligned, compare_tuple)
            aligned_out = list(map(list, zip(*aligned_with_spaces)))
            l.write(str(no_lm_store[wavfile]['latin_gold'] + "     :     " + no_lm_store[wavfile]['tif_pred']))
            l.write("\n")
            l.write("gold: " + str(aligned_out[0]))
            l.write("\n")
            l.write("pred: " + str(aligned_out[1]))
            l.write("\n")
            l.write("\n")
            gold_aligned_tif.append(aligned_out[0])
            pred_aligned_tif.append(aligned_out[1])
            confusion_matrix.add(aligned_with_spaces)

def pad(list):
    return (["SOS"]+list+["EOS"])

import pandas as pd
def createComparisonFrame(gold_list, pred_list):
    counts = []
    for gold, pred in zip(gold_list, pred_list):
        gold = pad(gold)
        pred = pad(pred)
        for i, (g, p) in enumerate(zip(gold, pred)):
            if i > 0 and i < len(gold)-1:
                prev_char = ""
                cur_char = g
                next_char = ""
                if gold[i+1] == g:
                    next_char = gold[i+2]
                    cur_char = g+g
                else:
                    next_char = gold[i+1]
                if gold[i-1] == g:
                    prev_char = gold[i-2]
                    cur_char = g+g
                else:
                    prev_char = gold[i-1]
                counts.append([p, cur_char, prev_char, next_char, 1])
    df = pd.DataFrame(counts, columns=["emission", "cur_char", "prev_char", "next_char", "count"])
    return df.groupby(["emission", "cur_char", "prev_char", "next_char"]).agg({'count': 'sum'}).reset_index()

tifinagh_comparison_frame = createComparisonFrame(gold_aligned_tif, pred_aligned_tif)
latin2tif_comparison_frame = createComparisonFrame(gold_aligned_l2t, pred_aligned_l2t)

import yaml
with open('/Users/mosaix/orthographic-ASR/transliterate/transliterators/latin_prealignment.yml') as f:
    # use safe_load instead load
    prealign_yaml = yaml.safe_load(f)


### Rekey the values
def label_CV_gold(row, column):
    lookup = prealign_yaml["tokens"]
    lookup["E"] = "EOS"
    lookup["ʊ"] = "vowel"
    lookup["ə"] = "vowel"
    lookup["ʁ"] = "consonant"
    lookup["ʃ"] = "consonant"
    lookup["ħ"] = "consonant"
    lookup["β"] = "consonant"
    lookup["ʒ"] = "consonant"
    lookup["χ"] = "consonant"
    lookup["ɡ"] = "consonant"
    lookup["S"] = "SOS"
    lookup["ʕ"] = "consonant"
    lookup["*"] = "no_space"
    if row[column][0] in prealign_yaml["tokens"]:
        return str(lookup[row[column][0]]).strip("[]'")
    else:
        undefined.add(row[column][0])

def is_Correct(row, emission, cur_char='cur_char', prev_char='prev_char', next_char='next_char'):
    compare_tuple = [row[emission][0], row[cur_char][0]]
    # Allow for 'j' - 'i'
    if compare_tuple[0] == 'j' and compare_tuple[1][0] == 'i':
            return True
    # Allow for 'w' - 'u'
    if compare_tuple[0] == 'w' and compare_tuple[1][0] in ['u', 'o']:
            return True
    if compare_tuple[0][0] == compare_tuple[1][0]:
        return True
    return False

def both_Match(row):
    if row['3b_emission'][0] == row['3c_emission'][0]:
        return True
    return False


def mismatch(row, pred_tf1='is_Correct_3b', pred_tf2='is_Correct_3c'):
    if len(set([row[pred_tf1], row[pred_tf2]])) > 1:
        return True
    else:
        return False

def eitherContains(row, char, pred_tf1='3b_emission', pred_tf2='3c_emission'):
    if row['3b_emission'][0] == char or row['3c_emission'][0] == char:
        return True
    else:
        return False

latin2tif_comparison_frame['emission_CV'] = latin2tif_comparison_frame.apply (lambda row: label_CV_gold(row, 'emission'), axis=1)
latin2tif_comparison_frame['cur_char_CV'] = latin2tif_comparison_frame.apply (lambda row: label_CV_gold(row, 'cur_char'), axis=1)
latin2tif_comparison_frame['prev_char_CV'] = latin2tif_comparison_frame.apply (lambda row: label_CV_gold(row, 'prev_char'), axis=1)
latin2tif_comparison_frame['next_char_CV'] = latin2tif_comparison_frame.apply (lambda row: label_CV_gold(row, 'next_char'), axis=1)
latin2tif_comparison_frame['is_Correct'] = latin2tif_comparison_frame.apply (lambda row: is_Correct(row), axis=1)
latin2tif_comparison_frame.to_csv("latin2tif_comparison_frame.csv")

tifinagh_comparison_frame['emission_CV'] = tifinagh_comparison_frame.apply (lambda row: label_CV_gold(row, 'emission'), axis=1)
tifinagh_comparison_frame['cur_char_CV'] = tifinagh_comparison_frame.apply (lambda row: label_CV_gold(row, 'cur_char'), axis=1)
tifinagh_comparison_frame['prev_char_CV'] = tifinagh_comparison_frame.apply (lambda row: label_CV_gold(row, 'prev_char'), axis=1)
tifinagh_comparison_frame['next_char_CV'] = tifinagh_comparison_frame.apply (lambda row: label_CV_gold(row, 'next_char'), axis=1)
tifinagh_comparison_frame['is_Correct'] = tifinagh_comparison_frame.apply (lambda row: is_Correct(row), axis=1)
tifinagh_comparison_frame.to_csv("tifinagh_comparison_frame.csv")



def alignTranscriptsNoGeminationMulti(compare_tuple, mergeGeminates=False):
    """
    Initial alignment of transcripts using Lingpy's 'prog_align' algorithm.
    """
    #spaced_geminates = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in compare_tuple]
    condensed = [re.sub(" ", "", x) for x in compare_tuple]
    condensed = [compare_tuple[0]] + [re.sub("j", "j/i", x) for x in compare_tuple[1:]]
    print(condensed)
    msa = Multiple(condensed, merge_geminates=mergeGeminates)
    msa.prog_align()
    aligned = [x.split("\t") for x in msa.__unicode__().split("\n")]
    aligned = list(zip(aligned[0],aligned[1],aligned[2]))
    return aligned


compare_tuple_test = (no_lm_store[wavfile]['tif_gold'], no_lm_store[wavfile]['tif_pred'], no_lm_store[wavfile]['l2t_pred'])
compare_tuple_test

no_lm_store[wavfile]
aligned_test = alignTranscriptsNoGeminationMulti((no_lm_store[wavfile]['tif_gold'], no_lm_store[wavfile]['tif_pred'], no_lm_store[wavfile]['l2t_pred']))
aligned_test


def addBackSpaces(original, aligned):

    final = []
    a_index = 0
    original_indices = [0] * len(original)

    def checkIfDone():
        for i, x in enumerate(original_indices):
            if x >= len(original[i])-1:
                yield True
            else:
                yield False

    def checkWhichSpaced():
        return [i for i, val in enumerate(original_indices) if original[i][min(val, len(original[i])-1)] == " "]

    while not all(checkIfDone()) and a_index != len(aligned):
        spaces = checkWhichSpaced()
        dont_move = [i for i,x in enumerate(original) if aligned[a_index][i] == "-"]
        if len(spaces) > 0:
            append_unit = ["*"] * len(original)
            for x in spaces:
                original_indices[x] += 1
                append_unit[x] = " "
            final.append(tuple(append_unit))
            continue
        else:
            for i, x in enumerate(original):
                if i not in dont_move:
                    original_indices[i] += 1
            append_unit = aligned[a_index]
            final.append(tuple(append_unit))
            a_index += 1

    while a_index != len(aligned):
        append_unit = aligned[a_index]
        final.append(tuple(append_unit))
        a_index += 1

    return final


def cf_TwoTifinaghs2Gold(no_lm_store):
    stripper = lambda x: x.strip()

    counts = []
    with open('transliterate/output/tri_alignments.txt', "w+") as outfile:
        for wav in tqdm(no_lm_store):
            try:
                latin_gold = no_lm_store[wav]["latin_gold"]
                tif_pred = no_lm_store[wav]["tif_pred"]
                l2t_pred = no_lm_store[wav]["l2t_pred"]
                original = [latin_gold]
                if tif_pred not in [" " or ""]:
                    original.append(tif_pred)
                else:
                    original.append("-")
                if l2t_pred not in [" " or ""]:
                    original.append(l2t_pred)
                else:
                    original.append("-")
                original = tuple(original)
                original = list(map(stripper, original))
                aligned = alignTranscriptsNoGeminationMulti(original)
                """
                try:
                    aligned = switchMatrisLectionis(aligned)
                except Exception(e):
                    print(e)
                """
                final = addBackSpaces(original, aligned)
                start_pad = tuple(["SOS"]*3)
                end_pad = tuple(["EOS"]*3)
                final.insert(0,start_pad)
                final.append(end_pad)
                finalT = list(zip(*final))
                outfile.write(wav)
                outfile.write("\n")
                outfile.write(str(original[0]) + "     :     " + str(original[1]) + "     :     " + str(original[2]))
                outfile.write("\n")
                outfile.write("gold:" + "\t" + str(finalT[0]))
                outfile.write("\n")
                outfile.write("tifn:" + "\t" + str(finalT[1]))
                outfile.write("\n")
                outfile.write("lt2_:" + "\t" + str(finalT[2]))
                outfile.write("\n")
                outfile.write("\n")
                for i, (g, t, l) in enumerate(final):
                    gold = finalT[0]
                    if i > 0 and i < len(gold)-1:
                        prev_char = ""
                        cur_char = g
                        next_char = ""
                        if gold[i+1] == g:
                            next_char = gold[i+2]
                            cur_char = g+g
                        else:
                            next_char = gold[i+1]
                        if gold[i-1] == g:
                            prev_char = gold[i-2]
                            cur_char = g+g
                        else:
                            prev_char = gold[i-1]
                        counts.append([t, l, cur_char, prev_char, next_char, 1])
            except:
                print(wav)

    df = pd.DataFrame(counts, columns=["3b_emission", "3c_emission", "cur_char", "prev_char", "next_char", "count"])
    return df.groupby(["3b_emission", "3c_emission", "cur_char", "prev_char", "next_char"]).agg({'count': 'sum'}).reset_index()

sample = sample_from_dict(no_lm_store)
out = cf_TwoTifinaghs2Gold(sample)

# Full
cf_TwoTifinaghs2Gold(no_lm_store)

aligned = [('i','-','-'),('-','j','k'),('k','-','g'),('i','j','j')]
switchedout = switchMatrisLectionis(aligned)

def printAlignment(aligned):
    for i in aligned:
        print(i)

threway_df = cf_TwoTifinaghs2Gold(no_lm_store)
threway_df.describe()

"""
wav = "common_voice_kab_17992085.wav"
latin_gold = no_lm_store[wav]["latin_gold"]
tif_pred = no_lm_store[wav]["tif_pred"]
l2t_pred = no_lm_store[wav]["l2t_pred"]
original = [latin_gold]

original = tuple(original)
original
aligned = alignTranscriptsNoGeminationMulti(original)
final = addBackSpaces(original, aligned)
"""



### Rekey the values
threway_df.columns
threway_df['3b_emission_CV'] = threway_df.apply (lambda row: label_CV_gold(row, '3b_emission'), axis=1)
threway_df['3c_emission_CV'] = threway_df.apply (lambda row: label_CV_gold(row, '3c_emission'), axis=1)
threway_df['cur_char_CV'] = threway_df.apply (lambda row: label_CV_gold(row, 'cur_char'), axis=1)
threway_df['prev_char_CV'] = threway_df.apply (lambda row: label_CV_gold(row, 'prev_char'), axis=1)
threway_df['next_char_CV'] = threway_df.apply (lambda row: label_CV_gold(row, 'next_char'), axis=1)
threway_df['is_Correct_3b'] = threway_df.apply (lambda row: is_Correct(row, '3b_emission'), axis=1)
threway_df['is_Correct_3c'] = threway_df.apply (lambda row: is_Correct(row, '3c_emission'), axis=1)
threway_df['emissions_match'] = threway_df.apply (lambda row: both_Match(row), axis=1)
threway_df['either_incorrect'] = threway_df.apply (lambda row: mismatch(row, 'is_Correct_3b','is_Correct_3c'), axis=1)
threway_df['either_j'] = threway_df.apply (lambda row: eitherContains(row, 'j', '3b_emission','3c_emission'), axis=1)
threway_df['either_w'] = threway_df.apply (lambda row: eitherContains(row, 'w', '3b_emission','3c_emission'), axis=1)

threway_df.to_csv("tif_decoding_comparison_frame.csv", index=False)

import random
def sample_from_dict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))





latin_gold = no_lm_store['common_voice_kab_17867289.wav']["latin_gold"]
tif_pred = no_lm_store['common_voice_kab_17867289.wav']["tif_pred"]
l2t_pred = no_lm_store['common_voice_kab_17867289.wav']["l2t_pred"]
original = [latin_gold]
if tif_pred not in [" " or ""]:
    original.append(tif_pred)
if l2t_pred not in [" " or ""]:
    original.append(l2t_pred)
original = tuple(original)
stripper = lambda x: x.strip()
original = list(map(stripper, original))


def alignTranscriptsNoGeminationMulti(compare_tuple, mergeGeminates=False):
    """
    Initial alignment of transcripts using Lingpy's 'prog_align' algorithm.
    """
    #spaced_geminates = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in compare_tuple]
    condensed = [re.sub(" ", "", x) for x in compare_tuple]
    """
    treated = []
    print(compare_tuple)
    for index, x in enumerate(compare_tuple):
        temp = []
        if index > 0:
            print(x)
            for k in x:
                if k[0] == "j":
                    temp.append("j/i")
                elif k[0] == "w":
                    temp.append("w/o/u")
                else:
                    temp.append(k)
            treated.append(temp)
        else:
            treated.append(list(x))
    condensed = treated
    """
    msa = Multiple(condensed, merge_geminates=mergeGeminates, model="sca_tif")
    msa.prog_align()
    aligned = [x.split("\t") for x in msa.__unicode__().split("\n")]
    aligned = list(zip(aligned[0],aligned[1],aligned[2]))
    return aligned


original
aligned = alignTranscriptsNoGeminationMulti(original)
aligned

from lingpy.settings import rcParams
rcParams
