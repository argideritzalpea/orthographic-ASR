import lingpy
import lingpy as lp
from lingpy import *
from lingpy.settings import rc

from collections import Counter

rc(merge_vowels=False)

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

examples = [('ad am yeṭṭef rebbi afus', "ad m yeṭṭfen dacu s"),
            ('hedreɣ yakan d unelmad a', "arruy aka d n lmida"),
            ('agerwaw', 'a ger wa uḥ'),
            ('ayen tuḥwaǧeḍ', 'ay e d ḥwaǧeɣ a'),
            ('and ya d taqbaylit rbant kaṭa', 'amedyataqbaylitkab'),
            ('lemqeggel', 'l nṭeg g yil'),
            ('akka i d tennamt', "akka i ten id"),
            ('iburaḥlaten', 'eg ruḥ d aten'),
            ('ur ččiɣ seksu ur seqqaɣ s lexliɛ', 'ur ččiɣ seksu ur seqqaɣ s lexliɛ'),
            ('neẓra abeddel d awezɣi maca d adday kan fiɣef ad yesseḥibiber umdan', 'neẓra abeddel d awezɣi maca d adday kan fiɣef ad yesseḥibiber umdan')
            ]
examplesTif = [("ⵗⵔⴱⵣ ⵏⵗ ⵉⵣⴳⴰ ⴷ ⴷⴳ ⵜⵓⵔⵔⵜ", "ⵗⵔⴱⵣ ⵏⴹ ⵉⵣⴳⴰ ⴷ ⴷⴳ ⵜⵓⵔⵛⵜ")]


## ConfusionMatrix
gold_aligned = []
pred_aligned = []
confusion_matrix = ConfusionDictionary()
possibilities = set(x for x in confusion_matrix.getPreds())
for x in confusion_matrix.getGolds():
    possibilities.add(x)


## Print to files
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
            print(i)


exa = ['kr nḍr ttlḍn dr', "fukken lehduṛ ifeṣṣel uqenduṛ"]
exa1 = ("d d iḍarren ččuren", "xeḍbeɣt m iḍaṛṛen yeččuren")
exa2 = ("d dr tm drn črn", "xḍbɣt m ḍṛn yčrn", "xeḍbeɣt m iḍaṛṛen yeččuren", "d dder tt mmi tarren il ččuren")
exa3 = ("d dr tm drn črn", "xḍbɣt m ḍṛn yčrn")


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

tifpred_ligned = [x[0] for x in addBackSpaces(ligned, exa2)]
tifgold_ligned = [x[1] for x in addBackSpaces(ligned, exa2)]
tifpred_orig = exa2[0]
tifgold_orig = exa2[1]

i = 0
for x in no_lm_store:
    print(no_lm_store[x])
    i += 1
    if i > 10:
        break

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
    if row[column][0] in prealign_yaml["tokens"]:
        return str(lookup[row[column][0]]).strip("[]")

def is_Correct(row):
    if row['emission'] == row['cur_char']:
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
