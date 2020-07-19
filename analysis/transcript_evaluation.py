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

def alignTranscripts(compare_tuple):
    """
    Initial alignment of transcripts using Lingpy's 'prog_align' algorithm.
    """
    spaced_geminates = [re.sub(r'(?P<char>\w) (?P=char)', r"\g<1> :", x) for x in compare_tuple]
    condensed = [re.sub(" ", "", x) for x in spaced_geminates]
    msa = Multiple(condensed)
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
            elif aligned[a_id][0].endswith(":"):
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

confusion = ConfusionDictionary()
aligned = alignTranscripts(examples[0])
addBackSpaces(aligned, examples[0])

for i in examples:
    aligned = alignTranscripts(i)
    aligned_with_spaces = addBackSpaces(aligned, i)
    confusion.add(aligned_with_spaces)

confusion.getPreds()
