"""
The MIT License (MIT)
Originally created at 10/5/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import numpy as np
from tqdm import tqdm

def lcs(s1, s2):
    """
    Longest common subsequence of two iterables. A subsequence is a
    sequence that appears in the same relative order, but not necessarily contiguous.
    :param s1: first iterable
    :param s2: second iterable
    :return: (list) the lcs
    """
    matrix = [[[] for _ in range(len(s2))] for _ in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = [s1[i]]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + [s1[i]]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
    cs = matrix[-1][-1]
    return cs


def captions_lcs_from_training_utterances(captions_tokenized, train_utters_tokenized):
    maximizers =  np.zeros(len(captions_tokenized), dtype=int)
    max_lcs = np.zeros(len(captions_tokenized))
    averages = np.zeros(len(captions_tokenized))
    for i, caption in enumerate(tqdm(captions_tokenized)):
        caption_res = [len(lcs(caption, tr_example)) for tr_example in train_utters_tokenized]
        max_loc = np.argmax(caption_res)
        maximizers[i] = max_loc
        max_lcs[i] = caption_res[max_loc]
        averages[i] = np.mean(caption_res)
    return max_lcs, averages, maximizers


###
# Panos Note:
# a) '[the] contours shadowing [and] details make this painting [look like a] photograph the way the hair is
# layered and [the eyes] gazing off to space are fantastic'
# b) '[the] red [and] black paint strokes [look like a] bunch on [the eyes]'
# (a), (b) have lcs = 7
# but,
# a) '[the woman] is pretty nice and [has a] welcoming [facial expression]'
# b) '[the woman] looks very elegant since she [has] such [a] beautiful [facial expression]'
# (a), (b) have lcs = 6
# implying that removing stop-word articles "a", "the" could make this more realistic, since the first pair is way more
# dissimilar than the second.
# also if you use this to compare to systems; the length of the utterance could be used to normalize the bias the length
# brings in.
###




