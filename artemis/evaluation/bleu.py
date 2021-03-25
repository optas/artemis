"""
BLEU via NLTK

The MIT License (MIT)
Originally created at 8/31/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

cc = SmoothingFunction()

def sentence_bleu_for_hypotheses(references, hypothesis, max_grams=4, smoothing_function=None):
    """ Compute the BLEU score for the hypothesis (e.g., generated captions) against given references acting
    as ground-truth.
    :param references: (list of lists of lists) of len M. Each sublist contains strings. [['a', 'boy'], ['rock', 'music']]
    :param hypothesis: (list of lists)
    :param max_grams: int, bleu-max_grams i.e., when 4, computes bleu-4
    :param smoothing_function:
    :return: a Series containing the scores in the same order as the input
    Note: see nltk.bleu_score.sentence_bleu
    """
    if len(references) != len(hypothesis):
        raise ValueError('Each reference (set) comes with a single hypothesis')
    if type(references[0]) != list or type(hypothesis[0]) != list:
        raise ValueError('Bad input types: use tokenized strings, and lists of tokens.')

    scores = []
    weights = (1.0 / max_grams, ) * max_grams

    for i in range(len(references)):
        scores.append(sentence_bleu(references[i], hypothesis[i], weights=weights,
                                    smoothing_function=smoothing_function))
    return pd.Series(scores)