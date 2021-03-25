"""
Handling ANP-data // injection of sentiment according to SentiCap: https://arxiv.org/pdf/1510.01431.pdf

The MIT License (MIT)
Originally created at 10/19/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab

Note:
    Given the lack of time to add comments: PLEASE SEE directly notebook "sentimentalize_utterances_with_anps"
for use-case.
"""

import nltk
import numpy.random as random
from collections import defaultdict

def read_senticap_anps(senticap_anp_file):
    """
    :param senticap_anp_file:
    :return: twp lists, first has positive ANPs [beautiful dog, nice person] the second negative.
    """
    positive_anps = []
    negative_anps = []
    current_sentiment = 'positive' # the file lists first the postives, then all the negatives
    with open(senticap_anp_file) as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue

            if "Negative ANPs:" in line:
                current_sentiment = 'negative'
                continue

            anp = line.rstrip()

            if len(anp) == 0:
                continue

            if current_sentiment == 'negative':
                negative_anps.append(anp)
            else:
                positive_anps.append(anp)
    return positive_anps, negative_anps


def build_senticap_noun_to_ajectives(pos_anps, neg_anps):
    res = dict()
    for tag, anps in zip(['positive', 'negative'], [pos_anps, neg_anps]):
        res[tag] = defaultdict(list)
        for anp in anps:
            adjective, noun = anp.split()
            res[tag][noun].append(adjective)
    return res


def nouns_and_adjectives_of_senticap(pos_sent_anp, neg_sent_anp):
    all_nouns = set()
    all_adjectives = set()
    for catalogue in [pos_sent_anp, neg_sent_anp]:
        for item in catalogue:
            adjective, noun = item.split()
            all_nouns.add(noun)
            all_adjectives.add(adjective)
    return all_nouns, all_adjectives


def add_anp_to_sentence(sentence_tokenized, noun_to_adj, rule='random_adjective'):
    """ Pick a noun of the sentence at that is a key of the noun_to_adj dictionary at random. Given the rule
    pick the corresponding adjective from the noun_to_adj and add it before the noun. Return the new sentence.
    If such a noun does not exist, apply no changes and return None.
    :param sentence_tokenized: ['a', 'running' 'dog']
    :param noun_to_adj: e.g., dog -> {happy, sad}, cat -> {funny, happy} etc.
    :param rule: if "most_frequent_adjective" the noun_to_adj also includes frequencies:
        e.g., dog -> {(happy 5), (sad, 1)}
    :return:
    """
    sentence_tokenized = sentence_tokenized.copy()
    pos = nltk.pos_tag(sentence_tokenized)
    noun_pos = [i for i, x in enumerate(pos) if x[1][0] == 'N'] # all noun locationns

    valid_noun_pos = []
    # Drop nouns that do not have adjective ANP.
    for p in noun_pos:
        if sentence_tokenized[p] in noun_to_adj:
            valid_noun_pos.append(p)

    if len(valid_noun_pos) == 0:
        return None


    valid_noun_pos = sorted(valid_noun_pos)    # sort for reproducibility
    random.shuffle(valid_noun_pos)
    picked_noun_pos = valid_noun_pos[0]        # pick a noun at random
    picked_noun = sentence_tokenized[picked_noun_pos]

    if rule == 'random_adjective':
        valid_adjectives = sorted(noun_to_adj[picked_noun]) # sort for reproducibility
        random.shuffle(valid_adjectives)
        picked_adjective = valid_adjectives[0]

    elif rule == 'most_frequent_adjective':
        most_freq_adjective_with_freq = sorted(noun_to_adj[picked_noun],  key=lambda x: x[1])[-1]
        picked_adjective = most_freq_adjective_with_freq[0]

    ## Avoid adding an existing adjective (e.g., happy happy man)
    if picked_noun_pos > 0 and sentence_tokenized[picked_noun_pos-1] == picked_adjective:
        pass
    else:
        sentence_tokenized.insert(picked_noun_pos, picked_adjective)

    return ' '.join(sentence_tokenized)