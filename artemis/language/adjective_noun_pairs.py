"""
Some operations to handle Adjective-Noun Pairs. E.g., useful for sentiment injection

The MIT License (MIT)
Originally created mid 2020, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

from collections import Counter
from .part_of_speech import  nltk_parallel_tagging_of_tokens

def collect_anps_of_sentence(tokenized_pos_tagged_sentence, tagset='universal'):
    """ return all ANPs that occur in consecutive positions.
    tokenized_pos_tagged_sentence: list, containing the result of calling from nltk.pos_tag on a tokenized sentence.
      E.g., [('a', 'DT'), ('big', 'JJ'), ('man', 'NN')]
    """
    n_tokens = len(tokenized_pos_tagged_sentence)
    collected = []

    if tagset == 'universal':
        for i, p in enumerate(tokenized_pos_tagged_sentence):
            if p[1] == 'ADJ' and i < n_tokens -1:
                if tokenized_pos_tagged_sentence[i+1][1] == 'NOUN':
                    collected.append(p[0] + ' ' + tokenized_pos_tagged_sentence[i+1][0])
    elif tagset == 'penn':
        for i, p in enumerate(tokenized_pos_tagged_sentence):
            if p[1].startswith('J') and i < n_tokens -1:
                if tokenized_pos_tagged_sentence[i+1][1].startswith('N'):
                    collected.append(p[0] + ' ' + tokenized_pos_tagged_sentence[i+1][0])
    else:
        raise ValueError()
    return collected


def collect_anp_statistics_of_collection(token_series):
    """ E.g., e.g., how frequent is the ANP "happy man" in the token_series.
    :param token_series: pd.Series, each row is a tokenized sentence
    :return:
    """
    part_of_s = nltk_parallel_tagging_of_tokens(token_series)
    anps = part_of_s.apply(collect_anps_of_sentence)
    anp_counter = Counter()
    anps.apply(anp_counter.update)
    return anp_counter, anps, part_of_s