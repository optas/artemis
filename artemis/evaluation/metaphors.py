"""
Greedy-approximate counting of similes/methaphors present in a set of sentences.

The MIT License (MIT)
Originally created at 9/1/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

metaphorical_substrings = {'could be',
                           'appears to be',
                           'appear to be',
                           'reminds me',
                           'remind me',
                           'seems like',
                           'looks like',
                           'look like',
                           'is like',
                           'are like',
                           'think of',
                           'resembles',
                           'resembling'
                           }


def makes_metaphor_via_substring_matching(sentences, substrings=None):
    """
    :param sentences: list of strings
    :param substrings: iterable with substrings of which the occurrence implies a metaphor is made
    :return: list with booleans
    """
    if substrings is None:
        substrings = metaphorical_substrings

    makes_metaphor = []
    for s in sentences:
        yes = False
        for m in substrings:
            if m in s:
                yes = True
                break
        makes_metaphor.append(yes)
    return makes_metaphor