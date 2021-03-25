"""
A set of functions that are useful for processing textual data.

The MIT License (MIT)
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
from itertools import tee, islice
from symspellpy.symspellpy import SymSpell

from .language_preprocessing import unquote_words, expand_contractions
from .language_preprocessing import manual_sentence_spelling, manual_tokenized_sentence_spelling
from ..language.spelling import sentence_spelling_dictionary as artemis_sentence_spelling_dictionary
from ..language.spelling import token_spelling_dictionary as artemis_token_spelling_dictionary
from ..language.spelling import missing_from_glove_but_are_actual_words
from ..neural_models.word_embeddings import load_glove_pretrained_embedding



def ngrams(lst, n):
    """ Return the ngrams of a list of tokens.
    :param lst: the tokens
    :param n: n of n-grams
    :return:
    """
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break


def parallel_apply(iterable, func, n_processes=None):
    """ Apply func in parallel to chunks of the iterable based on multiple processes.
    :param iterable:
    :param func: simple function that does not change the state of global variables.
    :param n_processes: (int) how many processes to split the data over
    :return:
    """
    n_items = len(iterable)
    if n_processes is None:
        n_processes = min(4 * mp.cpu_count(), n_items)
    pool = Pool(n_processes)
    chunks = int(n_items / n_processes)
    res = []
    for data in pool.imap(func, iterable, chunksize=chunks):
        res.append(data)
    pool.close()
    pool.join()
    return res


def tokenize_and_spell(df, glove_file, freq_file, tokenizer, parallel=True, inplace=True, spell_check=True):
    speller = SymSpell()
    loaded = speller.load_dictionary(freq_file, term_index=0, count_index=1)
    print('SymSpell spell-checker loaded:', loaded)
    golden_vocabulary = load_glove_pretrained_embedding(glove_file, only_words=True, verbose=True)
    golden_vocabulary = golden_vocabulary.union(missing_from_glove_but_are_actual_words)
    print('Updating Glove vocabulary with *valid* ArtEmis words that are missing from it.')
    missed_tokens = defaultdict(list)

    def automatic_token_speller(token_list, max_edit_distance=1):
        new_tokens = []
        for token in token_list:
            if token in golden_vocabulary:
                new_tokens.append(token) # no spell check
            else:
                spells = speller.lookup(token, max_edit_distance)
                if len(spells) > 0:  # found a spelled checked version
                    new_tokens.append(spells[0].term)
                else: # spell checking failed
                    context = " ".join(token_list)
                    missed_tokens[token].append(context)
                    new_tokens.append(token)
        return new_tokens

    if not spell_check:
        automatic_token_speller = None

    clean_text, tokens, spelled_tokens = pre_process_text(df.utterance,
                                                          artemis_sentence_spelling_dictionary,
                                                          artemis_token_spelling_dictionary,
                                                          tokenizer,
                                                          token_speller=automatic_token_speller,
                                                          parallel=parallel)

    if inplace:
        df['tokens'] = spelled_tokens
        df['tokens_len'] = df.tokens.apply(lambda x : len(x))
        df['utterance_spelled'] = df.tokens.apply(lambda x : ' '.join(x))
        return missed_tokens
    else:
        return missed_tokens, spelled_tokens


def pre_process_text(text, manual_sentence_speller, manual_token_speller,
                     tokenizer,  token_speller=None, parallel=True):

    clean_text = text.apply(lambda x: manual_sentence_spelling(x, manual_sentence_speller)) # sentence-to-sentence map
    clean_text = clean_text.apply(lambda x: x.lower())
    clean_text = clean_text.apply(unquote_words)

    if parallel:
        clean_text = pd.Series(parallel_apply(clean_text, expand_contractions))
    else:
        clean_text = clean_text.apply(expand_contractions)

    basic_punct = '.?!,:;/\-~*_=[–]{}$^@|%#<—>'
    punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
    clean_text = clean_text.apply(lambda x: x.translate(punct_to_space))

    if parallel:
        tokens = pd.Series(parallel_apply(clean_text, tokenizer))
    else:
        tokens = clean_text.apply(tokenizer)

    spelled_tokens = tokens.apply(lambda x: manual_tokenized_sentence_spelling(x,
                                                                               spelling_dictionary=manual_token_speller)
                                  )
    if token_speller is not None:
        spelled_tokens = spelled_tokens.apply(token_speller)

    return clean_text, tokens, spelled_tokens
