"""
Auxiliary routines to be used when analyzing/comparing ArtEmis in terms of its subjectivity, abstractness etc.
See also notebooks/analysis/concreteness_subjectivity_sentiment.ipynb

The MIT License (MIT)
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import numpy as np
from collections import defaultdict
from tqdm.notebook import tqdm as tqdm_notebook

from collections import Counter
from ..language.basics import ngrams

def contains_word(tokenized_sentences, word_set):
    boolean_mask = tokenized_sentences.apply(lambda x: len(set(x).intersection(word_set)) >= 1)
    return boolean_mask

def contains_bigrams(tokens, bigram_set):
    token_bigrams = set([' '.join(b) for b in ngrams(tokens, 2)])
    return any(x in bigram_set for x in token_bigrams)


def concreteness_of_sentence(tokens, word_to_concreteness, count_bigrams=True):
    "Sorry, will add add explanation in April..."

    bigram_vals = [] # concreteness values of found bigrams
    if count_bigrams:
        # find bigrams that occur and their multiplicity
        bigrams = Counter(ngrams(tokens, 2))
        utterance = ' '.join(tokens)
        for bigram, cnt in bigrams.items():
            bigram = ' '.join(bigram)
            if bigram in word_to_concreteness:
                for _ in range(cnt):
                    bigram_vals.append(word_to_concreteness[bigram])
                utterance = utterance.replace(bigram, '') # remove bigrams from the utterance
                                                          # to not double-count/score them
        tokens = utterance.split()

    unigram_vals = [word_to_concreteness[t] for t in tokens if t in word_to_concreteness]
    conc_vals = unigram_vals + bigram_vals

    if len(conc_vals) == 0:
        return None
    return sum(conc_vals) / len(conc_vals)


def pos_analysis(df, group_cols=None, round_decimal=1):
    # Assumes nltk universal pos-tagging
    # & df['pos'] has the part-of-speech tags
    # analysis along the POS used in the paper

    pos_syms = ['NOUN', 'PRON', 'ADJ', 'ADP', 'VERB']
    pos_names = ['Nouns', 'Pronouns', 'Adjectives', 'Adpositions', 'Verbs']

    if group_cols is not None:
        groups = df.groupby(group_cols)
        group_stats = []
        group_lens = []
        for n, gg in tqdm_notebook(groups):
            g_stats = defaultdict(set)
            group_lens.append(len(gg))
            for t, p in zip(gg.tokens, gg.pos):
                for x, y in zip(t, p):
                    g_stats[y[1]].add(x)
            group_stats.append(g_stats)

        for ps, pn in zip(pos_syms, pos_names):
            u_pos = []
            u_pos_norm = []
            for i, s in enumerate(group_stats):
                u_pos.append(len(s[ps]))
                u_pos_norm.append(u_pos[-1] / group_lens[i])
            print(pn, '{:.{}f}'.format(np.mean(u_pos), round_decimal),  '{:.{}f}'.format(np.mean(u_pos_norm), round_decimal))
    else:
        for ps, pn in zip(pos_syms, pos_names):
            print(pn, df.pos.apply(lambda x: len([i[0] for i in x if i[1] == ps])).mean().round(round_decimal))


