"""
Part of speech tagging at speed for two libraries.

The MIT License (MIT)
Originally created in 2020, for Python 3.x - last updated in early 2021.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import dask.dataframe as dd
import multiprocessing as mp
from nltk.tag import pos_tag

try:
    import spacy
except:
    pass


def nltk_parallel_tagging_of_tokens(tokens, n_partitions=None, tagset='universal'):
    """ pos-tagging
    :param tokens: pd.Series with tokenized utterances as rows. e.g., [['a', 'man'], ['a', 'big', 'man'], ...]
    :return: a pd.Series with the result of applying pos_tag in each row. e.g.,
        [(a, DT), (man, NN)], [('a', 'DT'), ('big', 'JJ'), ('man', 'NN')]]
    """
    if n_partitions is None:
        n_partitions = mp.cpu_count() * 4
    ddata = dd.from_pandas(tokens, npartitions=n_partitions)
    tagged_tokens =\
        ddata.map_partitions(lambda x: x.apply((lambda y: pos_tag(y, tagset=tagset)))).compute(scheduler='processes')

    return tagged_tokens


def spacy_pos_tagging(utterances, nlp=None):
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')

    utters = utterances.astype('unicode').values
    docs = nlp.pipe(utters, batch_size=1000, n_threads=-1)
    pos = [[t.pos_ for t in d if not t.is_space] for d in docs]
    return pos
