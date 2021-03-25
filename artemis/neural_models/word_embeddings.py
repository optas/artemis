"""
Utilities to load pretrained word embeddings like those of GloVe.

The MIT License (MIT)
Originally created in late 2019, for Python 3.x - last updated in 2021.
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import warnings
import torch
import numpy as np
from collections import Counter


def make_pretrained_embedding(vocab, pretrained_vectors, freeze=True, sigma=1, random_seed=None):
    """ Make a torch.nn.Embedding based for a given vocabulary and a collection of
    pretrained word-embedding vectors.
    :param vocab: speakers_listeners.build_vocab.Vocabulary
    :param pretrained_vectors: dictionary of words mapped to np.array vectors
    (like those returned from ```load_glove_pretrained_embedding```).
    :param freeze, (opt, boolean) if True the embedding is not using gradients to optimize itself (fine-tune).
    :param sigma, (opt, int) standard-deviation of Gaussian used to sample when a word is not in the pretrained_vectors
    :param random_seed (opt, int) to seed the numpy Gaussian
    :return: torch.nn.Embedding

    Note: this implementation will freeze all words if freeze=True, irrespectively of if the words are in the
    pretrained_vectors collection or not (OOV: Out-of-Vocabulary). If you want to fine-tune the OOV you need to adapt
    like this: https://discuss.pytorch.org/t/updating-part-of-an-embedding-matrix-only-for-out-of-vocab-words/33297
    """
    for ss in vocab.special_symbols:
        if ss in pretrained_vectors:
            warnings.warn('the special symbol {} is found in the pretrained embedding.')

    # Initialize weight matrix with correct dimensions and all zeros
    random_key = next(iter(pretrained_vectors))
    emb_dim = len(pretrained_vectors[random_key])
    emb_dtype = pretrained_vectors[random_key].dtype
    n_words = len(vocab)
    weights = np.zeros((n_words, emb_dim), dtype=emb_dtype)

    if random_seed is not None:
        np.random.seed(random_seed)

    for word, idx in vocab.word2idx.items():
        if word in pretrained_vectors:
            weights[idx] = pretrained_vectors[word]
        else:
            weights[idx] = sigma * np.random.randn(emb_dim)

    padding_idx = None
    if hasattr(vocab, 'pad'):
        print('using padding symbol of provided vocabulary.')
        padding_idx = vocab.pad
        weights[padding_idx] = np.zeros(emb_dim)

    embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(weights), freeze=freeze, padding_idx=padding_idx)
    return embedding


def load_glove_pretrained_embedding(glove_file, dtype=np.float32, only_words=False, verbose=False):
    """
    :param glove_file: file downloaded from Glove website
    :param dtype: how to save the word-embeddings
    :param only_words: do not return the embedding vectors, only the words considered
    :param verbose: print, or not side-information
    :return: dictionary of words mapped to np.array vectors
    """

    if verbose:
        print("Loading glove word embeddings.")

    embedding = dict()
    with open(glove_file) as f_in:
        for line in f_in:
            s_line = line.split()
            token = s_line[0]
            if only_words:
                embedding[token] = 0
            else:
                w_embedding = np.array([float(val) for val in s_line[1:]], dtype=dtype)
                embedding[token] = w_embedding
    if only_words:
        embedding = set(list(embedding.keys()))

    if verbose:
        print("Done.", len(embedding), "words loaded.")
    return embedding


def init_token_bias(encoded_token_list, vocab=None, dtype=np.float32, trainable=True):
    """ Make a bias vector based on the (log) probability of the frequency of each word
    in the training data similar to https://arxiv.org/abs/1412.2306
    This bias can used to initialize the hidden-to-next-word layer for faster convergence.
    :param encoded_token_list: [[tokens-of-utterance-1-as-ints] [tokens-of-utterance-2]...]
    :param vocab: speakers_listeners.build_vocab.Vocabulary
    :param dtype:
    :param trainable: (opt, bool) permit training or not of the resulting bias vector
    :return: (torch.Parameter) bias vector
    """
    counter = Counter()
    for tokens in encoded_token_list:
        counter.update(tokens)

    n_items = len(counter)
    if vocab is not None:
        if n_items != len(vocab):
            warnings.warn('init_token_bias: Vobab contains more tokens than given token lists.')
            n_items = max(n_items, len(vocab))
        counter[vocab.sos] = counter[vocab.pad] = min(counter.values())

    bias_vector = np.ones(n_items, dtype=dtype) # initialize

    for position, frequency in counter.items():
        bias_vector[position] = frequency

    #  Log probability
    bias_vector /= np.sum(bias_vector)
    bias_vector = np.log(bias_vector)
    bias_vector -= np.max(bias_vector)

    bias_vector = torch.from_numpy(bias_vector)
    bias_vector = torch.nn.Parameter(bias_vector, requires_grad=trainable)
    return bias_vector