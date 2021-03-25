"""
Helper functions for sampling (@test -- inference-time) a neural-speaker.

The MIT License (MIT)
Originally created at 20/1/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from ..neural_models.attentive_decoder import sample_captions, sample_captions_beam_search, properize_captions
from ..in_out.basics import wikiart_file_name_to_style_and_painting
from ..emotions import IDX_TO_EMOTION
from ..utils.vocabulary import UNK


def versatile_caption_sampler(speaker, data_loader, device, max_utterance_len, sampling_rule='beam',
                              beam_size=None, topk=None, temperature=1, drop_unk=True, use_bert_unk=False,
                              drop_bigrams=False):
    """Provides all implemented sampling methods according to the sampling_rule input parameter.
    """
    vocab = speaker.decoder.vocab

    if sampling_rule == 'beam':
        dset = data_loader.dataset
        loader = DataLoader(dset, num_workers=data_loader.num_workers) # batch-size=1

        max_iter = 8 * max_utterance_len # should be large enough
        beam_captions, alphas, beam_scores = sample_captions_beam_search(speaker, loader, beam_size,
                                                                         device, max_iter=max_iter,
                                                                         temperature=temperature,
                                                                         drop_unk=drop_unk,
                                                                         drop_bigrams=drop_bigrams)
        # first is highest scoring caption which is the only we keep here
        captions = [c[0] for c in beam_captions]
        alphas = [np.array(a[0]) for a in alphas]  # each alpha covers all tokens: <sos>, token1, ..., <eos>
    else:
        captions, alphas = sample_captions(speaker, data_loader, max_utterance_len=max_utterance_len,
                                           sampling_rule=sampling_rule, device=device, temperature=temperature,
                                           topk=topk, drop_unk=drop_unk, drop_bigrams=drop_bigrams)

        captions = properize_captions(captions, vocab).tolist()
    captions = tokens_to_strings(captions, vocab, bert_unk=use_bert_unk)
    return captions, alphas


def captions_as_dataframe(captions_dataset, captions_predicted, wiki_art_data=True):
    """convert the dataset/predicted-utterances (captions) to a pandas dataframe."""
    if wiki_art_data:
        temp = captions_dataset.image_files.apply(wikiart_file_name_to_style_and_painting)
        art_style, painting = zip(*temp)
        grounding_emotion = [IDX_TO_EMOTION.get(x, None) for x in captions_dataset.emotions.tolist()]
        df = pd.DataFrame([art_style, painting, grounding_emotion, captions_predicted]).transpose()
        column_names = ['art_style', 'painting', 'grounding_emotion', 'caption']
        df.columns = column_names
    else:
        image_files = captions_dataset.image_files.tolist()
        grounding_emotion = [IDX_TO_EMOTION.get(x, None) for x in captions_dataset.emotions.tolist()]
        df = pd.DataFrame([image_files, grounding_emotion, captions_predicted]).transpose()
        column_names = ['image_file', 'grounding_emotion', 'caption']
        df.columns = column_names
    return df


def tokens_to_strings(token_list, vocab, bert_unk=True):
    """ Bert uses [UNK] to represent the unknown symbol.
    :param token_list:
    :param vocab:
    :param bert_unk:
    :return:
    """
    res = [vocab.decode_print(c) for c in token_list]
    if bert_unk:
        res = [c.replace(UNK, '[UNK]') for c in res]
    return res

