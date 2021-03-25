#!/usr/bin/env python
# coding: utf-8

"""
Combine, clean, pre-process ArtEmis annotations.

The MIT License (MIT)
Originally created by Panos Achlioptas at 6/17/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import nltk
import argparse
import pprint
import pathlib
import json
import numpy as np
import pandas as pd
import os.path as osp

from artemis.utils.basic import make_train_test_val_splits
from artemis.analysis.paintings_meta_data import masterpieces_for_test
from artemis.utils.vocabulary import build_vocab
from artemis.language.basics import tokenize_and_spell
from artemis.in_out.arguments import str2bool
from artemis.in_out.basics import pickle_data, create_dir
from artemis.emotions import emotion_to_int


up_dir = osp.split(pathlib.Path(__file__).parent.absolute())[0]

#
# two files necessary for spell-checking!
#
freq_file = osp.join(up_dir, 'data/symspell_frequency_dictionary_en_82_765.txt')
glove_file = osp.join(up_dir, 'data/glove.6B.100d.vocabulary.txt')


def parse_arguments(notebook_options=None):
    parser = argparse.ArgumentParser(description='Preprocess ArtEmis dataset')

    # Required arguments
    parser.add_argument('-save-out-dir', type=str, required=True, help='where to save the processed data')
    parser.add_argument('-raw-artemis-data-csv', type=str, required=True, help='dataset (csv file) provided upon YOUR request!')

    #
    # Optional arguments
    #
    # The default values will be used if not provided, and they correspond to how we processed ArtEmis for
    # training/testing neural-speakers.

    parser.add_argument('--random-seed', type=int, default=2021, help='used to make a train/test/val split')

    parser.add_argument('--too-long-utter-prc', type=int, default=100, help='drop utterances that are longer than this '
                                                                           'percentile. this percentile is evaluated '
                                                                           'in train-split, 100 means no dropping.')

    parser.add_argument('--too-short-len', type=int, default=0, help='drop utterances that have less tokens than this '
                                                                     'number.')

    parser.add_argument('--split-loads', type=float, default=[0.85, 0.05, 0.1], nargs=3,  help='train-val-test split '
                                                                                               'percentages.')

    parser.add_argument('--min-word-freq', type=int, default=0, help='words that appear less than this in the train-split '
                                                                     'will be replaced by <UNK>.')

    parser.add_argument('--automatic-spell-check', type=str2bool, default=True, help='apply token-spell-checking '
                                                                                     'automatically via SymSpell. Remark: '
                                                                                     'manually-curated spell-checking (see: '
                                                                                     'artemis.language.spelling.py) will '
                                                                                     'be applied even if you set this '
                                                                                     'to False.')

    parser.add_argument('--too-high-repetition', type=int, default=-1, help='For 701 artworks we collected at least 41'
                                                                            'annotations. We did this to test how this hyper-param (coverage)'
                                                                            'affects standard metrics like BLEU. These are an exemption to most artworks'
                                                                            'that are covered by 5 or sometimes a few more annotators.'
                                                                            'Artworks with this (too-high-repetition) annotations will be put '
                                                                            'in a special split called "rest" and not in the typical [train, test, val]. If this '
                                                                            'seems too wasteful and you want to include them (randomly) in the typical splits, set this param to -1.')

    parser.add_argument('--group-gt-anno', type=str2bool, default=True, help='group (and save) the annotations _per_ artwork.')

    parser.add_argument('--preprocess-for-deep-nets', type=str2bool, default=False, help='if you wish to preprocess ArtEmis in the way we did for training '
                                                                                         'all our deep-nets, please set this to True. It will apply some non '
                                                                                         'trivial filtering on the dataset.')

    parser.add_argument('--n-train-examples', type=int, help='(optional), keep at most these many utterances. can be '
                                                             'handy for fast debugging etc.')

    if notebook_options is not None:  # Pass options directly (useful inside say jupyter)
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args() # Read from command line.

    # overwrite some defaults to align the result with what we used for the deep-nets on our CVPR21 paper (same as arXiv v2).
    if args.preprocess_for_deep_nets:
        args.too_long_utter_prc = 95
        args.too_short_len = 5
        args.min_word_freq = 5
        args.too_high_repetition = 41 # for the CVPR21 results on the deep-nets we ignore high-coverage artworks

    args_string = pprint.pformat(vars(args))
    print(args_string)
    return args


def group_gt_annotations(preprocessed_dataframe, vocab):
    """ Group the annotations according to the underlying artwork/stimulus.
    :param preprocessed_dataframe: dataframe carrying ArtEmis annotations, spell-checked, with splits etc.
    :param vocab: the corresponding Vocabulary object
    :return: dictionary, carrying for each split (tran/test/val) a dataframe that has for each artwork all its collected
        annotations grouped.
    """
    df = preprocessed_dataframe
    results = dict()
    for split, g in df.groupby('split'): # group-by split
        g.reset_index(inplace=True, drop=True)
        g = g.groupby(['art_style', 'painting']) # group-by stimulus

        # group utterances / emotions
        # a) before "vocabularization" (i.e., raw)
        refs_pre_vocab_grouped = g['utterance_spelled'].apply(list).reset_index(name='references_pre_vocab')
        # b) post "vocabularization" (e.g., contain <UNK>)
        tokens_grouped = g['tokens_encoded'].apply(list).reset_index(name='tokens_encoded')
        emotion_grouped = g['emotion_label'].apply(list).reset_index(name='emotion')

        assert all(tokens_grouped['painting'] == emotion_grouped['painting'])
        assert all(tokens_grouped['painting'] == refs_pre_vocab_grouped['painting'])

        # decode these tokens back to strings and name them "references"
        tokens_grouped['tokens_encoded'] =\
            tokens_grouped['tokens_encoded'].apply(lambda x: [vocab.decode_print(sent) for sent in x])
        tokens_grouped = tokens_grouped.rename(columns={'tokens_encoded': 'references'})

        # join results in a new single dataframe
        temp = pd.merge(emotion_grouped, refs_pre_vocab_grouped)
        result = pd.merge(temp, tokens_grouped)
        result.reset_index(drop=True, inplace=True)
        results[split] = result
    return results


def preprocess(args, verbose=True, test_masterpieces=False):
    """ Split data, drop too short/long, spell-check, make a vocabulary etc.
    """

    #1. load the provided raw ArtEmis csv
    df = pd.read_csv(args.raw_artemis_data_csv)
    if verbose:
        print('{} annotations were loaded'.format(len(df)))

    #2. handle artwork with high repetition coverage (see explanation in args)
    if args.too_high_repetition != -1:
        normal_rep_mask = df.repetition < args.too_high_repetition
        # keep them separately temporarily
        high_coverage_df = df[~normal_rep_mask]
        high_coverage_df.reset_index(drop=True, inplace=True)
        high_coverage_df = high_coverage_df.assign(split=len(high_coverage_df)*['rest'])
        # do the train/test/val split (next step) based on the remaining annotations
        df = df[normal_rep_mask]
        df.reset_index(drop=True, inplace=True)

    #3. split the data in train/val/test  (the splits are based on the unique combinations of (art_work, painting).
    df = make_train_test_val_splits(df, args.split_loads, args.random_seed)

    # 3b. put back those with high_repetition
    if args.too_high_repetition != -1:
        df = pd.concat([df, high_coverage_df], 0)
        df.reset_index(inplace=True, drop=True)

    #3c. place some masterpieces in the test-set, these can be nice examples to showcasing.
    if test_masterpieces:
        df.loc[df.painting.isin(masterpieces_for_test), 'split'] = 'test'

    if args.n_train_examples is not None:
        print('sub-selecting from original dataset')
        df = df.iloc[:args.n_train_examples].reset_index(drop=True)

    #4. apply-spell checking
    missed_tokens = tokenize_and_spell(df, glove_file, freq_file, nltk.word_tokenize, spell_check=args.automatic_spell_check)

    #5. drop too long/short
    too_short_mask = df.tokens_len < args.too_short_len
    print('{} annotations will be dropped as they contain less than {} tokens'.format(sum(too_short_mask), args.too_short_len))
    too_long_len = np.percentile(df[df.split == 'train']['tokens_len'], args.too_long_utter_prc)
    too_long_mask = df.tokens_len > too_long_len
    print('Too-long token length at {}-percentile is {}. {} annotations will be dropped'.format(args.too_long_utter_prc,
                                                                                                too_long_len, sum(too_long_mask)))
    df = df[~too_short_mask & ~too_long_mask]
    df.reset_index(drop=True, inplace=True)

    #6. make a word-vocabulary based on training data
    train_tokens = df[df.split =='train']['tokens']
    vocab = build_vocab(train_tokens, args.min_word_freq)
    if verbose:
        print('Using a vocabulary with {} tokens'.format(len(vocab)))

    #7. encode tokens as ints
    max_len = int(too_long_len)
    df['tokens_encoded'] = df.tokens.apply(lambda x: vocab.encode(x, max_len))

    #8. encode feelings as ints
    df['emotion_label'] = df.emotion.apply(emotion_to_int)
    return df, vocab, missed_tokens


if __name__ == '__main__':

    ####################################################################################################################

    ## IMPORTANT. To replicate our CVPR21-paper (== arXiv [v2] upcoming manuscript):

    # You should run this script _twice_ which will create two differently pre-processed versions of ArtEmis.
    # Once, by doing _minimal_ preprocessing so that you can use the output to analyze "fairly" the ArtEmis
    # to other datasets and keep it in its purest form. For this run it with the default (non-required) parameters.

    # And secondly, if you want to do exactly the pre-processing we did for all the deep-nets we trained
    # (image-classifiers, neural-speakers etc.). Re-run it similarly to above, but this time add the optional argument:
    #   --preprocess-for-deep-nets True.

    #
    # If you want to understand the details, please see the help messages of the argparse, or open an issue at the repo.
    #

    ## Two words regarding the OUTPUT of this script:
    #      - artemis_preprocessed.csv a file containing the pre-processed ArtEmis.
    #      - vocabulary.pkl stores an object of utils.vocabulary.py (e.g., see Step-3/4 for how this is used)
    #      - config.json.txt a json storing the arg-parse arguments you used
    #      - artemis_gt_references_grouped.pkl pre-processed ArtEmis with each artwork directly pointing to all
    #                                          its annotation. Useful for evaluation metrics, e.g., BLEU.

    # Panos, March 22, 2022
    ####################################################################################################################

    args = parse_arguments()
    create_dir(args.save_out_dir)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Clean, Tokenize, Make tran/test splits etc.
    df, vocab, missed_tokens = preprocess(args)
    df.to_csv(osp.join(args.save_out_dir, 'artemis_preprocessed.csv'), index=False)
    vocab.save(osp.join(args.save_out_dir, 'vocabulary.pkl'))
    with open(osp.join(args.save_out_dir, 'config.json.txt'), 'w') as f_out:
        json.dump(vars(args), f_out, indent=4)

    print('n-utterances kept:', len(df))
    print('vocab size:', len(vocab))
    if args.automatic_spell_check:
        print('tokens not in Glove/Manual vocabulary:', len(missed_tokens))

    # Save separately the grouped utterances of each stimulus (can be used for evaluation, e.g., for novelty-metrics).
    if args.group_gt_anno:
        groups = group_gt_annotations(df, vocab)
        pickle_data(osp.join(args.save_out_dir, 'artemis_gt_references_grouped.pkl'), groups)

    print(f'Done. Check saved results in provided save-out-dir: {args.save_out_dir}')