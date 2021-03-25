"""
Data Cleaning Utilities.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import pathlib
import os.path as osp
from tqdm import tqdm_notebook as tqdm
from ..in_out.basics import unpickle_data, splitall


def load_duplicate_paintings_of_wikiart(duplicates_pkl_file=None, verbose=True):
    """ Return a list containing wikiArt paintings that are double-listed.
    :param duplicates_pkl_file: (opt) pkl file containing the duplicate groups.
    :return: (list of list) each sublist contains tuples like (art_style, painting) that are duplicates.

    Note.  If duplicates_pkl_file==None, the stored inside the repo .pkl file will be used. The duplicates indicated in
    the .pkl were found by a combination of running the `fdupes' program and a manual check on Nearest-Neighbors of a
    pretrained ResNet on ImageNet that had very small distances.
    """
    if duplicates_pkl_file is None:
        up_dir = osp.split(pathlib.Path(__file__).parent.absolute())[0]
        duplicates_pkl_file = osp.join(up_dir, 'data/wiki_art_duplicate_paintings.pkl')
        # Note. This file contains duplicates that were found using
    duplicates_as_list = next(unpickle_data(duplicates_pkl_file))
    if verbose:
        print("Using {} groups of paintings that are visually identical (duplicates).".format(len(duplicates_as_list)))
    return duplicates_as_list


def drop_duplicate_paintings(wiki_art_image_files, duplicate_groups=None):
    """
    :param wiki_art_image_files: (list) with filenames of the form xx/xx/art_style/painting.jpg
    :param duplicate_groups: list of list, each item is a collection of (art_style, painting) tuples that are duplicates.
    :return: a new list where from each duplicate group only one (the first) painting is kept.
    """
    if duplicate_groups is None:
        duplicate_groups = load_duplicate_paintings_of_wikiart()

    drop_these = set()
    for dup_g in duplicate_groups:
        drop_these.update(dup_g[1:]) # drop all but first

    clean_img_files = []
    dropped = 0
    for img_file in wiki_art_image_files:
        tokens = splitall(img_file)
        painting = tokens[-1][:-len('.jpg')]
        art_style = tokens[-2]
        key = (art_style, painting)
        if key in drop_these:
            dropped += 1
        else:
            clean_img_files.append(img_file)
    print('Dropping {} from {} paintings that are duplicates of one painting that is kept.'.format(dropped,
                                                                                                   len(wiki_art_image_files)))
    return clean_img_files


def merge_artemis_annotations_on_wikiart_duplicates(dataset_df, duplicate_groups=None, verbose=True):
    """
    :param dataset_df:
    :param duplicate_groups:
    :return:
    """

    if duplicate_groups is None:
        duplicate_groups = load_duplicate_paintings_of_wikiart()

    n_merged_stimuli = 0
    for dup_g in tqdm(duplicate_groups):
        keep_this = dup_g[0]
        drop_these = dup_g[1:] # drop all but first
        for stimulus in drop_these:
            mask = (dataset_df['art_style'] == stimulus[0]) & (dataset_df['painting'] == stimulus[1])
            n_merged_stimuli += sum(mask)
            dataset_df.loc[mask, ['art_style']] = keep_this[0]
            dataset_df.loc[mask, ['painting']] = keep_this[1]
    if verbose:
        print('{} stimuli were merged.'.format(n_merged_stimuli))
    return dataset_df



