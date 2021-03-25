"""
Basic (simple) I/O Utilities.

The MIT License (MIT)
Originally created in 2019, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import re
import os
import json
import sys
import numpy as np
import pandas as pd
import os.path as osp
import pprint
import logging
from argparse import ArgumentParser
from IPython.display import display
from PIL import Image
from six.moves import cPickle, range
from ..emotions import ARTEMIS_EMOTIONS


def files_in_subdirs(top_dir, search_pattern):
    join = osp.join
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                yield full_name


def create_dir(dir_path):
    """ Creates a directory (or nested directories) if they don't exist.
    """
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """ Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def load_raw_amt_csv_hit_responses(top_csv_folder, verbose=True, only_approved=True,
                                   keep_cols=None, drop_rorschach=True, has_emotions=True):
    """
    :param top_csv_folder:
    :param verbose:
    :param only_approved:
    :param keep_cols:
    :param drop_rorschach:
    :param has_emotions: set to False to load wiki-art annotations that are objective (OLA-dataset)
    :return:
    """

    all_collected_csv = [f for f in files_in_subdirs(top_csv_folder, '.csv$')]

    if verbose:
        print('{} files loaded'.format(len(all_collected_csv)))

    all_csv_names = [osp.basename(f) for f in all_collected_csv]
    assert len(all_csv_names) == len(set(all_csv_names)) # unique names

    all_dfs = []
    for f in all_collected_csv:  # load each .csv
        df = pd.read_csv(f)
        # print(df['AssignmentStatus'].unique())
        in_submission_mode = (df['AssignmentStatus'] == 'Submitted').sum()
        if in_submission_mode > 0:
            print('In {}, {} examples are still in submitted mode.'.format(osp.basename(f), in_submission_mode))
        if only_approved:
            df = df[df['AssignmentStatus'] == 'Approved']
        all_dfs.append(df)
    df = pd.concat(all_dfs)

    # Rename columns
    new_cols = [c.replace('choice.', '') for c in [c.replace('Answer.', '') for c in df.columns]]
    new_cols = [c.lower() for c in new_cols]
    df.columns = new_cols
    df = df.reset_index()

    # Keep ML-related columns
    ml_related_cols = ['workerid', 'input.image_url', 'utterance']
    # Add potential extras requested at the input
    if keep_cols is not None:
        ml_related_cols += keep_cols

    if has_emotions:
        _, x = np.where(df[ARTEMIS_EMOTIONS])
        emotion_chosen = pd.Series(np.array(ARTEMIS_EMOTIONS)[x], name='emotion')
        df = pd.concat([df[ml_related_cols], emotion_chosen], axis=1)
    else:
        df = df[ml_related_cols]

    # Derivative columns
    def url_to_painting_name(x):
        tokens = x.split('/')
        return tokens[-1][:-len('.jpg')]

    def url_to_art_style(x):
        tokens = x.split('/')
        return tokens[-2]

    df['painting'] = df['input.image_url'].apply(url_to_painting_name)
    df['art_style'] = df['input.image_url'].apply(url_to_art_style)
    df = df.drop(['input.image_url'], axis=1)

    if drop_rorschach:
        df = df[df['art_style'] != 'test']
        df.reset_index(inplace=True, drop=True)

    if verbose:
        print('Loading responses:', len(df))
        print('Column Names:', [c for c in df.columns])

    return df


def splitall(path):
    """
    Examples:
        splitall('a/b/c') -> ['a', 'b', 'c']
        splitall('/a/b/c/')  -> ['/', 'a', 'b', 'c', '']

    NOTE: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = osp.split(path)
        if parts[0] == path:   # Sentinel for absolute paths.
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # Sentinel for relative paths.
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def wikiart_file_name_to_style_and_painting(filename):
    """
    Assumes a filename of a painting of wiki-art.
    :param filename:
    :return:
    """
    s = splitall(filename)
    return s[-2], s[-1][:-len('.jpg')]


def show_random_captions(df, top_img_dir):
    painting, art_style = df.sample(1)[['painting', 'art_style']].iloc[0]
    print(art_style, painting)
    display(Image.open(osp.join(top_img_dir, art_style, painting + '.jpg')))
    s = df[(df.painting == painting) & (df.art_style == art_style)]
    for e, u in zip(s['emotion'], s['utterance']):
        print('{}:\t{}'.format(e.upper(), u))


def read_saved_args(config_file, override_args=None, verbose=False):
    """
    :param config_file: json file containing arguments
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(osp.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger