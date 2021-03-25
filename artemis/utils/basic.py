"""
Various simple (basic) functions in the "utilities".

The MIT License (MIT)
Originally created at 8/31/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import torch
import multiprocessing as mp
import dask.dataframe as dd
from torch import nn
from sklearn.model_selection import train_test_split


def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def df_parallel_column_apply(df, func, column_name):
    n_partitions = mp.cpu_count() * 4
    d_data = dd.from_pandas(df, npartitions=n_partitions)

    res =\
    d_data.map_partitions(lambda df: df.apply((lambda row: func(row[column_name])), axis=1))\
    .compute(scheduler='processes')

    return res


def cross_entropy(pred, soft_targets):
    """ pred: unscaled logits
        soft_targets: target-distributions (i.e., sum to 1)
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def make_train_test_val_splits(datataset_df, loads, random_seed, unique_id_column=None):
    """ Split the data into train/val/test.
    :param datataset_df: pandas Dataframe containing the dataset (e.g., ArtEmis)
    :param loads: list with the three floats summing to one for train/val/test
    :param random_seed: int
    :return: changes the datataset_df in-place to include a column ("split") indicating the split of each row
    """
    if sum(loads) != 1:
        raise ValueError()

    train_size, val_size, test_size = loads
    print("Using a {},{},{} for train/val/test purposes".format(train_size, val_size, test_size))

    df = datataset_df
    ## unique id
    if unique_id_column is None:
        unique_id = df.art_style + df.painting # default for ArtEmis
    else:
        unique_id = df[unique_id_column]

    unique_ids = unique_id.unique()
    unique_ids.sort()

    train, rest = train_test_split(unique_ids, test_size=val_size+test_size, random_state=random_seed)
    train = set(train)

    if val_size != 0:
        val, test = train_test_split(rest, test_size=round(test_size*len(unique_ids)), random_state=random_seed)
    else:
        test = rest
    test = set(test)
    assert len(test.intersection(train)) == 0

    def mark_example(x):
        if x in train:
            return 'train'
        elif x in test:
            return 'test'
        else:
            return 'val'

    df = df.assign(split=unique_id.apply(mark_example))
    return df