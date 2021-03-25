"""
Minimal operations for loading conceptual_captions dataset.

The MIT License (MIT)
Originally created in early 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import pandas as pd


def load_conceptual_captions(train_tsv, val_tsv, only_language=True):
    if only_language:
        use_cols = [0]
    else:
        use_cols = [0, 1]

    df1 = pd.read_csv(train_tsv, sep='\t', header=None, usecols=use_cols)
    df2 = pd.read_csv(val_tsv, sep='\t', header=None, usecols=use_cols)
    df = pd.concat([df1, df2])
    df.reset_index(drop=True, inplace=True)

    df = df.rename(columns={0:'utterance'})

    if not only_language:
        df = df.rename(columns={1:'image_id'})

    return df