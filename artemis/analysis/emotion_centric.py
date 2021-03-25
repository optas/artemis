"""
Utilities for emotion-centric analysis.

The MIT License (MIT)
Originally created at 10/22/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import pandas as pd
import matplotlib.pylab as plt

from ..emotions import ARTEMIS_EMOTIONS, positive_negative_else


def df_to_emotion_histogram(df, palette=plt.cm.Pastel1, emotion_column='emotion', verbose=False):
    """ Take a dataset like ArtEmis and return a histogram over the emotion choices made by the annotators.
    :param df: dataframe carrying dataset
    :param palette: matplotlib color palette, e.g., plt.cm.jet
    :param emotion_column: (str) indicate which column of the dataframe carries the emotion
    :return: a list carrying the resulting histogram figure.
    """
    hist_vals = []
    for emotion in ARTEMIS_EMOTIONS:
        hist_vals.append(sum(df[emotion_column] == emotion) / len(df))

    norm = plt.Normalize(min(hist_vals), max(hist_vals))
    colors = palette(norm(hist_vals))

    s = pd.DataFrame({"emotions": ARTEMIS_EMOTIONS, "vals": hist_vals})
    s.set_index("emotions", drop=True, inplace=True)
    plt.figure()
    s.index.name = None
    ax = s.plot.bar(grid=True, figsize=(12,4), color=colors, fontsize=16, rot=45, legend=False, ec="k")
    ax.set_ylabel('Percentage of data', fontsize=15)

    for rec, col in zip(ax.patches, colors):
        rec.set_color(col)

    plt.tight_layout()
    res = [plt.gcf()]

    plt.figure()
    s = df[emotion_column].apply(positive_negative_else).value_counts() / len(df)

    if verbose:
        print('Pos-Neg-Else, percents:', s.round(3))

    ax = s.plot.bar(grid=True, figsize=(8,4), fontsize=16, rot=45, legend=False, color='gray')
    ax.set_xticklabels(['positive', 'negative', 'else'])
    plt.tight_layout()
    res.append(plt.gcf())

    return res


def has_emotion_max_dominance(grouped_df, exclude_se=False, return_max=False):
    """ I.e., same emotion was selected (among all nine emotions) at least by half annotators.
    :param grouped_df: dataframe of dataset grouped by stimuli, e.g., images.
    :param exclude_se: if True, ignore the groups where the maximizer is the something-else category
    :param return_max: return for each group that has dominance the emotion type that has the gathered the maximum annotations.
    :return:
    """
    vals = grouped_df.emotion.value_counts()
    maxim = vals.max()
    threshold = vals.sum() / 2
    res = maxim >= threshold
    if exclude_se:
        res &= vals.idxmax() != 'something else'
    if return_max:
        return res, vals.idxmax()
    else:
        return res