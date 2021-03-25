"""
Visualization utilities for playing with ArtEmis.

The MIT License (MIT)
Originally created early 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import numpy as np
import pandas as pd
import warnings
import skimage.transform
import matplotlib.pylab as plt
import matplotlib.cm as cm
import os.path as osp
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from PIL import Image

try:
    import plotly.figure_factory as ff
except:
    print('Plotly installation not found; some visualization features will not work.')


def plot_confusion_matrix(ground_truth, predictions, labels, normalize=True, round_decimal=2):
    """
    :param ground_truth: 1dim-iterable
    :param predictions: 1dim-iterable
    :param labels: list of strings describing the classes
    :param normalize: show raw confusion statistics or normalized to sum to 1
    :param round_decimal:
    :return:
    """
    cm = confusion_matrix(ground_truth, predictions)
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]

    cm_text = pd.DataFrame.from_records(cm).round(round_decimal)

    figure = ff.create_annotated_heatmap(z=cm, x=labels, y=labels,
                                         annotation_text=cm_text,
                                         showscale=True)
    return figure


def visualize_random_caption(df, top_img_dir, specific_id=None, imsize=(512, 512)):
    """

    :param df: pandas dataframe storing ArtEmis annotations
    :param top_img_dir: where you stored WikiArt
    :param specific_id: basically the row of the df (when provided randomness goes away).
    :param imsize:
    :return:
    """
    if specific_id is None:
        painting, art_style = df.sample(1)[['painting', 'art_style']].iloc[0]
    else:
        painting, art_style = df.iloc[specific_id][['painting', 'art_style']]

    print(art_style, painting)
    Image.open(osp.join(top_img_dir, art_style, painting + '.jpg')).resize(imsize)
    s = df[(df.painting == painting) & (df.art_style == art_style)]

    for e, u in zip(s['grounding_emotion'], s['caption']):
        print('{}:\t{}'.format(e.upper(), u))


def show_image_in_df_loc(df, loc, top_in_dir=None, img_column='Input.image_url', processed=False):
    if top_in_dir is None:
        top_in_dir = '/home/optas/DATA/Images/Wiki-Art/rescaled_max_size_to_600px_same_aspect_ratio'

    if processed:
        a = df.loc[loc]['art_style']
        n = df.loc[loc]['painting']
        f = osp.join(top_in_dir, a, n + '.jpg')
    else:
        a, n = df.loc[loc][img_column].split('/')[-2:]
        f = osp.join(top_in_dir, a, n)
    return Image.open(f)


def plot_overlayed_two_histograms(x1, x2, min_val, max_val, n_bins, labels, alpha=0.8):
    """ Plot the values of x1, x2 as two overlayed histograms.
    :param x1:
    :param x2:
    :param min_val:
    :param max_val:
    :param n_bins:
    :param labels:
    :param alpha:
    :return:
    """
    bins = np.linspace(min_val, max_val, n_bins)
    fig = pyplot.figure()
    pyplot.hist(x1, bins, alpha=alpha, label=labels[0])
    pyplot.hist(x2, bins, alpha=alpha, label=labels[1])
    pyplot.legend(loc='upper right')
    return fig


def visualize_attention_map_per_token(image, tokens, attention_map, pixel_upscale, smooth_attention=True,
                                      sigma=None, **kwargs):
    """
    :param image: PIL image
    :param tokens: list of strings (each being a token)
    :param attention_map: (np.array) for each token an attention map over the image
    :param pixel_upscale: re-scale the KxK input attention_map by this amount (in pixels)
    :param smooth_attention: (opt, boolean) to smooth the displayed attention values
    :param sigma: (opt, float) control ammount of smoothing
    :param kwargs: common parameters for matplotlib {'figsize', 'fontsize'}
    :return: nothing
    """
    figsize = kwargs.pop('figsize', 20)
    fontsize = kwargs.pop('fontsize', 14)

    plt.figure(figsize=(figsize, figsize))
    n_tokens, h, w = attention_map.shape
    if n_tokens != len(tokens):
        raise ValueError('Each token is expected to have an attention map.')

    if h != w:
        warnings.warn('code has not been tested')

    new_im_size = [pixel_upscale*h, pixel_upscale*w]
    image = image.resize(new_im_size, Image.LANCZOS)

    for t in range(n_tokens):
        plt.subplot(np.ceil(n_tokens / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (tokens[t]), color='black', backgroundcolor='white', fontsize=fontsize)
        plt.imshow(image)
        current_alpha = attention_map[t][:]

        if smooth_attention:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=pixel_upscale, sigma=sigma, multichannel=False)
        else:
            alpha = skimage.transform.resize(current_alpha, new_im_size)

        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()