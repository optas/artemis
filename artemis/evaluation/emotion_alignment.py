"""
Measuring the emotion-alignment between a generation and the ground-truth (emotion).

The MIT License (MIT)
Originally created at 8/31/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import torch
import numpy as np
from ..utils.basic import iterate_in_chunks


@torch.no_grad()
def image_to_emotion(img2emo_clf, data_loader, device):
    """ For each image of the underlying dataset predict an emotion
    :param img2emo_clf: nn.Module
    :param data_loader: torch loader of dataset to iterate
    :param device: gpu placement
    :return:
    """
    img2emo_clf.eval()
    emo_of_img_preds = []
    for batch in data_loader:
        predictions = img2emo_clf(batch['image'].to(device)).cpu()
        emo_of_img_preds.append(predictions)
    emo_of_img_preds = torch.cat(emo_of_img_preds)
    return emo_of_img_preds


@torch.no_grad()
def text_to_emotion(txt2em_clf, encoded_tokens, device, batch_size=1000):
    """
    :param txt2em_clf:
    :param encoded_tokens: Tensor carrying the text encoded
    :param device:
    :param batch_size:
    :return:
    """
    txt2em_clf.eval()
    emotion_txt_preds = []
    for chunk in iterate_in_chunks(encoded_tokens, batch_size):
        emotion_txt_preds.append(txt2em_clf(chunk.to(device)).cpu())

    emotion_txt_preds = torch.cat(emotion_txt_preds)
    maximizers = torch.argmax(emotion_txt_preds, -1)
    return emotion_txt_preds, maximizers


def unique_maximizer(a_list):
    """ if there is an element of the input list that appears
    strictly more frequent than any other element
    :param a_list:
    :return:
    """
    u_elements, u_cnt = np.unique(a_list, return_counts=True)
    has_umax = sum(u_cnt == u_cnt.max()) == 1
    umax = u_elements[u_cnt.argmax()]
    return has_umax, umax


def dominant_maximizer(a_list):
    """ if there is an element of the input list that appears
    at least half the time
    :param a_list:
    :return:
    """
    u_elements, u_cnt = np.unique(a_list, return_counts=True)

    has_umax = u_cnt.max() >= len(a_list) / 2

    if len(u_cnt) >= 2: # make sure the second most frequent does not match the first.
        a, b = sorted(u_cnt)[-2:]
        if a == b:
            has_umax = False

    umax = u_elements[u_cnt.argmax()]
    return has_umax, umax


def occurrence_list_to_distribution(list_of_ints, n_support):
    """e.g., [0, 8, 8, 8] -> [1/4, 0, ..., 3/4, 0, ...]"""
    distribution = np.zeros(n_support, dtype=np.float32)
    for i in list_of_ints:
        distribution[i] += 1
    distribution /= sum(distribution)
    return distribution