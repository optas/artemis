"""
Some grouping of various evaluation evaluation routines that assume that assume that for a given set of reference
sentences there is a _single_ caption (sample) generated.

The MIT License (MIT)
Originally created at 9/1/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import torch
import warnings
import pandas as pd
import numpy as np


from .bleu import sentence_bleu_for_hypotheses, cc
from .metaphors import makes_metaphor_via_substring_matching
from .emotion_alignment import text_to_emotion
from .pycocoevalcap import Bleu, Cider, Meteor, Spice, Rouge
from .emotion_alignment import dominant_maximizer, occurrence_list_to_distribution
from .longest_common_subseq import captions_lcs_from_training_utterances
from ..utils.basic import cross_entropy

ALL_METRICS = {'bleu', 'cider', 'spice', 'meteor', 'rouge', 'emo_alignment', 'metaphor', 'lcs'}


def emotional_alignment(hypothesis, emotions, vocab, txt2em_clf, device):
    """ text 2 emotion, then compare with ground-truth.
    :param hypothesis:
    :param emotions: (list of list of int) human emotion-annotations (ground-truth) e.g., [[0, 1] [1]]
    :param vocab:
    :param txt2em_clf:
    :param device:
    :return:
    """

    # from text to emotion
    hypothesis_tokenized = hypothesis.apply(lambda x: x.split())
    max_len = hypothesis_tokenized.apply(lambda x: len(x)).max()
    hypothesis = hypothesis_tokenized.apply(lambda x: np.array(vocab.encode(x, max_len=max_len)))
    hypothesis = torch.from_numpy(np.vstack(hypothesis))
    pred_logits, pred_maximizer = text_to_emotion(txt2em_clf, hypothesis, device)

    # convert emotion lists to distributions to measure cross-entropy
    n_emotions = 9
    emo_dists = torch.from_numpy(np.vstack(emotions.apply(lambda x: occurrence_list_to_distribution(x, n_emotions))))
    x_entropy = cross_entropy(pred_logits, emo_dists).item()

    # constrain predictions to those of images with dominant maximizer of emotion
    has_max, maximizer = zip(*emotions.apply(dominant_maximizer))
    emotion_mask = np.array(has_max)
    masked_emotion = np.array(maximizer)[emotion_mask]

    guess_correct = masked_emotion == pred_maximizer[emotion_mask].cpu().numpy()
    accuracy = guess_correct.mean()

    return accuracy, x_entropy


def bleu_scores_via_nltk(hypothesis, references, smoothing_function=cc.method1):
    """
    :param hypothesis: dataframe of strings
    :param references: dataframe of list of strings
    :param smoothing_function:
    :return:
    """

    # first tokenize
    hypothesis_tokenized = hypothesis.apply(lambda x: x.split())
    references_tokenized = references.apply(lambda x: [i.split() for i in x])

    results = dict()
    for max_grams in range(1, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = sentence_bleu_for_hypotheses(references_tokenized,
                                                  hypothesis_tokenized,
                                                  max_grams,
                                                  smoothing_function)
            results['BLEU-{}'.format(max_grams)] = scores
    return results


def dataframes_to_coco_eval_format(references, hypothesis):
    references = {i: [k for k in x] for i, x in enumerate(references)}
    hypothesis = {i: [x] for i, x in enumerate(hypothesis)}
    return references, hypothesis


def pycoco_bleu_scores(hypothesis, references):
    references, hypothesis = dataframes_to_coco_eval_format(references, hypothesis)
    scorer = Bleu()
    average_score, all_scores = scorer.compute_score(references, hypothesis)
    # Note: average_score takes into account epsilons: tiny/small
    # this won't be reflected if you take the direct average of all_scores.
    return average_score, all_scores


def pycoco_eval_scores(hypothesis, references, metric):
    references, hypothesis = dataframes_to_coco_eval_format(references, hypothesis)
    if metric == 'cider':
        scorer = Cider()
    elif metric == 'meteor':
        scorer = Meteor()
    elif metric == 'spice':
        scorer = Spice()
    elif metric == 'rouge':
        scorer = Rouge()
    else:
        raise ValueError
    avg, all_scores = scorer.compute_score(references, hypothesis)
    return pd.Series(all_scores)


def apply_basic_evaluations(hypothesis, references, ref_emotions, txt2emo_clf, text2emo_vocab,
                            lcs_sample=None, train_utterances=None, nltk_bleu=False, smoothing_function=cc.method1,
                            device="cuda", random_seed=2021,
                            methods_to_do=ALL_METRICS):
    """
    :param hypothesis: list of strings ['a man', 'a woman']
    :param references: list of list of strings [['a man', 'a tall man'], ['a woman']]
    :param ref_emotions: emotions corresponding to references list of list of integers [[0, 1] [1]]

    :param text2emo_vocab:
    :param txt2emo_clf:
    :param device:
    :param smoothing_function:
    :return:
    """
    results = []
    stat_track = ['mean', 'std']

    ##
    ## BLEU:1-4
    ##
    if 'bleu' in methods_to_do:
        if nltk_bleu:
            res = bleu_scores_via_nltk(hypothesis, references, smoothing_function=smoothing_function)
            for metric, scores in res.items():
                stats = scores.describe()[stat_track]
                stats = pd.concat([pd.Series({'metric': metric}), stats])
                results.append(stats)
        else:
            #py-coco based
            b_scores = pycoco_bleu_scores(hypothesis, references)
            for i in range(4):
                metric = f'BLEU-{i}'
                mu = b_scores[0][i]
                # note the std below reflects the values without the 'tiny' adaptation (unlike the mu)
                # avg_dummy = np.mean(b_scores[1][i]) # this is the average without the tiny adaptation.
                std = np.std(b_scores[1][i])
                stats = pd.concat([pd.Series({'metric': metric}), pd.Series({'mean': mu, 'std':std})])
                results.append(stats)
        print('BLEU: done')

    ##
    ## CIDER, SPICE, METEOR, ROUGE-L
    ##
    coco_requested = False
    for metric in ['cider', 'spice', 'meteor', 'rouge']:
        if metric in methods_to_do:
            stats = pycoco_eval_scores(hypothesis, references, metric).describe()[stat_track]
            stats = pd.concat([pd.Series({'metric': metric.upper()}), stats])
            results.append(stats)
            coco_requested = True
    if coco_requested:
        print('COCO-based-metrics: done')

    ##
    ## Emotional-Alignment
    ##
    if 'emo_alignment' in methods_to_do:
        emo_accuracy, emo_xentopy = emotional_alignment(hypothesis, ref_emotions, text2emo_vocab, txt2emo_clf, device)
        stats = pd.Series(emo_accuracy, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Emo-Alignment-ACC'}), stats])
        results.append(stats)

        stats = pd.Series(emo_xentopy, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Emo-Alignment-XENT'}), stats])
        results.append(stats)
        print('EMO-ALIGN: done')

    ##
    ## Metaphor-like expressions
    ##
    if 'metaphor' in methods_to_do:
        met_mask = makes_metaphor_via_substring_matching(hypothesis)
        stats = pd.Series(met_mask, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Metaphors'}), stats])
        results.append(stats)
        print('Metaphor-like expressions: Done')

    ##
    ## Novelty via Longest Common Subsequence
    ##
    if 'lcs' in methods_to_do:
        np.random.seed(random_seed) # since you will (normally) sub-sample
        train_utters_tokenized = [u.split() for u in train_utterances]
        uts = pd.Series(train_utters_tokenized).sample(lcs_sample[0]).to_list()
        hypo_token = hypothesis.apply(lambda x: x.split()).sample(lcs_sample[1]).to_list()

        max_lcs, mean_lcs, _ = captions_lcs_from_training_utterances(hypo_token, uts)
        stats = pd.Series(max_lcs).describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'max-LCS'}), stats])
        results.append(stats)
        stats = pd.Series(mean_lcs).describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'mean-LCS'}), stats])
        results.append(stats)
        print('Novelty via Longest Common Subsequence: Done')

    return results