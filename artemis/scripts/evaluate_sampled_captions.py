#!/usr/bin/env python
# coding: utf-8


import torch
import argparse
import pandas as pd
import pprint

from tqdm.notebook import tqdm

from speakers_listeners.utils.vocabulary import Vocabulary
from speakers_listeners.utils.in_out import str2bool

from ela.in_out.basics import unpickle_data, pickle_data
from ela.in_out.models import torch_load_model
from ela.evaluation.sentence_sim import Sentence2SentenceEvaluator
from ela.evaluation.single_caption_per_image import apply_all_basic_evaluations
from ela.evaluation.single_caption_per_image import apply_all_fancy_evaluations


parser = argparse.ArgumentParser(description='evaluation of synthetic captions')
parser.add_argument('-references-pkl-file', type=str, required=True)
parser.add_argument('-captions-pkl-file', type=str, required=True)
parser.add_argument('-text2emo-path', type=str, required=True, help='path to neural-net')
parser.add_argument('-vocab-path', type=str, required=True, help='vobab file path')
parser.add_argument('-save-file', type=str)
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--mask-file', type=str, help='constrain the captions/ground-truth')
parser.add_argument('--train-bert-embeddings', type=str)
parser.add_argument('--lcs-sample-size', type=int, nargs=2, default=[5000, 500])
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--debug', type=str2bool, default=False)

args = parser.parse_args()
args_string = pprint.pformat(vars(args))
print(args_string)

device = torch.device("cuda:" + args.gpu_id)
vocab = Vocabulary.load(args.vocab_path)

caption_data = next(unpickle_data(args.captions_pkl_file))
gt_data = next(unpickle_data(args.references_pkl_file))
gt_data = gt_data[args.split]

if args.mask_file:
    mask = next(unpickle_data(args.mask_file))
    print('Using a mask to keep {} of data'.format(mask.mean().round(4)))
else:
    mask = pd.Series([True]*len(gt_data))

if args.debug:
    print('***Debugging***')
    gt_data = gt_data.iloc[0:100]
    args.lcs_sample_size = [10, 2]

#Prepare BERT-NLI evaluator by embedding the references
semantic_dist_eval = Sentence2SentenceEvaluator(references=gt_data.references_pre_vocab[mask])

#Prepare Emotion-Alignment evaluation
txt2em_clf = torch_load_model(args.text2emo_path, map_location=device)

# Load BERT embeddings and/for unique train-utterances
unique_train_utters, train_bert_emb = unpickle_data(args.train_bert_embeddings)


results = []

for config_i, captions_i in tqdm(caption_data):
    if args.debug:
        captions_i = captions_i.iloc[0:100]

    merged = pd.merge(gt_data, captions_i) # this ensures proper order of captions to gt (via accessing merged.captions)
    merged = merged[mask].reset_index(drop=True)
    
    ## TODO IF results do not make sense check effect of merged new index HERE
    if sum(mask) == len(mask): # no issue with index
        assert all(merged.references_pre_vocab == gt_data.references_pre_vocab[mask])
    
    hypothesis = merged.caption
    references = merged.references
    ref_emotions = merged.emotion
    if len(results) == 0:
        print('|Masked Captions| Size:', len(hypothesis))
        
    basic_eval_res=\
        apply_all_basic_evaluations(hypothesis, references, ref_emotions, semantic_dist_eval, vocab, txt2em_clf, device)
    
    fancy_eval_res=\
        apply_all_fancy_evaluations(hypothesis, semantic_dist_eval, unique_train_utters, train_bert_emb, args.lcs_sample_size)

    eval_res = basic_eval_res + fancy_eval_res
    results.append([config_i, pd.DataFrame(eval_res)])

    if args.debug:
        if len(results) == 2:
            break

pickle_data(args.save_file, results)