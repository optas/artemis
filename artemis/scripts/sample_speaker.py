#!/usr/bin/env python
# coding: utf-8

"""
Load a trained speaker and images/data to create (sample) captions for them.

The MIT License (MIT)
Originally created at 10/3/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""


import torch
import json
import numpy as np

from artemis.in_out.basics import pickle_data
from artemis.in_out.arguments import parse_test_speaker_arguments
from artemis.in_out.neural_net_oriented import torch_load_model, load_saved_speaker, seed_torch_code
from artemis.neural_models.attentive_decoder import negative_log_likelihood
from artemis.captioning.sample_captions import versatile_caption_sampler, captions_as_dataframe
from artemis.in_out.datasets import sub_index_affective_dataloader
from artemis.in_out.datasets import default_grounding_dataset_from_affective_loader
from artemis.in_out.datasets import custom_grounding_dataset_similar_to_affective_loader


if __name__ == '__main__':
    args = parse_test_speaker_arguments()

    # Load pretrained speaker & its corresponding train-val-test data. If you do not provide a
    # custom set of images to annotate. Then based on the -split you designated it will annotate this data.
    speaker, epoch, data_loaders = load_saved_speaker(args.speaker_saved_args, args.speaker_checkpoint,
                                                      with_data=True, verbose=True)
    device = torch.device("cuda:" + args.gpu)
    speaker = speaker.to(device)
    eos = speaker.decoder.vocab.eos
    working_data_loader = data_loaders[args.split]

    if args.max_utterance_len is None:
        # use the maximum length in the underlying split.
        def utterance_len(tokens, eos=eos):
            return np.where(np.asarray(tokens) == eos)[0][0] -1 # -1 to remove sos
        args.max_utterance_len = working_data_loader.dataset.tokens.apply(utterance_len).max()

    use_custom_dataset = False
    if args.custom_data_csv is not None:
        use_custom_dataset = True

    if args.compute_nll and not use_custom_dataset:
        print('Computing Negative Log Likelihood of ground-truth annotations:')
        nll = negative_log_likelihood(speaker, working_data_loader, device)
        print('{} NLL: {}'.format(args.split, nll))

    img2emo_clf = None
    if args.img2emo_checkpoint:
        img2emo_clf = torch_load_model(args.img2emo_checkpoint, map_location=device)

    if use_custom_dataset:
        annotate_loader = custom_grounding_dataset_similar_to_affective_loader(args.custom_data_csv,
                                                                               working_data_loader, args.n_workers)
    else:
        # removes duplicate images and optionally uses img2emo_clf to create a grounding emotion.
        annotate_loader = default_grounding_dataset_from_affective_loader(working_data_loader, img2emo_clf,
                                                                          device, args.n_workers)

    if args.subsample_data != -1:
        sids = np.random.choice(len(annotate_loader.dataset.image_files), args.subsample_data)
        annotate_loader = sub_index_affective_dataloader(annotate_loader, sids)

    with open(args.sampling_config_file) as fin:
        sampling_configs = json.load(fin)

    print('Loaded {} sampling configurations to try.'.format(len(sampling_configs)))
    optional_params = ['max_utterance_len', 'drop_unk', 'drop_bigrams']  # if you did not specify them in the sampling-config
                                                                         # those from the argparse will be used
    final_results = []
    for config in sampling_configs:
        for param in optional_params:
            if param not in config:
                config[param] = args.__getattribute__(param)
        print('Sampling with configuration: ', config)

        if args.random_seed != -1:
            seed_torch_code(args.random_seed)

        captions_predicted, attn_weights = versatile_caption_sampler(speaker, annotate_loader, device, **config)
        df = captions_as_dataframe(annotate_loader.dataset, captions_predicted, wiki_art_data=not use_custom_dataset)
        final_results.append([config, df, attn_weights])
        print('Done.')

    pickle_data(args.out_file, final_results)
