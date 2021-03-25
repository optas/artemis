#!/usr/bin/env python
# coding: utf-8

"""
Training a neural-speaker.

The MIT License (MIT)
Originally created at 6/16/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import time
import numpy as np
import os.path as osp
from torch import nn
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

from artemis.utils.vocabulary import Vocabulary
from artemis.neural_models.word_embeddings import init_token_bias
from artemis.neural_models.attentive_decoder import single_epoch_train, negative_log_likelihood
from artemis.in_out.neural_net_oriented import save_state_dicts, load_state_dicts
from artemis.in_out.basics import create_dir, create_logger
from artemis.in_out.arguments import parse_train_speaker_arguments
from artemis.in_out.neural_net_oriented import df_to_pytorch_dataset, read_preprocessed_data_df, seed_torch_code
from artemis.neural_models.show_attend_tell import describe_model



if __name__ == '__main__':
    args = parse_train_speaker_arguments(save_args=True)

    if args.random_seed != -1:
        seed_torch_code(args.random_seed)

    ## Load/Prepare data
    vocab = Vocabulary.load(osp.join(args.data_dir, 'vocabulary.pkl'))
    print('Using a vocabulary of size', len(vocab))
    df = read_preprocessed_data_df(args, verbose=True)

    if args.debug:
        print(colored('**DEBUGGING** sub-sampling dataset.', 'red'))
        df = df.sample(2500, replace=False)
        df.reset_index(drop=True, inplace=True)

    data_loaders, _ = df_to_pytorch_dataset(df, args)
    print('Will use {} annotations for training.'.format(len(data_loaders['train'].dataset)))

    ## Prepare model
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(args.gpu))
    model = describe_model(vocab=vocab, args=args)
    token_bias = init_token_bias(data_loaders['train'].dataset.tokens, vocab)
    model.decoder.next_word.bias = token_bias

    if args.resume_path:
        print('FT in the most aggressive way. Just let the speaker continue training...')
        loaded_epoch = load_state_dicts(args.resume_path, map_location='cpu', model=model)
        print('Loaded a pre-trained model at epoch {}.'.format(loaded_epoch))

    model.to(device)

    ## Prepare Loss/Optimization
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
        {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.5,
                                                              patience=args.lr_patience,
                                                              verbose=True,
                                                              min_lr=5e-7)

    # Misc.
    best_epoch = -1
    best_val_nll = np.Inf
    print_freq = 10
    start_training_epoch = 1
    no_improvement = 0
    tb_writer = SummaryWriter(create_dir(osp.join(args.log_dir, 'tb_log')))
    model_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
    logger = create_logger(args.log_dir)
    train_args = dict()
    train_args['use_emotion'] = args.use_emo_grounding
    train_args['alpha_c'] = args.atn_cover_img_alpha

    ## Train.
    logger.info('Starting the training of the speaker.')
    for epoch in range(start_training_epoch, args.max_train_epochs + 1):
        start_time = time.time()
        epoch_loss = single_epoch_train(data_loaders['train'], model, criterion, optimizer, epoch, device,
                                        print_freq=print_freq, tb_writer=tb_writer, **train_args)

        logger.info('Epoch loss {:.3f} time {:.1f}'.format(epoch_loss, (time.time() - start_time) / 60))

        val_nll = negative_log_likelihood(model, data_loaders['val'], device)
        logger.info('Validation loss {:.3f}'.format(val_nll))
        lr_scheduler.step(val_nll)

        if val_nll < best_val_nll:
            logger.info('Validation loss, *improved* @epoch {}'.format(epoch))
            best_val_nll = val_nll
            best_epoch = epoch
            out_name = osp.join(model_dir,  'best_model.pt')
            save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
            no_improvement = 0
        else:
            logger.info('Validation loss did NOT improve @epoch {}'.format(epoch))
            no_improvement += 1

        if args.save_each_epoch:
            out_name = osp.join(model_dir, 'model_epoch_' + str(epoch) + '.pt')
            save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        tb_writer.add_scalar('training-entropy-per-epoch', epoch_loss, epoch)
        tb_writer.add_scalar('test-nnl-per-epoch', val_nll, epoch)
        tb_writer.add_scalar('encoder-learning-rate-per-epoch', optimizer.param_groups[0]['lr'], epoch)
        tb_writer.add_scalar('decoder-learning-rate-per-epoch', optimizer.param_groups[1]['lr'], epoch)

        if no_improvement == args.train_patience:
            logger.warning('Stopping the training @epoch-{} due to lack of progress in '
                           'validation-reduction (patience hit {} '
                           'epochs'.format(epoch, args.train_patience))
            break

    with open(osp.join(model_dir, 'final_result.txt'), 'w') as f_out:
        msg = ('Best Validation NLL: {:.4f} (achieved @epoch {})'.format(best_val_nll, best_epoch))
        f_out.write(msg)

    logger.info('Finished training properly.')
    tb_writer.close()