"""
I/O routines directly related to torch-based neural-models & their (training etc.) dataset processing.

The MIT License (MIT)
Originally created at 10/2/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import random
import warnings
import numpy as np
import pandas as pd
import os.path as osp
import multiprocessing as mp
import torchvision.transforms as transforms

from ast import literal_eval
from PIL import Image

from .basics import read_saved_args
from .datasets import AffectiveCaptionDataset, ImageClassificationDataset
from ..utils.vocabulary import Vocabulary
from ..neural_models.show_attend_tell import describe_model as describe_sat


image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]


def max_io_workers():
    """return all/max possible available cpus of machine."""
    return max(mp.cpu_count() - 1, 1)


def image_transformation(img_dim, lanczos=True):
    """simple transformation/pre-processing of image data."""

    if lanczos:
        resample_method = Image.LANCZOS
    else:
        resample_method = Image.BILINEAR

    normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    img_transforms = dict()
    img_transforms['train'] = transforms.Compose([transforms.Resize((img_dim, img_dim), resample_method),
                                                  transforms.ToTensor(),
                                                  normalize])

    # Use same transformations as in train (since no data-augmentation is applied in train)
    img_transforms['test'] = img_transforms['train']
    img_transforms['val'] = img_transforms['train']
    img_transforms['rest'] = img_transforms['train']
    return img_transforms


def df_to_pytorch_dataset(df, args):
    if args.num_workers == -1:
        n_workers = max_io_workers()
    else:
        n_workers = args.num_workers

    load_imgs = True
    if hasattr(args, 'use_imgs') and not args.use_imgs: # build a dataset without images (e.g., text/emotion only)
        load_imgs = False

    one_hot_emo = True
    if hasattr(args, 'one_hot_emo') and not args.one_hot_emo: # turn off the one-hot, keep the integer (e.g., when a using xentropy)
        one_hot_emo = False

    img_transforms = None
    if load_imgs:
        img_transforms = image_transformation(args.img_dim, lanczos=args.lanczos)

    if args.dataset == 'artemis':
        datasets = pass_artemis_splits_to_datasets(df, load_imgs, img_transforms, args.img_dir, one_hot_emo=one_hot_emo)
    elif args.dataset == 'ola': # Objective Language for Art.
        datasets = pass_artemis_splits_to_datasets(df, load_imgs, img_transforms, args.img_dir, n_emotions=0)
    elif args.dataset == 'coco':
        datasets = pass_coco_splits_to_datasets(df, load_imgs, img_transforms)
    else:
        raise ValueError('training dataset not recognized.')

    dataloaders = dict()
    for split in datasets:
        b_size = args.batch_size if split=='train' else args.batch_size * 2
        dataloaders[split] = torch.utils.data.DataLoader(dataset=datasets[split],
                                                         batch_size=b_size,
                                                         shuffle=split=='train',
                                                         num_workers=n_workers)
    return dataloaders, datasets


def pass_coco_splits_to_datasets(df, load_imgs, img_transforms, n_emotions=0):
    datasets = dict()
    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True) # so that direct ([]) indexing in get_item works
        img_files = None
        img_trans = None

        if load_imgs:
            img_files = g['image_files']
            img_trans = img_transforms[split]

        dataset = AffectiveCaptionDataset(img_files, g.tokens_encoded, g.emotion_label, img_transform=img_trans,
                                          n_emotions=n_emotions)
        datasets[split] = dataset
    return datasets


def pass_artemis_splits_to_datasets(df, load_imgs, img_transforms, top_img_dir, n_emotions=9, one_hot_emo=True):
    datasets = dict()
    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True) # so that direct ([]) indexing in get_item works
        img_files = None
        img_trans = None

        if load_imgs:
            img_files = g.apply(lambda x : osp.join(top_img_dir, x.art_style,  x.painting + '.jpg'), axis=1)
            img_files.name = 'image_files'
            img_trans = img_transforms[split]

        dataset = AffectiveCaptionDataset(img_files, g.tokens_encoded, g.emotion_label, n_emotions=n_emotions,
                                          img_transform=img_trans, one_hot_emo=one_hot_emo)

        datasets[split] = dataset
    return datasets


def read_preprocessed_data_df(args, verbose=False):
    if args.dataset == 'artemis':
        file_name = 'artemis_preprocessed.csv'
    elif args.dataset == 'coco':
        file_name = 'coco_preprocessed.csv'
    else:
        raise ValueError('Unknown Dataset.')

    if hasattr(args, 'fine_tune_data') and  args.fine_tune_data:
        df = pd.read_csv(args.fine_tune_data) # allow explicit data passing
    else:
        df = pd.read_csv(osp.join(args.data_dir, file_name))

    df.tokens_encoded = df.tokens_encoded.apply(literal_eval)

    if verbose:
        print('Loaded {} utterances'.format(len(df)))
    return df


def image_emotion_distribution_df_to_pytorch_dataset(df, args, drop_thres=None):
    """ Convert the pandas dataframe that carries information about images and emotion (distributions) to a
    dataset that is amenable to deep-learning (e.g., for an image2emotion classifier).
    :param df:
    :param args:
    :param drop_thres: (optional, float) if provided each distribution of the training will only consist of examples
        for which the maximizing emotion aggregates more than this (drop_thres) mass.
    :return: pytorch dataloaders & datasets
    """
    dataloaders = dict()
    datasets = dict()
    img_transforms = image_transformation(args.img_dim, lanczos=args.lanczos)

    if args.num_workers == -1:
        n_workers = max_io_workers()
    else:
        n_workers = args.num_workers

    for split, g in df.groupby('split'):
        g.reset_index(inplace=True, drop=True)

        if split == 'train' and drop_thres is not None:
            noise_mask = g['emotion_distribution'].apply(lambda x: max(x) > drop_thres)
            print('Keeping {} of the training data, since for the rest their emotion-maximizer is too low.'.format(noise_mask.mean()))
            g = g[noise_mask]
            g.reset_index(inplace=True, drop=True)


        img_files = g.apply(lambda x : osp.join(args.img_dir, x.art_style,  x.painting + '.jpg'), axis=1)
        img_files.name = 'image_files'

        dataset = ImageClassificationDataset(img_files, g.emotion_distribution,
                                             img_transform=img_transforms[split])

        datasets[split] = dataset
        b_size = args.batch_size if split=='train' else args.batch_size * 2
        dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                         batch_size=b_size,
                                                         shuffle=split=='train',
                                                         num_workers=n_workers)
    return dataloaders, datasets


def seed_torch_code(seed, strict=False):
    """Control pseudo-randomness for reproducibility.
    :param manual_seed: (int) random-seed
    :param strict: (boolean) if True, cudnn operates in a deterministic manner
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_state_dicts(checkpoint_file, epoch=None, **kwargs):
    """ Save torch items with a state_dict
    """
    checkpoint = dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    for key, value in kwargs.items():
        checkpoint[key] = value.state_dict()

    torch.save(checkpoint, checkpoint_file)


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    """ Load torch items from saved state_dictionaries
    """
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get('epoch')
    if epoch:
        return epoch


def torch_save_model(model, path):
    """ Wrap torch.save to catch standard warning of not finding the nested implementations.
    :param model:
    :param path:
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.save(model, path)


def torch_load_model(checkpoint_file, map_location=None):
    """ Wrap torch.load to catch standard warning of not finding the nested implementations.
    :param checkpoint_file:
    :param map_location:
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(checkpoint_file, map_location=map_location)
    return model


def load_saved_speaker(args_file, model_ckp, with_data=False, override_args=None, verbose=False):
    """
    :param args_file: saved argparse arguments with model's description (and location of used data)
    :param model_ckp: saved checkpoint with model's parameters.
    :param with_data:
    :param override_args:
    :return:
    Note, the model is loaded and returned in cpu.
    """
    if verbose:
        print('Loading saved speaker trained with parameters:')
    args = read_saved_args(args_file, override_args=override_args, verbose=verbose)

    # Prepare empty model
    vocab = Vocabulary.load(osp.join(args.data_dir, 'vocabulary.pkl'))
    print('Using a vocabulary of size', len(vocab))
    model = describe_sat(vocab, args)

    # Load save weights
    epoch = load_state_dicts(model_ckp, model=model, map_location='cpu')
    print('Loading speaker model at epoch {}.'.format(epoch))

    # Load data
    if with_data:
        df = read_preprocessed_data_df(args, verbose=True)
        data_loaders, _ = df_to_pytorch_dataset(df, args)
    else:
        data_loaders = None

    return model, epoch, data_loaders


def deprocess_img(img, std=None, mean=None, clamp=None, inplace=False):
    if not inplace:
        img = img.clone()

    if img.ndimension() == 4:  # batch of images
        pass
        # single_img = False
    elif img.ndimension() == 3:  # single image
        img = img.view([1] + list(img.shape))
        # single_img = True
    else:
        raise ValueError()

    dtype = img.dtype
    n_channels = img.size(1)

    if std is not None:
        std = torch.as_tensor(std, dtype=dtype, device=img.device)
        img.mul_(std.view([1, n_channels, 1, 1]))

    if mean is not None:
        mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
        img.add_(mean.view([1, n_channels, 1, 1]))

    if clamp is not None:
        img.clamp_(clamp[0], clamp[1])

    return img


def to_img(tensor, mean=None, std=None):
    """ Convert tensor object to PIL.Image(s)
    :param tensor:
    :param mean:
    :param std:
    :return:
    """
    image = tensor.clone().detach()
    image = deprocess_img(image, mean=mean, std=std)
    # Add 0.5 after un-normalizing to [0, 255] to round to nearest integer
    array = image.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    image = []
    for im in array:
        image.append(Image.fromarray(im))
    return image
