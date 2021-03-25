"""
The MIT License (MIT)
Originally in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from ..evaluation.emotion_alignment import image_to_emotion
from ..emotions import emotion_to_int


class AffectiveCaptionDataset(Dataset):
    """ Basically, an image, with a caption, and an indicated emotion.
    """
    def __init__(self, image_files, tokens, emotions, n_emotions=9, img_transform=None, one_hot_emo=True):
        super(AffectiveCaptionDataset, self).__init__()
        self.image_files = image_files
        self.tokens = tokens
        self.emotions = emotions
        self.n_emotions = n_emotions
        self.img_transform = img_transform
        self.one_hot_emo = one_hot_emo

    def __getitem__(self, index):
        text = np.array(self.tokens[index]).astype(dtype=np.long)

        if self.image_files is not None:
            img = Image.open(self.image_files[index])

            if img.mode is not 'RGB':
                img = img.convert('RGB')

            if self.img_transform is not None:
                img = self.img_transform(img)
        else:
            img = []

        if self.n_emotions > 0:
            if self.one_hot_emo:
                emotion = np.zeros(self.n_emotions, dtype=np.float32)
                emotion[self.emotions[index]] = 1
            else:
                emotion = self.emotions[index]
        else:
            emotion = []

        res = {'image': img, 'emotion': emotion, 'tokens': text, 'index': index}
        return res

    def __len__(self):
        return len(self.tokens)


class ImageClassificationDataset(Dataset):
    def __init__(self, image_files, labels=None, img_transform=None, rgb_only=True):
        super(ImageClassificationDataset, self).__init__()
        self.image_files = image_files
        self.labels = labels
        self.img_transform = img_transform
        self.rgb_only = rgb_only

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])

        if self.rgb_only and img.mode is not 'RGB':
            img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)

        label = []
        if self.labels is not None:
            label = self.labels[index]

        res = {'image': img, 'label': label, 'index': index}
        return res

    def __len__(self):
        return len(self.image_files)


def sub_sample_dataloader(dataloader, sample_size, seed=None, shuffle=False):
    """ Given any torch dataloader create a sub-sampled version of it.
    :param dataloader:
    :param sample_size:
    :param seed:
    :param shuffle:
    :return: dataloader of Subset
    """

    dataset = dataloader.dataset
    n_total = len(dataset)

    if sample_size > n_total:
        raise ValueError

    if seed is not None:
        torch.manual_seed(seed)

    sb_dataset = torch.utils.data.random_split(dataset, [sample_size, n_total-sample_size])[0]
    bsize = min(dataloader.batch_size, sample_size)
    sample_loader = torch.utils.data.DataLoader(dataset=sb_dataset,
                                                batch_size=bsize,
                                                shuffle=shuffle,
                                                num_workers=dataloader.num_workers)
    return sample_loader



def sub_index_affective_dataloader(affective_dataloader, indices, shuffle=False):
    """ Given a torch dataloader and a sequence of integers; extract the corresponding items of the
    carried dataset on the specific indices and make a new dataloader with them.
    :param affective_dataloader: torch.utils.data.DataLoader for AffectiveCaptionDataset
    :param indices: sequence of integers indexing the underlying dataset (dataframe).
    :param shuffle: shuffle the data of the resulting dataloader
    :return: dataloader of AffectiveCaptionDataset
    """
    dataset = affective_dataloader.dataset
    r_img_files = dataset.image_files.iloc[indices].copy()
    r_tokens = dataset.tokens.iloc[indices].copy()
    r_emotions = dataset.emotions.iloc[indices].copy()

    r_img_files.reset_index(inplace=True, drop=True)
    r_tokens.reset_index(inplace=True, drop=True)
    r_emotions.reset_index(inplace=True, drop=True)

    r_dset = AffectiveCaptionDataset(image_files=r_img_files, tokens=r_tokens,
                                    emotions=r_emotions, img_transform=dataset.img_transform)

    batch_size = min(len(indices), affective_dataloader.batch_size)

    r_loader = torch.utils.data.DataLoader(r_dset,
                                           shuffle=shuffle,
                                           batch_size=batch_size,
                                           num_workers=affective_dataloader.num_workers)
    return r_loader


def group_annotations_per_image(affective_dataset):
    """ Group the annotations per image.
    :param affective_dataset: an AffectiveCaptionDataset
    :return: for each image its tokens/emotions as pandas Dataframes
    """
    df = pd.concat([affective_dataset.image_files, affective_dataset.tokens, affective_dataset.emotions], axis=1)
    tokens_grouped = df.groupby('image_files')['tokens_encoded'].apply(list).reset_index(name='tokens_encoded')
    emotion_grouped = df.groupby('image_files')['emotion_label'].apply(list).reset_index(name='emotion')
    assert all(tokens_grouped['image_files'] ==  emotion_grouped['image_files'])
    return tokens_grouped['image_files'], tokens_grouped, emotion_grouped


def default_grounding_dataset_from_affective_loader(loader, img2emo_clf=None, device=None, n_workers=None):
    """
    Convenience function. Given a loader carrying an affective dataset, make a new loader only w.r.t.
    unique images of the dataset, & optionally add to each image the emotion predicted by the img2emo_clf.
    The new loader can be used to sample utterances over the unique images.
    :param loader:
    :param img2emo_clf:
    :param device:
    :return:
    """
    affective_dataset = loader.dataset
    img_files, tokens, emotions = group_annotations_per_image(affective_dataset)

    img_trans = affective_dataset.img_transform
    batch_size = loader.batch_size

    if n_workers is None:
        n_workers = loader.num_workers

    dummy = pd.Series(np.ones(len(img_files), dtype=int) * -1)

    # possibly predict grounding emotions
    if img2emo_clf is not None:
        temp_dataset = ImageClassificationDataset(image_files=img_files,
                                                  img_transform=img_trans)
        img_dataloader = DataLoader(temp_dataset, batch_size, num_workers=n_workers)
        emo_pred_distribution = image_to_emotion(img2emo_clf, img_dataloader, device)

        grounding_emo = pd.Series(emo_pred_distribution.argmax(-1).tolist())  # use maximizer of emotions.
    else:
        grounding_emo = dummy

    new_dataset = AffectiveCaptionDataset(img_files, tokens=dummy, emotions=grounding_emo,
                                          img_transform=img_trans)

    new_loader = DataLoader(dataset=new_dataset, batch_size=batch_size, num_workers=n_workers)
    return new_loader


def custom_grounding_dataset_similar_to_affective_loader(grounding_data_csv, loader, n_workers=None):
    """
    Convenience function. Given a csv indicating (grounding) images on the hard-drive and a loader carrying an affective
    dataset, make a new loader with the csv images using the same configuration (e.g., img_transform) as the loader.
    :param grounding_data_csv: (csv filename)
        - has to have one column named "image_file" that corresponds to the file-names of the images.
        - (optionally) can have also a "grounding_emotion" column with values like "contentment"
    :param loader:
    :return:
    """
    df = pd.read_csv(grounding_data_csv)
    image_files = df['image_file']
    dummy = pd.Series(np.ones(len(image_files), dtype=int) * -1)
    if 'grounding_emotion' in df.columns:
        emotions = df.emotion.apply(emotion_to_int)
    else:
        emotions = dummy

    standard_dset = loader.dataset
    custom_dataset = AffectiveCaptionDataset(image_files, dummy, emotions=emotions,
                                             n_emotions=standard_dset.n_emotions,
                                             img_transform=standard_dset.img_transform,
                                             one_hot_emo=standard_dset.one_hot_emo)
    if n_workers is None:
        n_workers = loader.num_workers

    custom_data_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                     batch_size=min(loader.batch_size, len(custom_dataset)),
                                                     num_workers=n_workers)
    return custom_data_loader

