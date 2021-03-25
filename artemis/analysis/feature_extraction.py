"""
Routines to extract features from images.

The MIT License (MIT)
Originally created at 6/14/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision import models

from ..in_out.datasets import ImageClassificationDataset
from ..in_out.neural_net_oriented import image_net_mean, image_net_std
from ..neural_models.resnet_encoder import ResnetEncoder


@torch.no_grad()
def get_forward_features_of_dataset(encoder, dataloader, device, data_in_batch='image'):
    b_size = dataloader.batch_size
    for i, batch in enumerate(dataloader):
        feats = encoder(batch[data_in_batch].to(device))
        feats = feats.cpu().numpy().astype('float32')

        if i == 0:
            features = np.zeros((len(dataloader.dataset), feats.shape[1]), dtype='float32')

        if i < len(dataloader) - 1:
            features[i * b_size: (i + 1) * b_size] = feats
        else:
            # special treatment for final batch
            features[i * b_size:] = feats
    return features


def image_transformation(img_dim, pretraining='image_net'):
    if pretraining == 'image_net':
        normalize = transforms.Normalize(mean=image_net_mean, std=image_net_std)
    else:
        raise NotImplementedError('')

    res = transforms.Compose([transforms.Resize((img_dim, img_dim), Image.LANCZOS),
                              transforms.ToTensor(), normalize])

    return res


def vgg_encoder(device):
    vgg = models.vgg16_bn(pretrained=True).to(device).eval()
    feature_storage = []
    def hook(module, hook_input, hook_output):
        feature_storage.append(hook_output.detach_().cpu().numpy())
    vgg.classifier[4].register_forward_hook(hook) # last relu layer before classification.
    return vgg, feature_storage


@torch.no_grad()
def extract_visual_features(image_files, img_dim, method='resnet18',
                            batch_size=128, n_workers=12, device='cuda'):


    img_transform = image_transformation(img_dim)
    dataset = ImageClassificationDataset(image_files, img_transform=img_transform)

    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=n_workers)

    if method.startswith('resnet'):
        vis_encoder = ResnetEncoder(method, 1).to(device).eval()
        features = get_forward_features_of_dataset(vis_encoder, loader, device)

    elif method.startswith('vgg'):
        vis_encoder, features = vgg_encoder(device)
        for batch in loader:
            vis_encoder(batch['image'].to(device))
        features = np.vstack(features)

    elif method.startswith('random'):
        vis_encoder = ResnetEncoder('resnet18', 1, pretrained=False).to(device).eval()
        features = get_forward_features_of_dataset(vis_encoder, loader, device)

    return features