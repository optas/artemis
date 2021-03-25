"""
Rensnet Wrapper.

The MIT License (MIT)
Originally created in late 2019, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""


import torch
from torch import nn
from torchvision import models


class ResnetEncoder(nn.Module):
    """Convenience wrapper around resnet models"""
    def __init__(self, backbone, adapt_image_size=None, drop=2, pretrained=True, verbose=False):
        """
        :param backbone: (string) resnet-S, S in [18, 34, 50, 101]
        :param adapt_image_size: (opt, int) if given forward feature has
            [B, adapt_image_size, adapt_image_size, feat-dim]
        :param drop: how many of the last layers/blocks to drop.
        :param pretrained: (Boolean)
        :param verbose: (opt, Boolean) if true print actions taken.
        Note: in total there are 10 layers/blocks. The last two are an adaptive_pooling and an FC, the
        previous layers give rise to convolutional maps of increasing spatial size.
        """

        if drop == 0 and adapt_image_size is not None:
            raise ValueError('Trying to apply adaptive pooling while keeping the entire model (drop=0).')

        super(ResnetEncoder, self).__init__()
        backbones = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
        }

        self.name = backbone
        self.drop = drop
        self.resnet = backbones[self.name](pretrained=pretrained)

        # Remove linear and last adaptive pool layer
        if drop > 0:
            modules = list(self.resnet.children())
            if verbose:
                print('Removing the last {} layers of a {}'.format(drop, self.name))
                print(modules[-drop:])
            modules = modules[:-drop]
            self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = None
        if adapt_image_size is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((adapt_image_size, adapt_image_size))

        if pretrained:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def __call__(self, images):
        """Forward prop.
            :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
            :return: encoded images
        """
        out = self.resnet(images) # (B, F, ceil(image_size/32), ceil(image_size/32))

        if self.adaptive_pool is not None:
            out = self.adaptive_pool(out)  # (B, F, adapt_image_size, adapt_image_size)

        if self.drop > 0: # convolutional-like output
            out = out.permute(0, 2, 3, 1)      # bring feature-dim last.
            out = torch.squeeze(torch.squeeze(out, 1), 1)  # In case adapt_image_size == 1, remove dimensions
        return out

    def unfreeze(self, level=5, verbose=False):
        """Allow or prevent the computation of gradients for blocks after level.
        The smaller the level, the less pretrained the resnet will be.
        """
        all_layers = list(self.resnet.children())

        if verbose:
            ll = len(all_layers)
            print('From {} layers, you are unfreezing the last {}'.format(ll, ll-level))

        for c in all_layers[level:]:
            for p in c.parameters():
                p.requires_grad = True
        return self

    def embedding_dimension(self):
        """The feature (channel) dimension of the last layer"""
        if self.drop == 0:
            return 1000  #Imagenet Classes

        if self.drop == 2:
            return 512 if int(self.name.replace('resnet', '')) < 50 else 2048

        if self.drop == 3:
            return 256 if int(self.name.replace('resnet', '')) < 50 else 1024

        raise NotImplementedError

