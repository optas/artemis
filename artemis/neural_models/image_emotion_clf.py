"""
Given an image guess a distribution over the emotion labels.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.notebook import tqdm as tqdm_notebook

from ..utils.stats import AverageMeter


class ImageEmotionClassifier(nn.Module):
    def __init__(self, img_encoder, clf_head):
        super(ImageEmotionClassifier, self).__init__()
        self.img_encoder = img_encoder
        self.clf_head = clf_head

    def __call__(self, img):
        feat = self.img_encoder(img)
        logits = self.clf_head(feat)
        return logits


def single_epoch_train(model, data_loader, criterion, optimizer, device):
    epoch_loss = AverageMeter()
    model.train()
    for batch in tqdm_notebook(data_loader):
        img = batch['image'].to(device)
        labels = batch['label'].to(device) # emotion_distribution
        logits = model(img)

        # Calculate loss
        loss = criterion(logits, labels)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b_size = len(labels)
        epoch_loss.update(loss.item(), b_size)
    return epoch_loss.avg


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criterion, device, detailed=True, kl_div=True):
    epoch_loss = AverageMeter()
    model.eval()
    epoch_confidence = []
    for batch in tqdm_notebook(data_loader):
        img = batch['image'].to(device)
        labels = batch['label'].to(device) # emotion_distribution
        logits = model(img)

        # Calculate loss
        loss = criterion(logits, labels)

        if detailed:
            if kl_div:
                epoch_confidence.append(torch.exp(logits).cpu())  # logits are log-soft-max
            else:
                epoch_confidence.append(F.softmax(logits, dim=-1).cpu()) # logits are pure logits

        b_size = len(labels)
        epoch_loss.update(loss.item(), b_size)

    if detailed:
        epoch_confidence = torch.cat(epoch_confidence).numpy()

    return epoch_loss.avg, epoch_confidence