"""
Given an utterance (an optionally an image) guess a distribution over the emotion labels.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torch.nn.functional as F
from torch import nn
from tqdm.notebook import tqdm as tqdm_notebook

from ..utils.stats import AverageMeter


class TextEmotionClassifier(nn.Module):
    def __init__(self, text_encoder, clf_head, img_encoder=None):
        super(TextEmotionClassifier, self).__init__()
        self.text_encoder = text_encoder
        self.clf_head = clf_head
        self.img_encoder = img_encoder

    def __call__(self, text, img=None):
        if img is not None:
            img_feat = self.img_encoder(img)
            feat = self.text_encoder(text, img_feat)
        else:
            feat = self.text_encoder(text)

        logits = self.clf_head(feat)
        return logits


def single_epoch_train(model, data_loader, use_vision, criterion, optimizer, device):
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    model.train()
    for batch in tqdm_notebook(data_loader):
        labels = batch['emotion'].to(device)
        tokens = batch['tokens'].to(device)

        if use_vision:
            img = batch['image'].to(device)
            logits = model(tokens, img)
        else:
            logits = model(tokens)

        # Calculate loss
        loss = criterion(logits, labels)
        acc = torch.mean((logits.argmax(1) == labels).double())

        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b_size = len(labels)
        epoch_loss.update(loss.item(), b_size)
        epoch_acc.update(acc.item(), b_size)
    return epoch_loss.avg, epoch_acc.avg


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, use_vision, criterion, device, detailed=True):
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    model.eval()
    epoch_confidence = []
    for batch in tqdm_notebook(data_loader):
        labels = batch['emotion'].to(device)
        tokens = batch['tokens'].to(device)
        if use_vision:
            img = batch['image'].to(device)
            logits = model(tokens, img)
        else:
            logits = model(tokens)

        # Calculate loss
        loss = criterion(logits, labels)
        guessed_correct = logits.argmax(1) == labels
        acc = torch.mean(guessed_correct.double())

        if detailed:
            epoch_confidence.append(F.softmax(logits, dim=-1).cpu())

        b_size = len(labels)
        epoch_loss.update(loss.item(), b_size)
        epoch_acc.update(acc.item(), b_size)

    if detailed:
        epoch_confidence = torch.cat(epoch_confidence).numpy()

    return epoch_loss.avg, epoch_acc.avg, epoch_confidence