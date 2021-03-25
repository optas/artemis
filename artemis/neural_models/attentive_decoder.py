"""
Decoding module for a neural speaker (with attention capabilities).

The MIT License (MIT)
Originally created at 06/15/19, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import random
import time
import warnings
import tqdm
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm_

from .attention import AdditiveVisioLinguistic
from ..utils.stats import AverageMeter


class AttentiveDecoder(nn.Module):
    """
    Note: code adapted from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    implementing a solid version of Show, Attend, and Tell. Many thanks Sagar and the team.

    Special (optional) features:
        - use stochastic teacher forcer
        - add auxiliary input data at each decoding step (besides each 'previous' token).
        - tie the weights of the encoder/decoder weight matrices
    """
    def __init__(self, word_embedding, rnn_hidden_dim, encoder_dim, attention_dim,
                 vocab, dropout_rate=0, tie_weights=False, teacher_forcing_ratio=1,
                 auxiliary_net=None, auxiliary_dim=0):
        """
        :param word_embedding: nn.Embedding
        :param rnn_hidden_dim: hidden (and thus output) dimension of the decoding rnn
        :param encoder_dim: feature dimension of encoded stimulus
        :param attention_dim: feature dimension over which attention is computed
        :param vocab: artemis.utils.vocabulary instance
        :param dropout: dropout rate
        :param tie_weights: (opt, boolean) if True, the hidden-to-word weights are equal (tied) to the word-embeddings,
            see https://arxiv.org/abs/1611.01462 for explanation of why this might be a good idea.
        :param teacher_forcing_ratio:
        :param auxiliary_net: (optional) nn.Module that will be feeding the decoder at each time step
            with some "auxiliary" information (say an emotion label). Obviously, this information is separate than the
            output of the typically used image-encoder.
        :param auxiliary_dim: (int, optional) the output feature-dimension of the auxiliary net.
        """
        super(AttentiveDecoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.word_embedding = word_embedding
        self.auxiliary_net = auxiliary_net
        self.uses_aux_data = False

        if auxiliary_dim > 0:
            self.uses_aux_data = True

        self.decode_step = nn.LSTMCell(word_embedding.embedding_dim + encoder_dim + auxiliary_dim, rnn_hidden_dim)
        self.attention = AdditiveVisioLinguistic(encoder_dim, rnn_hidden_dim, attention_dim)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        else:
            self.dropout = nn.Identity()

        self.init_h = nn.Linear(encoder_dim, rnn_hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, rnn_hidden_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(rnn_hidden_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.next_word = nn.Linear(rnn_hidden_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()
        self.teacher_forcing_ratio = teacher_forcing_ratio

        if tie_weights:
            if self.word_embedding.embedding_dim != rnn_hidden_dim:
                raise ValueError('When using the tied weights')
            print('tying weights of encoder/decoder')
            self.next_word.weight = self.word_embedding.weight

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def init_weights(self, init_range=0.1):
        """ Better initialization """
        self.word_embedding.weight.data.uniform_(-init_range, init_range)  # remove if pre-trained model comes up
        self.next_word.bias.data.zero_()
        self.next_word.weight.data.uniform_(-init_range, init_range)

    def __call__(self, encoder_out, captions, auxiliary_data=None):
        """ Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param auxiliary_data: extra information associated with the images (batch_size, some_dim)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        return self.sort_captions_and_forward(encoder_out, captions, auxiliary_data=auxiliary_data)

    def sort_captions_and_forward(self, encoder_out, captions, auxiliary_data=None):
        """ Feed forward that ...
        :param encoder_out:
        :param captions:
        :return:
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        decode_lengths = torch.where(captions == self.vocab.eos)[1] # "<sos> I am <eos>" => decode_length = 3
                                                                    # we do not feed <eos> as input to generate
                                                                    # something after it

        # Sort input data by decreasing lengths to reduce compute below
        decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        if auxiliary_data is not None:
            auxiliary_data = auxiliary_data[sort_ind]
            auxiliary_data = self.auxiliary_net(auxiliary_data)

        # prepare for unravelling
        embeddings = self.word_embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        decode_lengths = decode_lengths.tolist()
        device = embeddings.device

        # Create tensors to hold word prediction logits and attention maps (alphas)
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h = h[:batch_size_t] # effective h
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            if use_teacher_forcing or t == 0:
                decoder_lang_input = embeddings[:batch_size_t, t]
            else:
                _, top_pred = preds[:batch_size_t].topk(1)
                top_pred = top_pred.squeeze(-1).detach()  # detach from history as input
                decoder_lang_input = self.word_embedding(top_pred)

            if auxiliary_data is not None:
                auxiliary_data_t = auxiliary_data[:batch_size_t]
                decoder_in = torch.cat([decoder_lang_input, attention_weighted_encoding, auxiliary_data_t], dim=1)
            else:
                decoder_in = torch.cat([decoder_lang_input, attention_weighted_encoding], dim=1)

            h, c = self.decode_step(decoder_in, (h, c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.next_word(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t] = preds
            alphas[:batch_size_t, t] = alpha
        return predictions, captions, decode_lengths, alphas, sort_ind

    def attend_and_predict_next_word(self, encoder_out, h, c, tokens, aux_data=None):
        """Given current hidden/memory state of the decoder and the input tokens, guess the next tokens
        and update the hidden/memory states.
        :param encoder_out: the grounding
        :param h: current hidden state
        :param c: current memory state
        :param tokens: current token input to the decoder
        :return: logits over vocabulary distribution, updated h/c
        """
        attention_weighted_encoding, alpha = self.attention(encoder_out, h)
        gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding
        embeddings = self.word_embedding(tokens)  # (batch_size, embed_dim)

        decoder_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)

        if aux_data is not None:
            aux_feat = self.auxiliary_net(aux_data)
            decoder_input = torch.cat([decoder_input, aux_feat], dim=1)

        h, c = self.decode_step(decoder_input, (h, c))  # (batch_size_t, decoder_dim)
        logits = self.next_word(h)  # (batch_size_t, vocab_size)
        return h, c, logits, alpha


def single_epoch_train(train_loader, model, criterion, optimizer, epoch, device, tb_writer=None, **kwargs):
    """ Perform training for one epoch.
    :param train_loader: DataLoader for training data
    :param model: nn.ModuleDict with 'encoder', 'decoder' keys
    :param criterion: loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param device:
    """
    alpha_c = kwargs.get('alpha_c', 1.0)  # Weight of doubly stochastic (attention) regularization.
    grad_clip = kwargs.get('grad_clip', 5.0) # Gradient clipping (norm magnitude)
    print_freq = kwargs.get('print_freq', 100)
    use_emotion = kwargs.get('use_emotion', False)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    entropy_loss_meter = AverageMeter()  # entropy loss (per word decoded)
    total_loss_meter = AverageMeter()
    start = time.time()
    steps_taken = (epoch-1) * len(train_loader.dataset)
    model.train()

    for i, batch in enumerate(train_loader):
        imgs = batch['image'].to(device)
        caps = batch['tokens'].to(device)
        b_size = len(imgs)
        data_time.update(time.time() - start)

        if use_emotion:
            emotion = batch['emotion'].to(device)
            res = model.decoder(model.encoder(imgs), caps, emotion)
        else:
            res = model.decoder(model.encoder(imgs), caps)
        logits, caps_sorted, decode_lengths, alphas, sort_ind = res

        # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
        targets = caps_sorted[:, 1:]

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        logits = pack_padded_sequence(logits, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        ent_loss = criterion(logits.data, targets.data)
        total_loss = ent_loss

        # Add doubly stochastic attention regularization
        # Note. some implementation simply do this like: d_atn_loss = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # here we take care of the fact that some samples in the same batch have more/less tokens than others.
        if alpha_c > 0:
            total_energy = torch.from_numpy(np.array(decode_lengths)) / alphas.shape[-1]   # n_tokens / num_pixels
            total_energy.unsqueeze_(-1)  # B x 1
            total_energy = total_energy.to(device)
            d_atn_loss = alpha_c * ((total_energy - alphas.sum(dim=1)) ** 2).mean()
            total_loss += d_atn_loss

        # Back prop.
        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        entropy_loss_meter.update(ent_loss.item(), sum(decode_lengths))
        total_loss_meter.update(total_loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        steps_taken += b_size

        # Print status
        if print_freq is not None and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss=total_loss_meter))
        if tb_writer is not None:
            tb_writer.add_scalar('training-entropy-loss-with-batch-granularity', entropy_loss_meter.avg, steps_taken)

    return total_loss_meter.avg


@torch.no_grad()
def negative_log_likelihood(model, data_loader, device):
    """
    :param model:
    :param data_loader:
    :param device:
    :param phase:
    :return:
    """
    model.eval()
    nll = AverageMeter()

    aux_data = None
    for batch in data_loader:
        imgs = batch['image'].to(device)
        caps = batch['tokens'].to(device)

        # TODO Refactor
        if model.decoder.uses_aux_data:
            aux_data = batch['emotion'].to(device)

        logits, caps_sorted, decode_lengths, alphas, sort_ind = model.decoder(model.encoder(imgs), caps, aux_data)

        # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
        targets = caps_sorted[:, 1:]

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        logits = pack_padded_sequence(logits, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = F.cross_entropy(logits.data, targets.data)
        nll.update(loss.item(), sum(decode_lengths))
    return nll.avg


@torch.no_grad()
def log_prob_of_caption(model, img, tokens, temperature=1):
    """Given a captioning model, return the log-probability of a caption given an image.
    This version expects a batch of images, each assotiated with a single caption.
    :param model: encoder/decoder speaker
    :param img: Tensor B x channels x spatial-dims
    :param tokens: Tensor B x max-n-tokens
    :return log_probs: Tensor of size B x max-n-tokens holding the log-probs of each token of each caption
    """

    encoder = model.encoder
    decoder = model.decoder

    assert all(tokens[:, 0] == decoder.vocab.sos)

    max_steps = tokens.shape[1]
    encoder_out = encoder(img)
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

    # Create tensors to hold log-probs
    log_probs = torch.zeros(batch_size, max_steps).to(tokens.device)
    h, c = decoder.init_hidden_state(encoder_out)

    for t in range(max_steps - 1):
        h, c, pred_t, _ = decoder.attend_and_predict_next_word(encoder_out, h, c, tokens[:, t])

        if temperature != 1:
            pred_t /= temperature

        pred_t = F.log_softmax(pred_t, dim=1)
        log_probs[:, t] = pred_t[torch.arange(batch_size), tokens[:, t+1]] # prob. of guessing next token

    lens = torch.where(tokens == decoder.vocab.eos)[1] # true tokens + 1 for <eos>
    mask = torch.zeros_like(log_probs)
    mask[torch.arange(mask.shape[0]), lens] = 1
    mask = mask.cumsum(dim=1).to(torch.bool)
    log_probs.masked_fill_(mask, 0) # set to zero all positions after the true size of the caption
    return log_probs, lens


@torch.no_grad()
def sample_captions(model, loader, max_utterance_len, sampling_rule, device, temperature=1,
                    topk=None, drop_unk=True, drop_bigrams=False):
    """
    :param model:
    :param loader:
    :param max_utterance_len: maximum allowed length of captions
    :param sampling_rule: (str) 'argmax' or 'multinomial', or 'topk'
    :return:
        attention_weights: (torch cpu Tensor) N-images x encoded_image_size (e.g., 7 x 7) x  max_utterance_len
            attention_weights[:,0] corresponds to the attention map over the <SOS> symbol
    """
    if sampling_rule not in ['argmax', 'multinomial', 'topk']:
        raise ValueError('Unknown sampling rule.')

    model.eval()
    all_predictions = []
    attention_weights = []
    unk = model.decoder.vocab.unk

    use_aux_data = model.decoder.uses_aux_data
    aux_data = None

    for batch in loader:
        imgs = batch['image'].to(device)

        if use_aux_data:
            aux_data = batch['emotion'].to(device)

        encoder_out = model.encoder(imgs)
        enc_image_size = encoder_out.size(1)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        # Create tensors to hold word predictions
        max_steps = max_utterance_len + 1  # one extra step for EOS marker
        predictions = torch.zeros(batch_size, max_steps).to(device)

        # Initialize decoder state
        decoder = model.decoder
        h, c = decoder.init_hidden_state(encoder_out) # (batch_size, decoder_dim)

        # Tensor to store previous words at each step; now they're just <sos>
        prev_words = torch.LongTensor([decoder.vocab.sos] * batch_size).to(device)

        for t in range(max_steps):
            h, c, pred_t, alpha = decoder.attend_and_predict_next_word(encoder_out, h, c, prev_words, aux_data=aux_data)
            if t > 0: # at t=1 it sees <sos> as the previous word
                alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (bsize, enc_image_size, enc_image_size)
                attention_weights.append(alpha.cpu())

            pred_t /= temperature

            if drop_unk:
                pred_t[:, unk] = -math.inf

            if t > 0:
                pred_t[:, prev_words] = -math.inf # avoid repeating the same word twice

            if t > 1:
                pred_t[:, predictions[:,t-2].long()] = -math.inf # avoid repeating the prev-prev word

            if drop_bigrams and t > 1:
                prev_usage = predictions[:, :t-1] # of the previous word (e.g, xx yy xx) (first xx)
                x, y = torch.where(prev_usage == torch.unsqueeze(prev_words, -1))
                y += 1 # word-after-last-in-prev-usage  (e.g., yy in above)
                y = prev_usage[x, y].long()
                pred_t[x, y] = -math.inf

            if sampling_rule == 'argmax':
                prev_words = torch.argmax(pred_t, 1)
            elif sampling_rule == 'multinomial':
                probability = torch.softmax(pred_t, 1)
                prev_words = torch.multinomial(probability, 1).squeeze_(-1)
            elif sampling_rule == 'topk':
                row_idx = torch.arange(batch_size)
                row_idx = row_idx.view([1, -1]).repeat(topk, 1).t()
                # do soft-max after you zero-out non topk (you could also do this before, ask me/Panos if need be:) )
                val, ind = pred_t.topk(topk, dim=1)
                val = torch.softmax(val, 1)
                probability = torch.zeros_like(pred_t) # only the top-k logits will have non-zero prob.
                probability[row_idx, ind] = val
                prev_words = torch.multinomial(probability, 1).squeeze_(-1)

            predictions[:, t] = prev_words
        all_predictions.append(predictions.cpu().long())
    all_predictions = torch.cat(all_predictions)
    attention_weights = torch.stack(attention_weights, 1)
    return all_predictions, attention_weights


@torch.no_grad()
def sample_captions_beam_search(model, data_loader, beam_size, device, temperature=1, max_iter=500,
                                drop_unk=True, drop_bigrams=False):
    """
    :param model (encoder, decoder)
    :param data_loader:
    :param beam_size:
    :param drop_unk:
    :return:

        hypotheses_alphas: list carrying the attention maps over the encoded-pixel space for each produced token.
    Note: batch size must be one.
    """

    if data_loader.batch_size != 1:
        raise ValueError('not implemented for bigger batch-sizes')

    model.eval()
    decoder = model.decoder
    vocab = model.decoder.vocab

    captions = list()
    hypotheses_alphas = list()
    caption_log_prob = list()

    aux_feat = None
    for batch in tqdm.tqdm(data_loader):  # For each image (batch-size = 1)
        image = batch['image'].to(device)  # (1, 3, H, W)

        if model.decoder.uses_aux_data:
            aux_data = batch['emotion'].to(device)
            aux_feat = model.decoder.auxiliary_net(aux_data)

        k = beam_size
        encoder_out = model.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <sos>
        k_prev_words = torch.LongTensor([[vocab.sos]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <sos>
        seqs = k_prev_words # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s (below) is a number less than or equal to k, because sequences are removed
        # from this process once they hit <eos>
        while True:
            embeddings = decoder.word_embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, alpha = decoder.attention(encoder_out, h)   # (s, encoder_dim), (s, num_pixels)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            decoder_input = torch.cat([embeddings, awe], dim=1)

            if aux_feat is not None:
                af = torch.repeat_interleave(aux_feat, decoder_input.shape[0], dim=0)
                decoder_input = torch.cat([decoder_input, af], dim=1)

            h, c = decoder.decode_step(decoder_input, (h, c))  # (s, decoder_dim)
            scores = decoder.next_word(h)  # (s, vocab_size)

            if temperature != 1:
                scores /= temperature

            scores = F.log_softmax(scores, dim=1)

            if drop_unk:
                scores[:, vocab.unk] = -math.inf

            if drop_bigrams and step > 2:
                # drop bi-grams with frequency higher than 1.
                prev_usage = seqs[:, :step-1]
                x, y = torch.where(prev_usage == k_prev_words)
                y += 1 # word-after-last-in-prev-usage
                y = seqs[x, y]
                scores[x,y] = -math.inf

            if step > 2:
                ## drop x and x
                and_token = decoder.vocab('and')
                x, y = torch.where(k_prev_words == and_token)
                pre_and_word = seqs[x, step-2]
                scores[x, pre_and_word] = -math.inf

            # Add log-probabilities
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / len(vocab)  # (s)
            next_word_inds = top_k_words % len(vocab)  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <eos>)?
            incomplete_inds = [ind for ind, word in enumerate(next_word_inds) if word != vocab.eos]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]

            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > max_iter:
                break
            step += 1

        s_idx = np.argsort(complete_seqs_scores)[::-1]
        complete_seqs_scores = [complete_seqs_scores[i] for i in s_idx]
        complete_seqs = [complete_seqs[i] for i in s_idx]
        alphas = [complete_seqs_alpha[i] for i in s_idx]

        captions.append(complete_seqs)
        caption_log_prob.append(complete_seqs_scores)
        hypotheses_alphas.append(alphas)
    return captions, hypotheses_alphas, caption_log_prob


@torch.no_grad()
def properize_captions(captions, vocab, add_sos=True):
    """
    :param captions: torch Tensor holding M x max_len integers
    :param vocab:
    :param add_sos:
    :return:
    """
    # ensure they end with eos.

    new_captions = []
    missed_eos = 0
    for caption in captions.cpu():
        ending = torch.where(caption == vocab.eos)[0]
        if len(ending) >= 1: # at least one <eos> symbol is found
            first_eos = ending[0]
            if first_eos < len(caption):
                caption[first_eos+1:] = vocab.pad
        else:
            missed_eos += 1
            caption[-1] = vocab.eos
        new_captions.append(caption)

    new_captions = torch.stack(new_captions)

    dummy = torch.unique(torch.where(new_captions == vocab.eos)[0])
    assert len(dummy) == len(new_captions) # assert all have an eos.

    if add_sos:
        sos = torch.LongTensor([vocab.sos] * len(new_captions)).view(-1, 1)
        new_captions = torch.cat([sos, new_captions], dim=1)
    if missed_eos > 0:
        warnings.warn('{} sentences without <eos> were generated.'.format(missed_eos))
    return new_captions


def log_prob_of_dataset(model, data_loader, device, temperature=1):
    all_log_probs = []
    all_lens = []
    model.eval()
    for batch in data_loader:
        imgs = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        log_probs, n_tokens = log_prob_of_caption(model, imgs, tokens, temperature=temperature)
        all_log_probs.append(log_probs.cpu())
        all_lens.append(n_tokens.cpu())

    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_lens = torch.cat(all_lens, dim=0)
    return all_log_probs, all_lens


def perplexity_of_dataset(model, data_loader, device):
    """ for a test corpus perplexity is 2 ^ {-l} where l is log_2(prob_of_sentences) * M, where M is the number
    of tokens in the dataset.
    :param model:
    :param data_loader:
    :param device:
    :return:
    """
    all_log_probs, all_lens = log_prob_of_dataset(model, data_loader, device)
    log_prob_per_sent = torch.sum(all_log_probs, 1).double() # sum over tokens to get the log_p of each utterance
    prob_per_sent = torch.exp(log_prob_per_sent)
    n_tokens = torch.sum(all_lens).double()  # number of words in dataset
    average_log_prob = torch.sum(torch.log2(prob_per_sent)) / n_tokens   # log_2 for perplexity
    perplexity = 2.0 ** (-average_log_prob)
    return perplexity, prob_per_sent, all_lens

