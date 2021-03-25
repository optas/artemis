"""
Encoding discrete tokens with LSTMs.

The MIT License (MIT)
Originally created at 2019, (updated on January 2020) for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """A feed-forward network that processes discrete tokens via an LSTM."""

    def __init__(self, n_input, n_hidden, word_embedding, word_transformation=None,
                 bidirectional=False, init_h=None, init_c=None, eos_symbol=None, feature_type='last'):
        """
        :param n_input: (int) input dim of LSTM
        :param n_hidden: (int) hidden dim of LSTM
        :param word_embedding: (nn.Embedding) vectors representing words
        :param word_transformation: (opt, nn.Module) to apply some transformation on the word
        embeddings before they are consumed by the LSTM.
        :param bidirectional: boolean, whether to use a bi-RNN
        :param init_h: (opt, nn.Module) for initializing LSTM hidden state
        :param init_c: (opt, nn.Module) for initializing LSTM memory
        :param eos_symbol: (opt, int) integer marking end of sentence
        :param feature_type: (opt, string) how to process the output of the LSTM,
            valid options = ['last', 'max', 'mean', 'all']
        """

        super().__init__()
        self.word_embedding = word_embedding
        self.n_hidden = n_hidden
        self.eos = eos_symbol
        self.feature_type = feature_type

        # auxiliary (optional) networks
        self.word_transformation = word_transformation
        self.init_h = init_h
        self.init_c = init_c

        self.rnn = nn.LSTM(input_size=n_input, hidden_size=n_hidden,
                           bidirectional=bidirectional, batch_first=True)

    def out_dim(self):
        rnn = self.rnn
        mult = 2 if rnn.bidirectional else 1
        return rnn.num_layers * rnn.hidden_size * mult

    def __call__(self, tokens, grounding=None, len_of_sequence=None):
        """
        :param tokens:
        :param grounding: (Tensor, opt)
        :param len_of_sequence: (Tensor:, opt) singleton tensor of shape (B,) carrying the length of the tokens
        :return: the encoded by the LSTM tokens
            Note: a) tokens are padded with the <sos> token
        """
        w_emb = self.word_embedding(tokens[:, 1:]) # skip <sos>
        if self.word_transformation is not None:
            w_emb = self.word_transformation(w_emb)

        device = w_emb.device

        if len_of_sequence is None:
            len_of_sequence = torch.where(tokens == self.eos)[1] - 1  # ignore <sos>

        x_packed = pack_padded_sequence(w_emb, len_of_sequence, enforce_sorted=False, batch_first=True)

        self.rnn.flatten_parameters()

        if grounding is not None:
            h0 = self.init_h(grounding).unsqueeze(0)  # rep-mat if multiple LSTM cells.
            c0 = self.init_c(grounding).unsqueeze(0)
            rnn_out, _ = self.rnn(x_packed, (h0, c0))
        else:
            rnn_out, _ = self.rnn(x_packed)

        rnn_out, dummy = pad_packed_sequence(rnn_out, batch_first=True)

        if self.feature_type == 'last':
            batch_size = len(tokens)
            lang_feat = rnn_out[torch.arange(batch_size), len_of_sequence-1]
        elif self.feature_type == 'max':
            lang_feat = rnn_out.max(1).values
        elif self.feature_type == 'mean':
            lang_feat = rnn_out.sum(1)
            lang_feat /= len_of_sequence.view(-1, 1)  # broadcasting
        elif self.feature_type == 'all':
            lang_feat = rnn_out
        else:
            raise ValueError('Unknown LSTM feature requested.')

        return lang_feat
