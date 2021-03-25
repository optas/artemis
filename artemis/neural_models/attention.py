"""
Language-Vision Attention Utilities.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""


from torch import nn


class AdditiveVisioLinguistic(nn.Module):
    """
    Given a vector summarizing the linguistic information processed by a pipeline
    (e.g. k-th output of RNN) attend to a 2D grid (e.g., image pixels).
    This mechanism *adds* the two sources of information to compute the attention (hence the name additive).
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: (int) feature size (last dimension) of encoded images (e.g., [B x H x W] x encoder_dim)
        :param decoder_dim: (int) feature size of decoder's output (summarizing linguistic information)
        :param attention_dim: (int) feature size size of the attention space
        """
        super(AdditiveVisioLinguistic, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def __call__(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha
