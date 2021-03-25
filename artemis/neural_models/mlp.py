"""
Multi-Linear Perceptron packaged nicely for convenience.

The MIT License (MIT)
Originally created in late 2019, for Python 3.x. Last updated in 2021.
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

from torch import nn
import numpy as np

def optional_repeat(value, times):
    """ helper function, to repeat a parameter's value many times
    :param value: an single basic python type (int, float, boolean, string), or a list with length equals to times
    :param times: int, how many times to repeat
    :return: a list with length equal to times
    """
    if type(value) is not list:
        value = [value]

    if len(value) != 1 and len(value) != times:
        raise ValueError('The value should be a singleton, or be a list with times length.')

    if len(value) == times:
        return value # do nothing

    return np.array(value).repeat(times).tolist()


class MLP(nn.Module):
    """ Multi-near perceptron. That is a k-layer deep network where each layer is a fully-connected layer, with
    (optionally) batch-norm, a non-linearity and dropout. The last layer (output) is always a 'pure' linear function.
    """
    def __init__(self, in_feat_dims, out_channels, b_norm=True, dropout_rate=0,
                 non_linearity=nn.ReLU(inplace=True), closure=None):
        """Constructor
        :param in_feat_dims: input feature dimensions
        :param out_channels: list of ints describing each the number hidden/final neurons. The
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        :param closure: optional nn.Module to use at the end of the MLP
        """
        super(MLP, self).__init__()
        self.hidden_dimensions = out_channels[:-1]
        self.embedding_dimension = out_channels[-1]

        n_layers = len(out_channels)
        dropout_rate = optional_repeat(dropout_rate, n_layers-1)
        b_norm = optional_repeat(b_norm, n_layers-1)

        previous_feat_dim = in_feat_dims
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:
                if b_norm[depth]:
                    all_ops.append(nn.BatchNorm1d(out_dim))

                if non_linearity is not None:
                    all_ops.append(non_linearity)

                if dropout_rate[depth] > 0:
                    all_ops.append(nn.Dropout(p=dropout_rate[depth]))

            previous_feat_dim = out_dim

        if closure is not None:
            all_ops.append(closure)

        self.net = nn.Sequential(*all_ops)

    def __call__(self, x):
        return self.net(x)
