"""
Utilities for distance measurements in GPU.

The MIT License (MIT)
Originally created at 07/2019, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
from torch.nn.functional import normalize

def cdist(x1, x2, epsilon=1e-16):
    """
    :param x1: N x Feat-dim
    :param x2: N x Feat-dim
    :param epsilon:
    :return: N x N matrix
    """
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    inner_prod = torch.mm(x1, x2.t())
    res = x1_norm - 2.0 * inner_prod + x2_norm.t()   # You need to transpose for broadcasting to be correct.
    res.clamp_min_(epsilon).sqrt_()
    return res


def exclude_identity_from_neighbor_search(all_pairwise_dists, identities):
    """
    :param all_pairwise_dists: M x N matrix of distances
    :param identities: the k-th row of all_pairwise_dists, should exclude the identities[k] entry.
    :return:
    """
    all_pairwise_dists[range(all_pairwise_dists.size(0)), identities] = float("Inf")
    return all_pairwise_dists


def k_euclidean_neighbors(k, x1, x2, exclude_identity=False, identities=None):
    """ For each row vector in x1 the k-nearest neighbors in x2.
    :param k:
    :param x1: M x Feat-dim
    :param x2: N x Feat-dim
    :param exclude_identity:
    :param identities:
    :return: M x k
    """
    all_cross_pairwise_dists = cdist(x1, x2)
    if exclude_identity:
        all_cross_pairwise_dists = exclude_identity_from_neighbor_search(all_cross_pairwise_dists, identities)
    n_dists, n_ids = all_cross_pairwise_dists.topk(k=k, dim=1, largest=False, sorted=True)
    return n_dists, n_ids


def k_cosine_neighbors(k, x1, x2, exclude_identity=False, identities=None):
    """ For each row vector in x1 the k-nearest neighbors in x2.
    :param k:
    :param x1: M x Feat-dim
    :param x2: N x Feat-dim
    :param exclude_identity:
    :param identities:
    :return: M x k
    """
    all_cross_pairwise_dists = torch.mm(normalize(x1, dim=1, p=2), normalize(x2, dim=1, p=2).t())
    all_cross_pairwise_dists = 1.0 - all_cross_pairwise_dists
    if exclude_identity:
        all_cross_pairwise_dists = exclude_identity_from_neighbor_search(all_cross_pairwise_dists, identities)
    n_dists, n_ids = all_cross_pairwise_dists.topk(k=k, dim=1, largest=False, sorted=True)
    return n_dists, n_ids