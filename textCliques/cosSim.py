# Copyright(C) Shruti Phadke

import numpy as np
import json
import ast
import os
import torch
from torchmetrics.functional import pairwise_cosine_similarity


def get_pairwise_cossim(embeddings):
    x = torch.tensor(embeddings, dtype=torch.float32)
    sample_cosSim = pairwise_cosine_similarity(x)
    sample_cosSim = np.array(sample_cosSim)
    return sample_cosSim


def filter_cosSim_matrix(sample_cosSim, threshold_similarity=0.9):
    sample_cosSim[np.isnan(sample_cosSim)] = 0
    sample_cosSim[np.isinf(sample_cosSim)] = 0
    np.fill_diagonal(sample_cosSim, 0)
    sample_cosSim[sample_cosSim < threshold_similarity] = 0
    return sample_cosSim


def get_nonzero_similarity_tuples(sample_cosSim):
    nonzero_entries = np.nonzero(sample_cosSim)
    tot_nonzero = nonzero_entries[0].shape[0]
    return tot_nonzero, nonzero_entries
    
    
    
