from enum import Enum

import numpy as np
import torch
from scipy.sparse import coo_matrix, issparse

VAR_TYPE = Enum("VAR_TYPE", "VARIABLE_PARAMETER CONSTANT EXPRESSION")

def gen_tensor(value, dtype=torch.float64) -> torch.Tensor:
    """This function generates a tensor from an np.array or a sparse matrix.
    If the input is a sparse matrix, a sparse tensor is generated."""
    if not issparse(value):
        return torch.tensor(value, dtype=dtype)
    value_coo = coo_matrix(value)
    vals = value_coo.data
    inds = np.vstack((value_coo.row, value_coo.col))
    i = torch.LongTensor(inds)
    v = torch.FloatTensor(vals)
    return torch.sparse.FloatTensor(i, v, torch.Size(value_coo.shape)).to(dtype)

def tensor_reshape_fortran(value: torch.Tensor, shape: tuple) -> torch.Tensor:
    """This function reshapes a tensor in Fortran order (similar to numpy.reshape with order="F").
    This functionality is not included in Pytorch."""
    # reverse_shape = list(shape)
    # reverse_shape.reverse() #reverse a list in place
    # return torch.reshape(value.reshape(reverse_shape).t(), shape=shape)
    # A more compact solution based on
    # https://stackoverflow.com/questions/64433896/pytorch-equivalent-of-numpy-reshape-function.
    return torch.reshape(value.T, shape[::-1]).T
