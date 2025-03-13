from enum import Enum

import numpy as np
import torch
from cvxpy.expressions.expression import Expression
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
    # Use permute instead of .T to avoid the deprecation warning
    ndim = value.ndim
    perm = torch.arange(ndim - 1, -1, -1)
    return torch.reshape(value.permute(*perm), shape[::-1]).permute(*perm)

def get_torch_numeric(expr: Expression) -> callable:
    """
    This function returns the torch_numeric function of this atom.
    It supports creating custom torch_numeric functions for atoms by the user.
    If the user provides a torch_numeric function for the atom, this function will return it.
    Otherwise, get the default one provided by this pacjage (from EXPR2TORCH).
    """
    from cvxtorch.utils.exp2tch import EXPR2TORCH
    torch_numeric = getattr(expr, "torch_numeric", None)
    if torch_numeric:
        return torch_numeric
    return EXPR2TORCH.get(type(expr))
