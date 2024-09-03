from cvxpy.expressions.expression import Expression
import torch

def tensor_reshape_fortran(value: torch.Tensor, shape: tuple) -> torch.Tensor:
    """This function reshapes a tensor in Fortran order (similar to numpy.reshape with order="F").
    This functionality is not included in Pytorch."""
    # reverse_shape = list(shape)
    # reverse_shape.reverse() #reverse a list in place
    # return torch.reshape(value.reshape(reverse_shape).t(), shape=shape)
    # A more compact solution based on
    # https://stackoverflow.com/questions/64433896/pytorch-equivalent-of-numpy-reshape-function.
    return torch.reshape(value.T, shape[::-1]).T

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if expr.order == "F":
        return tensor_reshape_fortran(values[0], expr.shape)
    return torch.reshape(values[0], expr.shape) #order="C" reshaping