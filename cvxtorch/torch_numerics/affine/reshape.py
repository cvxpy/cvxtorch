import torch
from cvxpy.expressions.expression import Expression

from cvxtorch.utils.torch_utils import tensor_reshape_fortran


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if expr.order == "F":
        return tensor_reshape_fortran(values[0], expr.shape)
    return torch.reshape(values[0], expr.shape) #order="C" reshaping