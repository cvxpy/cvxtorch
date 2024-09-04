import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]):
    #values.new(1) creates a new 0 tensor on the same device as values[0].
    return torch.logaddexp(values[0].new(1), values[0])
