import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    # return torch.ones(values[0].shape) - values[0]
    # 1+values[0].new(values[0].shape) creates a ones tensor in the same device as values[0]
    return 1+values[0].new(values[0].shape)-values[0]
