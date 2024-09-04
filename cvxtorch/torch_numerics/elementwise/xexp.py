import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    return values[0] * torch.exp(values[0])
