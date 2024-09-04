import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr:Expression, values: list[torch.Tensor]) -> torch.Tensor:
    values = values[0]
    if expr.axis is None:
        values = values.flatten()
    return torch.norm(values, p=1, dim=expr.axis)
