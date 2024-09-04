import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    values = values[0]
    if expr.axis is None:
        values = values.flatten()

    if expr.p < 1 and torch.any(values<0).item():
        return -torch.inf
    if expr.p < 0 and torch.any(values==0).item():
        return 0.0

    return torch.linalg.norm(values, float(expr.p), axis=expr.axis, keepdims=expr.keepdims)
