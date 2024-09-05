import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]):
    if expr.axis is None:
        if values[0].is_sparse:
            values = values[0].todense().A.flatten()
        else:
            values = values[0].flatten()
    else:
        values = values[0]
    return torch.linalg.norm(values, torch.inf, dim=expr.axis)
