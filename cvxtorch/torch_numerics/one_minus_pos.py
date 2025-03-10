import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    # return torch.ones(values[0].shape) - values[0]. Should be in the same device.
    return 1-values[0]
