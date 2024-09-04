import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    x = values[0]
    x = torch.sign(x)
    x[x==0] = -1.0
    return x
