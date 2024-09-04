import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.norm(values[0] - expr.a) / torch.linalg.norm(values[0] - expr.b)
