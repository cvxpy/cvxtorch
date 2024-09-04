import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.inv(torch.eye(expr.args[0].shape[0]) - values[0])