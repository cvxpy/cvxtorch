from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cumsum(values[0], axis=expr.axis)
