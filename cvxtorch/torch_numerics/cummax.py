from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cummax(values[0], dim=expr.axis)[0]