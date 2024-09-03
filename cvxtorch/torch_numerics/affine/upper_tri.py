from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    inds = torch.triu_indices(row=values[0].shape[0], col=values[0].shape[1], offset=1)
    return values[0][*inds]
