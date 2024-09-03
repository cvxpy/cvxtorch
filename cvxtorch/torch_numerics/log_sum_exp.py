from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if expr.axis is None:
        return torch.special.logsumexp(values[0].flatten(), dim=0, keepdim=expr.keepdims)
    return torch.special.logsumexp(values[0], dim=expr.axis, keepdim=expr.keepdims)
