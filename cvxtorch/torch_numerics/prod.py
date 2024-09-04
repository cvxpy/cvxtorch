import cvxpy.interface as intf
import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if intf.is_sparse(values[0]):
        if expr.axis is None:
            result = torch.prod(values[0])
        else:
            result = torch.prod(values[0], axis=expr.axis)
        if not expr.keepdims and expr.axis is not None:
            result = result.A.flatten()
    else:
        if expr.axis is None:
            result = torch.prod(values[0])
        else:
            result = torch.prod(values[0], axis=expr.axis, keepdims=expr.keepdims)
    return result
