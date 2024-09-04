import cvxpy.interface as intf
import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if intf.is_sparse(values[0]):
        result = torch.sum(values[0], axis=expr.axis)
        if not expr.keepdims and expr.axis is not None:
            result = result.A.flatten()
    else:
        result = torch.sum(values[0], axis=expr.axis, keepdims=expr.keepdims)
    return result
