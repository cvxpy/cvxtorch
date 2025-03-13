import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    X = values[0]
    P = values[1]
    if expr.args[0].is_complex():
        # Use permute instead of .T to avoid the deprecation warning
        ndim = X.ndim
        conj_x_perm = (torch.conj(X)).permute(*torch.arange(ndim - 1, -1, -1))
        product = conj_x_perm @ (torch.linalg.inv(P)) @ X
    else:
        # Use permute instead of .T to avoid the deprecation warning
        ndim = X.ndim
        product = X.permute(*torch.arange(ndim - 1, -1, -1)) @ (torch.linalg.inv(P)) @ X
    return product.trace() if len(product.shape) == 2 else product
