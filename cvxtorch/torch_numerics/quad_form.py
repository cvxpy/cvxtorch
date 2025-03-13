import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    def multiply(x, prod):
        """
        This is an inner function that multiplies x by prod (scalar or tensor)
        """
        if prod.shape:
            return x @ prod
        return x*prod
    prod = values[1] @ (values[0])
    if expr.args[0].is_complex():
        # Use permute instead of .T to avoid the deprecation warning
        ndim = values[0].ndim
        quad = multiply(torch.conj(values[0]).permute(*torch.arange(ndim - 1, -1, -1)), prod)
    else:
        # Use permute instead of .T to avoid the deprecation warning
        ndim = values[0].ndim
        quad = multiply(values[0].permute(*torch.arange(ndim - 1, -1, -1)), prod)
    return torch.real(quad)
