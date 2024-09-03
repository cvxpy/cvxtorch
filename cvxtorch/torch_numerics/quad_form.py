from cvxpy.expressions.expression import Expression
import torch

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
        quad = multiply(torch.conj(values[0]).T, prod)
    else:
        quad = multiply(values[0].T, prod)
    return torch.real(quad)
