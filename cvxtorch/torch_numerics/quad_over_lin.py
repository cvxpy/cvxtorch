import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if expr.args[0].is_complex():
        return (torch.square(values[0].imag) + torch.square(values[0].real)).sum()/values[1]
    return torch.square(values[0]).sum()/values[1]
