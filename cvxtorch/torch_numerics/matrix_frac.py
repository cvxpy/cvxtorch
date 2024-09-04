import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    X = values[0]
    P = values[1]
    if expr.args[0].is_complex():
        product = (torch.conj(X)).T @ (torch.linalg.inv(P)) @ X
    else:
        product = (X.T) @ (torch.linalg.inv(P)) @ X
    return product.trace() if len(product.shape) == 2 else product
