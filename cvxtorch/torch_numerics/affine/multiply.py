from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    if values[0].is_sparse:
        return values[0].multiply(values[1])
    elif values[1].is_sparse:
        return values[1].multiply(values[0])
    else:
        return torch.multiply(values[0], values[1])
