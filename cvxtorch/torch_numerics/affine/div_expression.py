from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    for i in range(2):
        if values[i].is_sparse:
            values[i] = values[i].todense().A
    return torch.divide(values[0], values[1])
