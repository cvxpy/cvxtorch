from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    logX = torch.log(values[0])
    return torch.exp(torch.tensor(expr.A.value) @ logX)
