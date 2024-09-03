from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr, values: list[torch.Tensor]) -> torch.Tensor:
    return torch.max(torch.abs(torch.linalg.eig(values[0])[0]))
