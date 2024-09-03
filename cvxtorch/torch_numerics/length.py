import cvxpy.settings as s
from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    outside_tol = torch.abs(values[0]) > s.ATOM_EVAL_TOL
    return torch.max(torch.nonzero(outside_tol)) + 1
