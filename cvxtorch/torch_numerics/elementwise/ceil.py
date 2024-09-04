import cvxpy.settings as s
import numpy as np
import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    decimals = int(np.abs(np.log10(s.ATOM_EVAL_TOL))) #np by design
    return torch.ceil(torch.round(values[0], decimals=decimals))
