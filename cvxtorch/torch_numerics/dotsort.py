import numpy as np
import torch
from cvxpy.expressions.expression import Expression


def _get_args_from_values(values: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    x = values[0].flatten()
    w = values[1].flatten()

    w_padded = torch.zeros_like(x)  # pad in case size(W) < size(X)
    w_padded[:len(w)] = w
    return x, w_padded

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    x, w_padded = _get_args_from_values(values)
    return torch.sort(x)[0] @ torch.sort(w_padded)[0]