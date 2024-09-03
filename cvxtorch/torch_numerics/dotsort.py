from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    x, w_padded = expr._get_args_from_values(values, mod=torch)
    return torch.sort(x)[0] @ torch.sort(w_padded)[0]