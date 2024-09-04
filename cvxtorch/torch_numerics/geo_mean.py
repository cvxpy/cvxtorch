import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    values = values[0]
    values = values.flatten()
    log_tensor = torch.log(values)
    mean_log = torch.mean(log_tensor)
    return torch.exp(mean_log)
