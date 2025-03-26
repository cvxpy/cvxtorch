import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    # Convert numpy arrays to tensors first to avoid __array_wrap__ deprecation warnings
    a_tensor = torch.tensor(expr.a, dtype=values[0].dtype, device=values[0].device)
    b_tensor = torch.tensor(expr.b, dtype=values[0].dtype, device=values[0].device)
    
    # Calculate norms using tensor operations
    norm_a = torch.linalg.norm(values[0] - a_tensor)
    norm_b = torch.linalg.norm(values[0] - b_tensor)
    return norm_a / norm_b
