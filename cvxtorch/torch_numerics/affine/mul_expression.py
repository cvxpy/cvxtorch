import cvxpy.interface as intf
import torch


def torch_numeric(expr, values: list[torch.Tensor]) -> torch.Tensor:
    if values[0].shape == () or values[1].shape == () or \
        intf.is_sparse(values[0]) or intf.is_sparse(values[1]):
        return values[0] * values[1]
    else:
        return torch.matmul(values[0], values[1])
