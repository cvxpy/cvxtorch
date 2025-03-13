import torch


def torch_numeric(expr, values: list[torch.Tensor]):
    if expr.axes is None:
        # Use permute instead of .T to avoid the deprecation warning
        ndim = values[0].ndim
        return values[0].permute(*torch.arange(ndim - 1, -1, -1))
    return values[0].permute(expr.axes)
