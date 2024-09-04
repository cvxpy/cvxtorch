import torch


def torch_numeric(expr, values: list[torch.Tensor]):
    if expr.axes is None:
        return values[0].T
    return values[0].permute(expr.axes)
