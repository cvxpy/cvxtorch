import torch


def torch_numeric(self, values: list[torch.Tensor]) -> torch.Tensor:
    return torch.trace(values[0])
