from functools import reduce

import torch


def torch_numeric(self, values: list[torch.Tensor]) -> torch.Tensor:
    return reduce(torch.minimum, values)
