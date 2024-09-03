from functools import reduce
from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(self, values: list[torch.Tensor]) -> torch.Tensor:
    return reduce(torch.minimum, values)
