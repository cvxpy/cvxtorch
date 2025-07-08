import torch
from cvxpy.expressions.expression import Expression


class sin:
    @staticmethod
    def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
        return torch.sin(values[0])

class cos:
    @staticmethod
    def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cos(values[0])
