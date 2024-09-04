import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    # if values[0] isn't Hermitian then return np.inf
    if torch.any(torch.linalg.norm(values[0] - values[0].T.conj()) >= 1e-8).item():
        return torch.inf
    # take symmetric part of the input to enhance numerical stability
    symm = (values[0] + values[0].T)/2
    eigVal = torch.linalg.eigvalsh(symm)
    if min(eigVal) <= 0:
        return torch.inf
    return torch.sum(eigVal**-1)
