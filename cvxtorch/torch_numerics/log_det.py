from cvxpy.expressions.expression import Expression
import torch

def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
    symm = (values[0] + torch.conj(values[0].T))/2
    sign, logdet = torch.linalg.slogdet(symm)
    if np.isclose(np.real(sign), 1): #This has to be np
        return logdet
    else:
        return -torch.inf
