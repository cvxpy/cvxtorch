import torch
from cvxpy.expressions.expression import Expression


def torch_numeric(expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:    
    inds = torch.triu_indices(row=values[0].shape[0], col=values[0].shape[1], offset=1)
    # This can be simplified as `return values[0][*inds]`. However, this doesn't work on <3.11.
    # I ended up doing the following solution to support both versions (checking the version
    # during runtime causes issues during testing).
    # In future Python versions >=3.11, this should be replaced.
    if len(inds) != 2:
        raise ValueError(f"Expected inds to be of length 2, but got {len(inds)} instead.")
    return values[0][inds[0], inds[1]]
