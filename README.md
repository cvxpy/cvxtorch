# CVXTorch

This package generates Pytorch expressions from CVXPY expressions and constraints.
A Pytorch expression is a function that evaluates similarly to a CVXPY's ``numeric`` function.
A torch expression and a mapping from CVXPY leaves to their indices in the Pytorch expression
are returned.

## Installation
To install CVXTorch, clone the repo from the source, and
```
pip install -e .
```

## Example

```python
import cvxpy as cp
import torch
from cvxtorch import TorchExpression

n = 5
x = cp.Variable(n, name="x")
y = cp.Parameter(n, name="y")
z = 3
exp = x-y+2*z

tch_x = torch.arange(1, n+1)
tch_y = torch.arange(0, n)

tch_exp = TorchExpression(exp).tch_exp #tch_exp implements x-y+2*z, where x and y are torch.Tensor.
tch_res = tch_exp(tch_x, tch_y) #Contains a torch.Tensor [7.0]*n
```

## Returned object

A CVXTorch TorchExpression object is created. It has two properties:
* **tch_exp (callable)**: The generated torch expression.
  
* **vars_dict (cvxtorch.VariablesDict)**:
      An object that maps from CVXPY atoms to their indices in the generated torch expression.
