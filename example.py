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

tch_exp = TorchExpression(exp) #tch_exp implements x-y+2*z, where x and y are torch.Tensor.
tch_exp_eval = tch_exp.torch_expression #Torch expression evaluation function.
tch_exp_dict = tch_exp.variables_dictionary #Variables mapping
tch_res = tch_exp_eval(tch_x, tch_y) #Contains a torch.Tensor [7.0]*n
print(f"The result tensor is {tch_res}.")
