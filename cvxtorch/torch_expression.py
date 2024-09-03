from functools import partial

import torch

from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable

from cvxtorch.variables_dict.variables_dict import VariablesDict
from cvxtorch.utils.torch_utils import VAR_TYPE, gen_tensor
from cvxtorch.utils.exp2tch import EXPR2TORCH

#TODO: Add unit tests to
#TODO: Add docstring
class TorchExpression():
    """
    
    """
    @property
    def tch_exp(self):
        return self._tch_exp
    
    @property
    def vars_dict(self):
        return self._vars_dict

    def __init__(self, expr: Expression, provided_vars_list:list = [], implemented_only: bool=True):
        self.implemented_only = implemented_only
        self._tch_exp, self._vars_dict = self._gen_torch_exp(expr=expr, provided_vars_list=provided_vars_list)

    #TODO: Update the docstring
    def _gen_torch_exp(self, expr: Expression, provided_vars_list:list = []) -> tuple[callable, VariablesDict]:
        """
        This function generates a torch expression from the args of this expression.
        A torch expression is a function that implements a torch function that evaluates the same
        mathematical expression as the CVXPY expression.

        For example, if the expression is a subtraction expression between two variables,
        then the generated torch expression is a function that subtracts two tensors.

        .. code:: python
            import torch
            import cvxpy as cp
            x = cp.Variable()
            y = cp.Variable()
            exp = x-y
            tch_exp, _ = exp.gen_torch_exp() # This implements x-y
            tch_exp(torch.tensor(5), torch.tensor(3)) # This returns a torch.Tensor(2).

        The order of the arguments in the torch expression is the same order as in args of this
        expression by default, or can be passed by the user.

        Arguments:
            provided_vars_list (list): default=[]
                A list of CVXPY atoms. This list determines the argument positions of the generated
                torch expression.
                If an empty list is passed (default), then the order of arguments is the same as
                in args of this expression (from left to right).
        
        Returns:
            tch_exp (callable):
                The generated torch expression.
            
            vars_dict (cvxpy.utilities.torch_utils.VariablesDict):
                An object that maps from CVXPY atoms to their indices in the generated torch
                expression.
        """

        def _gen_var_type(arg) -> VAR_TYPE:
            """
            This is a helper function that generates a VAR_TYPE from an arg.
            """
            if isinstance(arg, Constant):
                return VAR_TYPE.CONSTANT
            elif isinstance(arg, Parameter) or isinstance(arg, Variable):
                return VAR_TYPE.VARIABLE_PARAMETER
            else:
                return VAR_TYPE.EXPRESSION

        def _gen_consts_vars(expr: Expression, vars_dict: VariablesDict) -> dict:
            """ This is a helper function that generates the index -> (value, type) dictionary. """
            ind_to_value_type = dict() #Local dictionary
            for i, arg in enumerate(expr.args):
                var_type = _gen_var_type(arg)
                if isinstance(arg, Constant):
                    ind_to_value_type[i] = (gen_tensor(arg.value, dtype=torch.float64), var_type)
                elif isinstance(arg, Parameter) or isinstance(arg, Variable):
                    ind_to_value_type[i] = (arg, var_type)
                    vars_dict.add_var(arg)
                else:
                    ind_to_value_type[i] = (arg, var_type)
                    _gen_consts_vars(arg, vars_dict)
            return ind_to_value_type
        
        def wrapped_func(self, expr: Expression, ind_to_value_type: dict, vars_dict: VariablesDict, *args):
            def transpose_if_matmul(expr: Expression, res: list, should_transpose: list) -> None:
                """
                This function transposes the second element if the wrapped function is a dot product
                between two vectors. While transposing a vector in CVXPY does nothing, this function
                is important because it helps overloading this function to the case where the
                one of the elements is a matrix, where each row is a vector to be multiplied with.
                """

                def is_matmul(expr: Expression) -> bool:
                    """
                    This function checks if self is a valid matrix multiplication
                    """
                    from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
                    if not isinstance(expr, MulExpression):
                        return False
                    #Check if the current expression is not elementwise multiplication
                    if isinstance(expr, multiply):
                        return False
                    return True

                def check_should_transpose(should_transpose: list) -> bool:
                    """
                    This is a helper function that determines if transpose should happen, based
                    on the CVXPY types.
                    """
                    if len(should_transpose) != 2: #Only support a dot product, hence two elements
                        return False
                    return all(should_transpose)
                
                def is_valid_input(res: list) -> tuple[bool, list]:
                    """
                    This function checks if res is valid for transpose, and if it does, returns
                    the ndims list.
                    """
                    ndims = None
                    valid = False

                    #######################
                    #Check input validity
                    #######################
                    #Transpose only a dot product between two elements
                    if len(res) != 2:
                        return valid, ndims
                    
                    #Work only on objects with ndim (torch/numpy/cvxpy etc.) and ndim==1
                    ndims = [0]*2
                    for i, vec in enumerate(res):
                        if not hasattr(vec, "ndim"):
                            return valid, ndims
                        if vec.ndim >2:
                            return valid, ndims#Only deal with scalars, vectors, or matrices
                        ndims[i] = vec.ndim

                    if max(ndims)<2:
                        #No need to transpose scalars and vectors
                        #If both elements are matrices, do not transpose -
                        # we only make a vector-by-matrix compatible.
                        return valid, ndims 
                    if min(ndims)==2:
                        return valid, ndims
                    valid = True
                    return valid, ndims

                if not is_matmul(expr):
                    return
                if not check_should_transpose(should_transpose):
                    return
                valid, ndims = is_valid_input(res)
                if not valid:
                    return
                matrix_ind = ndims.index(2)
                #Since ndims is a vector with 2 elements, 1-matrix_ind returns the other index
                vector_ind = 1-matrix_ind
                if res[vector_ind].shape[0] == res[matrix_ind].shape[matrix_ind]:
                    res[matrix_ind] = res[matrix_ind].T

            def should_transpose(curr_arg):
                """
                This is a helper function that determines if an element should be transposed for
                matrix multiplication (allows broadcasting vectors into matrices)
                """
                ndim = getattr(curr_arg[0], "ndim", False)
                if curr_arg[1]==VAR_TYPE.CONSTANT and ndim != 1:
                    return False
                return ndim==1
            res =  []
            #In order to support dot products of vectors,
            #where one vector is represented by a matrix, we need to see if:
            #1. The operation is matmul
            #2. between two vectors (variables/parameters with ndim==1)
            #These checks should happen on the original variables/parameters,
            #and NOT on the elements of args.
            #So the transpose happens only if should_tranpose contains two Trues.
            transposable_elements = []
            #Iterate over the range instead of the dictionary directly:
            #dictionaries have no order, but we must iterate in order
            for ind in range(len(ind_to_value_type)): 
                curr_arg = ind_to_value_type[ind]
                if curr_arg[1]==VAR_TYPE.CONSTANT:
                    res.append(curr_arg[0])
                elif curr_arg[1]==VAR_TYPE.VARIABLE_PARAMETER:
                    res.append(args[vars_dict.vars_dict[curr_arg[0]]])
                else:
                    rec_ind_to_value_type = _gen_consts_vars(curr_arg[0], vars_dict)
                    res.append(wrapped_func(curr_arg[0], rec_ind_to_value_type, vars_dict, *args))
                transposable_elements.append(should_transpose(curr_arg))
            #If this is a matrix multiplicaiton operation between 2 elements, transpose the second.
            #This helps with overloading this function to be used with matrices.
            transpose_if_matmul(expr, res, transposable_elements)
            return self.apply_torch_numeric(expr, res)
        
        vars_dict = VariablesDict(provided_vars_list=provided_vars_list)
        ind_to_value_type = _gen_consts_vars(expr, vars_dict)
        return partial(wrapped_func, self, expr, ind_to_value_type, vars_dict), vars_dict
    

    def apply_torch_numeric(self, expr: Expression, values: list[torch.Tensor]) -> torch.Tensor:
        """
        This function returns self.torch_numeric(values) if it exists,
        and self.numeric(values) otherwise.
        """
        torch_numeric = EXPR2TORCH.get(type(expr))
        if torch_numeric:
            return torch_numeric(expr, values)
        elif not self.implemented_only:
            return expr.numeric(values)
        else:
            raise NotImplementedError(f"The torch_numeric function of {expr} is not implemented."
                                      f"If you want to use CVXPY's numeric instead, pass"
                                      f"implemented_only=False.")
