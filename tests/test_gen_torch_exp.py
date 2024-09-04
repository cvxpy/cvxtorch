import unittest
import pytest

import cvxpy as cp
import torch
import numpy as np

from cvxtorch.torch_expression import TorchExpression

class TestGenTorchExp(unittest.TestCase):
    """ Unit tests for the atoms module. """

    def test_gen_torch_exp(self):
        #Tests the functionality of gen_torch_exp
        n = 3
        m = 2
        x = cp.Variable(n)
        w = cp.Parameter(n)
        w.value=np.ones(n)
        Q = np.array([[2,2,1],[1,-1,2],[-1,-1,1]]) #3x3
        a = 3*np.ones(n)
        t1 = torch.randn(n)
        t2 = torch.randn(n)
        T1 = torch.ones((m,n)) #2x3
        T2 = torch.ones((m,n)) #2x3
        X = cp.Variable((m,n))
        Y = cp.Parameter((m,n))

        exp1 = x+w+a+x+w
        exp2 = x+w+a+x@w+x
        exp3 = cp.norm(Q@x+w+a)
        exp4 = x-w
        exp5 = w-x
        exp6 = X@Y.T

        torch_exp1 = TorchExpression(exp1).tch_exp
        torch_exp2 = TorchExpression(exp2).tch_exp
        torch_exp3 = TorchExpression(exp3).tch_exp
        torch_exp4 = TorchExpression(exp4).tch_exp
        torch_exp5 = TorchExpression(exp5).tch_exp
        torch_exp6 = TorchExpression(exp6).tch_exp

        test1 = torch_exp1(5*torch.ones(n), torch.tensor([1.,2.,3.]))
        test2 = torch_exp2(1*torch.ones(n), torch.tensor([1.,2.,3.]))
        test3 = torch_exp3(2*torch.ones(n, dtype=float), torch.tensor([2.,1.,2.]))
        test4 = torch_exp4(t1, t2)
        test5 = torch_exp5(t1, t2)
        test6 = torch_exp6(T1, T2)

        self.assertTrue(all(test1==torch.tensor([15., 17., 19.])))
        self.assertTrue(all(test2==torch.tensor([12, 13, 14])))
        self.assertTrue(np.isclose(test3, 17.2626))
        #Variables and parameters are treated similarly
        self.assertTrue(all(np.isclose(test4, test5))) 
        self.assertTrue((test6==n*torch.ones((m,m))).all())