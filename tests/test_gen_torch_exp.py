import unittest

import cvxpy as cp
import numpy as np
import torch
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero

from cvxtorch.torch_expression import TorchExpression

seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)

class TestGenTorchExp(unittest.TestCase):
    """ Unit tests for the atoms module. """

    def test_exp(self):
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

        torch_exp1 = TorchExpression(exp1).torch_expression
        torch_exp2 = TorchExpression(exp2).torch_expression
        torch_exp3 = TorchExpression(exp3).torch_expression
        torch_exp4 = TorchExpression(exp4).torch_expression
        torch_exp5 = TorchExpression(exp5).torch_expression
        torch_exp6 = TorchExpression(exp6).torch_expression

        test1 = torch_exp1(5*torch.ones(n), torch.tensor([1.,2.,3.]))
        test2 = torch_exp2(1*torch.ones(n), torch.tensor([1.,2.,3.]))
        test3 = torch_exp3(2*torch.ones(n, dtype=torch.float64), torch.tensor([2.,1.,2.]))
        test4 = torch_exp4(t1, t2)
        test5 = torch_exp5(t1, t2)
        test6 = torch_exp6(T1, T2)

        self.assertTrue(all(test1==torch.tensor([15., 17., 19.])))
        self.assertTrue(all(test2==torch.tensor([12, 13, 14])))
        self.assertTrue(np.isclose(test3, 17.2626))
        #Variables and parameters are treated similarly
        self.assertTrue(all(np.isclose(test4, test5))) 
        self.assertTrue((test6==n*torch.ones((m,m))).all())

    def test_constraint(self) -> None:
        #Tests generating a torch expression from a constraint
        m = 2
        n = 3
        s0 = np.array([-3, 2])
        s0 = np.maximum(s0, 0)
        x0 = np.ones(n)
        A = np.array([[1, -1, 2], [3, 1, -1]]) #2x3
        b = A @ x0 + s0 #[2,5]

        x = cp.Variable(n)
        z = cp.Variable(m)
        w = cp.Parameter(n)
        w.value=np.ones(n)
        X = cp.Variable((m,n))
        Y = cp.Parameter((m,n))

        constraint1 = (A @ x + z <= b) #Arbitrary constraint
        constraint2 = (z==0) #Equality
        constraint3 = (w@x <= 1) # <=
        constraint4 = (w@x >= 1) # >=
        constraint5 = 5*cp.norm(A@x) <= 1 #Unary operation
        constraint6 = X@Y.T >= 0
        constraint7 = X@Y.T <= 0
        X@Y.T

        exp1 = TorchExpression(constraint1).torch_expression
        exp2 = TorchExpression(constraint2).torch_expression
        exp3 = TorchExpression(constraint3).torch_expression
        exp4 = TorchExpression(constraint4).torch_expression
        exp5 = TorchExpression(constraint5).torch_expression
        exp6 = TorchExpression(constraint6).torch_expression
        exp7 = TorchExpression(constraint7).torch_expression

        x_test = torch.tensor([1,2,3], dtype=float)
        z_test = torch.zeros(m, dtype=float)
        w_test = torch.tensor([-1,0,1], dtype=float)
        T1 = torch.ones((m,n), dtype=float)
        T2 = torch.ones((m,n), dtype=float)

        test1 = exp1(x_test, z_test)
        test2 = exp2(z_test)
        test3 = exp3(w_test, x_test)
        test4 = exp4(w_test, x_test)
        test5 = exp5(x_test)
        test6 = exp6(T1, T2)
        test7 = exp7(T1, T2)

        self.assertTrue(all(test1==torch.tensor([3, -3])))
        self.assertTrue(all(test2==torch.tensor([0, 0])))
        self.assertTrue(all(test3==torch.tensor([1])))
        self.assertTrue(all(test4==torch.tensor([-1])))
        self.assertTrue(np.isclose(test5, 25.9258))
        self.assertTrue((test6==-n*torch.ones((m,m))).all())
        self.assertTrue((test7==n*torch.ones((m,m))).all())

class TestGenTorchExpAdvanced(unittest.TestCase):
    """ Unit tests for gen_torch_exp"""

    def setUp(self) -> None:
        #Tests the functionality of gen_torch_exp
        self.n = 3
        self.m = 2
        self.x = cp.Variable(self.n)
        self.w = cp.Parameter(self.n)
        self.w.value=np.ones(self.n)
        self.Q = np.array([[2,2,1],[1,-1,2],[-1,-1,1]]) #3x3
        self.a = 3*np.ones(self.n)
        self.t1 = torch.randn(self.n)
        self.t2 = torch.randn(self.n)
        self.T1 = torch.ones((self.m,self.n), dtype=torch.float64) #2x3
        self.T2 = torch.ones((self.m,self.n), dtype=torch.float64) #2x3
        self.X = cp.Variable((self.m,self.n))
        self.Y = cp.Parameter((self.m,self.n))
        self.c = cp.Constant(self.n)

    def test_gen_torch_exp(self):
        exp1  = self.x+self.w+self.a+self.x+self.w
        exp2  = self.x+self.w+self.a+self.x@self.w+self.x
        exp3  = cp.norm(self.Q@self.x+self.w+self.a)
        exp4  = self.x-self.w
        exp5  = self.w-self.x
        exp6  = self.X@self.Y.T
        exp7  = self.x@(self.w+self.w+self.w)
        exp8  = self.x
        exp9  = self.w
        exp10 = self.c
        exp11 = self.x+2*self.w+3*self.c

        torch_exp1 = TorchExpression(exp1).torch_expression
        torch_exp2 = TorchExpression(exp2).torch_expression
        torch_exp3 = TorchExpression(exp3).torch_expression
        torch_exp4 = TorchExpression(exp4).torch_expression
        torch_exp5 = TorchExpression(exp5).torch_expression
        torch_exp6 = TorchExpression(exp6).torch_expression
        torch_exp7 = TorchExpression(exp7).torch_expression
        torch_exp8 = TorchExpression(exp8).torch_expression
        torch_exp9 = TorchExpression(exp9).torch_expression
        torch_exp10 = TorchExpression(exp10).torch_expression
        torch_exp11_unordered = TorchExpression(exp11).torch_expression
        torch_exp11 = TorchExpression(exp11, provided_vars_list=[self.w, self.x]).torch_expression

        test1  = torch_exp1(5*torch.ones(self.n, dtype=torch.float64),
                            torch.tensor([1.,2.,3.], dtype=torch.float64))
        test2  = torch_exp2(1*torch.ones(self.n, dtype=torch.float64),
                            torch.tensor([1.,2.,3.], dtype=torch.float64))
        test3  = torch_exp3(2*torch.ones(self.n, dtype=torch.float64),
                            torch.tensor([2.,1.,2.], dtype=torch.float64))
        test4  = torch_exp4(self.t1, self.t2)
        test5  = torch_exp5(self.t1, self.t2)
        test6  = torch_exp6(self.T1, self.T2)
        test7  = torch_exp7(torch.tensor(self.t1), torch.tensor(self.t2))
        test8  = torch_exp8(self.t1)
        test9  = torch_exp9(self.t1)
        test10 = torch_exp10()
        test11_unordered = torch_exp11_unordered(self.t1,self.t2)
        test11 = torch_exp11(self.t1, self.t2)

        self.assertTrue(all(test1==torch.tensor([15., 17., 19.])))
        self.assertTrue(all(test2==torch.tensor([12, 13, 14])))
        self.assertTrue(np.isclose(test3, 17.2626))
        #Variables and parameters are treated similarly
        self.assertTrue(all(np.isclose(test4, test5))) 
        self.assertTrue((test6==self.n*torch.ones((self.m,self.m))).all())
        self.assertTrue(torch.all(test7==torch.tensor(self.t1)@(3*torch.tensor(self.t2))).item())
        self.assertTrue(torch.all(self.t1==test8))
        self.assertTrue(torch.all(self.t1==test9))
        self.assertTrue(torch.all(test10==self.n).item())
        self.assertTrue(torch.all(test11_unordered==self.t1+2*self.t2+3*self.c.value).item())
        self.assertTrue(torch.all(test11==self.t2+2*self.t1+3*self.c.value).item())

class TestSpecialConstraints(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 5
        self.x = cp.Variable(self.n)
        self.y = cp.Parameter(self.n)
        self.a = 8
        self.b = 4
        self.c = cp.Constant(7)
        self.exp1 = self.x
        self.exp2 = self.a*self.x + self.b*self.y + self.c
        self.t1 = torch.randn(self.n)
        self.t2 = torch.randn(self.n)
        self.t2_exp = (self.a*self.t1+self.b*self.t2+self.c.value).float()
    
    def test_nonpos(self) -> None:
        tch_exp1 = TorchExpression(NonPos(self.exp1)).torch_expression
        tch_exp2 = TorchExpression(NonPos(self.exp2), dtype=torch.float32).torch_expression

        test1 = tch_exp1(self.t1)
        test2 = tch_exp2(self.t1, self.t2)

        torch.testing.assert_close(test1, self.t1)
        torch.testing.assert_close(test2, self.t2_exp)

    def test_nonneg(self) -> None:
        tch_exp1 = TorchExpression(NonNeg(self.exp1)).torch_expression
        tch_exp2 = TorchExpression(NonNeg(self.exp2), dtype=torch.float32).torch_expression

        test1 = tch_exp1(self.t1)
        test2 = tch_exp2(self.t1, self.t2)

        torch.testing.assert_close(test1, -self.t1)
        torch.testing.assert_close(test2, -self.t2_exp)

    def test_zero(self) -> None:
        tch_exp1 = TorchExpression(Zero(self.exp1)).torch_expression
        tch_exp2 = TorchExpression(Zero(self.exp2), dtype=torch.float32).torch_expression

        test1 = tch_exp1(self.t1)
        test2 = tch_exp2(self.t1, self.t2)

        torch.testing.assert_close(test1, self.t1)
        torch.testing.assert_close(test2, self.t2_exp)

class TestDtype(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 5
        self.c = cp.Constant(self.n)

    def test_dtypes(self):
        for dtype in [torch.float64, torch.float32, torch.int64, torch.int32, torch.int16,\
                        torch.int8]:
            tch_exp = TorchExpression(self.c, dtype=dtype).torch_expression
            test = tch_exp()
            self.assertTrue(torch.all(test==torch.Tensor([self.n])).all())
            self.assertTrue(test.dtype==dtype)
