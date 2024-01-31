import unittest

import cvxpy as cp
import torch

from lropt import Ellipsoidal
from lropt.batch_dotproduct import ElementwiseDotProduct
from lropt.uncertain import UncertainParameter


class TestElementwiseDotproduct(unittest.TestCase):

    def test_elementwise_dotproduct(self):
        n = 3
        x = cp.Variable(n)
        uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
        u = UncertainParameter(n,uncertainty_set=uncertainty_set)

        expr1 = x@u
        expr2 = (x+1)@(2*u)

        expr1 = ElementwiseDotProduct.matmul_to_elementwise_dotproduct(expr1)
        expr2 = ElementwiseDotProduct.matmul_to_elementwise_dotproduct(expr2)
        torch_expr1, _ = expr1.gen_torch_exp()
        torch_expr2, _ = expr2.gen_torch_exp()
        
        a = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
        b = torch.tensor([[1.,2.,1.],[-1.,-2.,5.]])
        c = torch.tensor([[1.,2.,3.],[3.,4.,1.],[-1.,-2.,-1.]])
        d = torch.tensor([[5.,-1.,2.],[0.,2.,1.],[-1.,0.,1.]])

        res1 = torch_expr1(a,b)
        res2 = torch_expr1(c,d)
        res3 = torch_expr2(a,b)
        res4 = torch_expr2(c,d)
        
        self.assertTrue(torch.all(res1==torch.tensor([8,16])))
        self.assertTrue(torch.all(res2==torch.tensor([9,9,0])))
        self.assertTrue(torch.all(res3==torch.tensor([24,36])))
        self.assertTrue(torch.all(res4==torch.tensor([30,24,0])))