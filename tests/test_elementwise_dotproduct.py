import unittest

import cvxpy as cp
import numpy as np
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
        e = torch.tensor([1., 2., 3.])
        f = torch.tensor([1., -1., 0.])

        res1 = torch_expr1(a,b)
        res2 = torch_expr1(c,d)
        res3 = torch_expr2(a,b)
        res4 = torch_expr2(c,d)
        res5 = torch_expr1(e,f)
        res6 = torch_expr2(e,f)
        
        self.assertTrue(torch.all(res1==torch.tensor([8,16])))
        self.assertTrue(torch.all(res2==torch.tensor([9,9,0])))
        self.assertTrue(torch.all(res3==torch.tensor([24,36])))
        self.assertTrue(torch.all(res4==torch.tensor([30,24,0])))
        self.assertTrue(torch.all(res5==-1))
        self.assertTrue(torch.all(res6==-2))

    def test_elementwise_batches_vec_mat(self):
        m = 3
        n = 4
        b = 5 #Batch size

        mat_batch = torch.ones((b, m, n), dtype=torch.double)
        mat_unbtc = torch.ones((m,n), dtype=torch.double)
        vec_right_batch = torch.ones((b,n), dtype=torch.double)
        vec_right_unbtc = torch.ones(n, dtype=torch.double)
        vec_left_batch = torch.ones((b,m), dtype=torch.double)
        vec_left_unbtc = torch.ones(m, dtype=torch.double)

        for is_vec_batch in (False, True):
            vec_right = cp.Variable(n) if is_vec_batch else cp.Constant(np.ones(n))
            vec_left = cp.Variable(m) if is_vec_batch else cp.Constant(np.ones(m))
            vec_right_val = vec_right_batch if is_vec_batch else vec_right_unbtc
            vec_left_val = vec_left_batch if is_vec_batch else vec_left_unbtc
            for is_mat_batch in (False, True):
                mat = cp.Variable((m,n)) if is_mat_batch else cp.Constant(np.ones((m,n)))
                mat_val = mat_batch if is_mat_batch else mat_unbtc
                
                right_args = []
                left_args = []
                if is_mat_batch:
                    right_args.append(mat_val)
                    left_args.append(mat_val)
                if is_vec_batch:
                    right_args.append(vec_right_val)
                    left_args.insert(0, vec_left_val)

                #Test this combination
                expr_vec_right = mat@vec_right
                expr_vec_left = vec_left@mat                
                expr_vec_right = ElementwiseDotProduct.matmul_to_elementwise_dotproduct(
                    expr_vec_right)
                expr_vec_left = ElementwiseDotProduct.matmul_to_elementwise_dotproduct(
                    expr_vec_left)
                torch_expr_vec_right, _ = expr_vec_right.gen_torch_exp()
                torch_expr_vec_left, _ = expr_vec_left.gen_torch_exp()
                res_vec_right = torch_expr_vec_right(*right_args)
                res_vec_left = torch_expr_vec_left(*left_args)
                self.assertTrue(torch.all(res_vec_right==n*torch.ones((b,m))).item())
                self.assertTrue(torch.all(res_vec_left==m*torch.ones((b,n))).item())

    def test_elementwise_batches_mat_mat(self):
        m = 3
        n = 4
        k = 5
        b = 6 #Batch size

        mat_left_batch = torch.ones((b, m, n), dtype=torch.double)
        mat_left_unbtc = torch.ones((m,n), dtype=torch.double)
        mat_right_batch = torch.ones((b,n,k), dtype=torch.double)
        mat_right_unbtc = torch.ones((n,k), dtype=torch.double)

        for left_batch in (False, True):
            mat_left = cp.Variable((m,n)) if left_batch else cp.Constant(np.ones((m,n)))
            mat_left_val = mat_left_batch if left_batch else mat_left_unbtc
            for right_batch in (False, True):
                mat_right = cp.Variable((n,k)) if right_batch else cp.Constant(np.ones((n,k)))
                mat_right_val = mat_right_batch if right_batch else mat_right_unbtc
                
                args = []
                if left_batch:
                    args.append(mat_left_val)
                if right_batch:
                    args.append(mat_right_val)

                #Test this combination
                expr = mat_left@mat_right
                expr = ElementwiseDotProduct.matmul_to_elementwise_dotproduct(expr)
                torch_expr, _ = expr.gen_torch_exp()
                res = torch_expr(*args)
                if (not left_batch) and (not right_batch):
                    self.assertTrue(torch.all(res==n*torch.ones((m,k))).item())
                else:
                    self.assertTrue(torch.all(res==n*torch.ones((b,m,k))).item())

    def test_elementwise_batches_vec_vec(self):
        n = 4
        b = 6 #Batch size

        vec_left_batch = torch.ones((b, n), dtype=torch.double)
        vec_left_unbtc = torch.ones(n, dtype=torch.double)
        vec_right_batch = torch.ones((b, n), dtype=torch.double)
        vec_right_unbtc = torch.ones(n, dtype=torch.double)

        for left_batch in (False, True):
            vec_left = cp.Variable(n) if left_batch else cp.Constant(np.ones(n))
            vec_left_val = vec_left_batch if left_batch else vec_left_unbtc
            for right_batch in (False, True):
                vec_right = cp.Variable(n) if right_batch else cp.Constant(np.ones(n))
                vec_right_val = vec_right_batch if right_batch else vec_right_unbtc
                
                args = []
                if left_batch:
                    args.append(vec_left_val)
                if right_batch:
                    args.append(vec_right_val)

                #Test this combination
                expr = vec_left@vec_right
                expr = ElementwiseDotProduct.matmul_to_elementwise_dotproduct(expr)
                torch_expr, _ = expr.gen_torch_exp()
                res = torch_expr(*args)
                if (not left_batch) and (not right_batch):
                    self.assertTrue(torch.all(res==n).item())
                else:
                    self.assertTrue(torch.all(res==n*torch.ones((b))).item())