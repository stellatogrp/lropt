import unittest

import cvxpy as cp
import numpy as np
import torch
from cvxpy.utilities.torch_utils import tensor_reshape_fortran

from lropt import Ellipsoidal
from lropt.train.batch import batchify
from lropt.uncertain_parameter import UncertainParameter

torch.manual_seed(1234)

def reshape_tensor(x: torch.Tensor, shape: tuple[int], order: str, batch: bool):
    """
    This functions reshapes x into shape. Supports order="F" (Fortran) order.
    """
    if order=="F":
        if batch:
            batch_size = x.shape[0]
            return (torch.stack([tensor_reshape_fortran(x[b, :], shape=shape[1:]) \
                                    for b in range(batch_size)]))
        return tensor_reshape_fortran(x, shape)
    else:
        return x.reshape(shape)
    

def _check_expr(test, expr, input, desired_output):
        """
        This is an internal function that helps automate the tests.
        """
        expr = batchify(expr)
        torch_expr, _ = expr.gen_torch_exp()
        output = torch_expr(input)
        test.assertTrue(torch.all(output==desired_output))

class TestElementwiseDotproduct(unittest.TestCase):
    def test_elementwise_dotproduct(self):
        n = 3
        x = cp.Variable(n)
        uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
        u = UncertainParameter(n,uncertainty_set=uncertainty_set)

        expr1 = x@u
        expr2 = (x+1)@(2*u)

        expr1 = batchify(expr1)
        expr2 = batchify(expr2)
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
                expr_vec_right = batchify(expr_vec_right)
                expr_vec_left = batchify(expr_vec_left)
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
                expr = batchify(expr)
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
                expr = batchify(expr)
                torch_expr, _ = expr.gen_torch_exp()
                res = torch_expr(*args)
                if (not left_batch) and (not right_batch):
                    self.assertTrue(torch.all(res==n).item())
                else:
                    self.assertTrue(torch.all(res==n*torch.ones((b))).item())

class TestSlicing(unittest.TestCase):
    def setUp(self):
        self.n = 7 #Dimension
        self.k = 2 #Multiplicative constant
        self.j = 3 #Additive constnat
        self.b = 5 #Batch size
        self.x = cp.Variable(self.n)
        self.y = cp.Variable((self.n,self.n))
        self.expr0 = self.x
        self.expr1 = self.k*self.x
        self.expr2 = self.j+self.y
        self.input_vec = torch.randn(self.n)
        self.input_mat = torch.randn((self.n, self.n))
        self.input_vec_batch = torch.randn((self.b, self.n))
        self.input_mat_batch = torch.randn((self.b, self.n, self.n))

    def test_no_slicing(self):
        #No batch
        _check_expr(self, self.expr0, self.input_vec, self.input_vec)
        _check_expr(self, self.expr1, self.input_vec, self.k*self.input_vec)
        _check_expr(self, self.expr2, self.input_mat, self.j+self.input_mat)
        #Batch
        _check_expr(self, self.expr0, self.input_vec_batch, self.input_vec_batch)
        _check_expr(self, self.expr1, self.input_vec_batch, self.k*self.input_vec_batch)
        _check_expr(self, self.expr2, self.input_mat_batch, self.j+self.input_mat_batch)

    def test_single_slice(self):
        #No batch
        _check_expr(self, self.expr0[1], self.input_vec, self.input_vec[1])
        _check_expr(self, self.expr0[-5:-2], self.input_vec, self.input_vec[-5:-2])
        _check_expr(self, self.expr1[-1], self.input_vec, self.k*self.input_vec[-1])
        _check_expr(self, self.expr1[:3], self.input_vec, self.k*self.input_vec[:3])
        _check_expr(self, self.expr1[slice(0, self.n, 2)], self.input_vec,
                         self.k*self.input_vec[slice(0, self.n, 2)])
        _check_expr(self, self.expr2[-1,:], self.input_mat, self.j+self.input_mat[-1,:])
        _check_expr(self, self.expr2[:,0], self.input_mat, self.j+self.input_mat[:,0])
        _check_expr(self, self.expr2[:,:4], self.input_mat, self.j+self.input_mat[:,:4])
        #Batch
        _check_expr(self, self.expr0[1], self.input_vec_batch, self.input_vec_batch[:,1])
        _check_expr(self, self.expr0[-3], self.input_vec_batch, self.input_vec_batch[:,-3])
        _check_expr(self, self.expr1[-1], self.input_vec_batch, self.k*self.input_vec_batch[:,-1])
        _check_expr(self, self.expr1[0], self.input_vec_batch, self.k*self.input_vec_batch[:,0])
        _check_expr(self, self.expr1[3:7], self.input_vec_batch, self.k*self.input_vec_batch[:,3:7])
        _check_expr(self, self.expr1[slice(1, self.n, 3)],
                         self.input_vec_batch, self.k*self.input_vec_batch[:,slice(1, self.n, 3)])
        _check_expr(self, self.expr2[-1,:], self.input_mat_batch,
                         self.j+self.input_mat_batch[:,-1,:])
        _check_expr(self, self.expr2[:,0], self.input_mat_batch, self.j+self.input_mat_batch[:,:,0])
        _check_expr(self, self.expr2[-5:,:], self.input_mat_batch,
                         self.j+self.input_mat_batch[:,-5:,:])
        _check_expr(self, self.expr2[:-3,:], self.input_mat_batch,
                         self.j+self.input_mat_batch[:,:-3,:])

    def test_double_slice(self):
        #No batch
        _check_expr(self, self.expr2[-2,0], self.input_mat, self.j+self.input_mat[-2,0])
        _check_expr(self, self.expr2[3,4], self.input_mat, self.j+self.input_mat[3,4])
        _check_expr(self, self.expr2[0,-3], self.input_mat, self.j+self.input_mat[0,-3])
        _check_expr(self, self.expr2[1:-1,2:4], self.input_mat, self.j+self.input_mat[1:-1,2:4])
        _check_expr(self, self.expr2[slice(3, 6, 2),-3:], self.input_mat,
                         self.j+self.input_mat[slice(3, 6, 2),-3:])
        #Batch
        _check_expr(self, self.expr2[-4,3], self.input_mat_batch,
                         self.j+self.input_mat_batch[:,-4,3])
        _check_expr(self, self.expr2[1,1], self.input_mat_batch, self.j+self.input_mat_batch[:,1,1])
        _check_expr(self, self.expr2[:4,6], self.input_mat_batch,
                         self.j+self.input_mat_batch[:,:4,6])
        _check_expr(self, self.expr2[-3:,slice(2, 7, 3)],
            self.input_mat_batch, self.j+self.input_mat_batch[:,-3:,slice(2, 7, 3)])
        _check_expr(self, self.expr2[slice(None, None, None),slice(0, self.n, 1)],
            self.input_mat_batch,
            self.j+self.input_mat_batch[:,slice(None, None, None),slice(0, self.n, 1)])

class TestReshape(unittest.TestCase):
    def setUp(self):
        self.n = 6 #Dimension
        self.p = 8 #Second dimension
        self.k = 2 #Multiplicative constant
        self.j = 3 #Additive constnat
        self.b = 5 #Batch size
        self.x = cp.Variable(self.n)
        self.y = cp.Variable((self.p,self.n))
        self.expr0 = self.x
        self.expr1 = self.j + self.k*self.x
        self.expr2 = self.k*self.y
        self.input_vec = torch.randn(self.n)
        self.input_mat = torch.randn((self.p, self.n))
        self.input_vec_batch = torch.randn((self.b, self.n))
        self.input_mat_batch = torch.randn((self.b, self.p, self.n))

    def test_reshape(self):
        for order in ["C", "F"]:
            #No batch
            _check_expr(self, cp.reshape(self.expr0, (self.n//2, 2), order=order), self.input_vec,
                        reshape_tensor(self.input_vec, (self.n//2, 2), order=order, batch=False))
            _check_expr(self, cp.reshape(self.expr1, (2, self.n//2), order=order), self.input_vec,
                        reshape_tensor(self.j+self.k*self.input_vec, (2, self.n//2), order=order,
                        batch=False))
            _check_expr(self, cp.reshape(self.expr2, (self.n, self.p), order=order), self.input_mat,
                        reshape_tensor(self.k*self.input_mat, (self.n, self.p), order=order,
                        batch=False))
            #Batch
            _check_expr(self, cp.reshape(self.expr0, (self.n//2, 2), order=order),
                        self.input_vec_batch, reshape_tensor(self.input_vec_batch,
                        (self.b, self.n//2, 2), order=order, batch=True))
            _check_expr(self, cp.reshape(self.expr1, (2, self.n//2), order=order),
                        self.input_vec_batch, reshape_tensor(self.j+self.k*self.input_vec_batch,
                        (self.b, 2, self.n//2), order=order, batch=True))
            _check_expr(self, cp.reshape(self.expr2, (self.n, self.p), order=order),
                        self.input_mat_batch, reshape_tensor(self.k*self.input_mat_batch,
                        (self.b, self.n, self.p), order=order, batch=True))

#TODO: Add tests for BatchedHstack, BatchedVstack, BatchedAddExpression
