import unittest

import cvxpy as cp
import numpy as np
import torch
from cvxtorch import TorchExpression
from cvxtorch.utils.torch_utils import tensor_reshape_fortran

from cvxro import Ellipsoidal
from cvxro.train.batch import batchify
from cvxro.uncertain_parameter import UncertainParameter

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
        torch_expr = TorchExpression(expr).torch_expression
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

        torch_expr1 = TorchExpression(expr1).torch_expression
        torch_expr2 = TorchExpression(expr2).torch_expression

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
                torch_expr_vec_right = TorchExpression(expr_vec_right).torch_expression
                torch_expr_vec_left = TorchExpression(expr_vec_left).torch_expression
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
                torch_expr = TorchExpression(expr).torch_expression
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
                torch_expr = TorchExpression(expr).torch_expression
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

class TestBatchedAddExpression(unittest.TestCase):
        def test_basic_addition(self):
            n = 3
            x = cp.Variable(n)
            uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
            u = UncertainParameter(n, uncertainty_set=uncertainty_set)

            expr1 = x + u
            expr2 = (x + 1) + (2 * u)
            expr3 = (x * 2) + u

            expr1 = batchify(expr1)
            expr2 = batchify(expr2)
            expr3 = batchify(expr3)
            torch_expr1 = TorchExpression(expr1).torch_expression
            torch_expr2 = TorchExpression(expr2).torch_expression
            torch_expr3 = TorchExpression(expr3).torch_expression


            a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
            b = torch.tensor([[1., 2., 1.], [-1., -2., 5.]])


            res1 = torch_expr1(a,b)
            res2 = torch_expr2(a,b)
            self.assertTrue(torch.all(res1==torch.tensor([[2., 4., 4.], [3., 3., 11.]])))
            self.assertTrue(torch.all(res2 == torch.tensor([[4., 7., 6.], [3., 2., 17.]]) ))

            c = torch.tensor([[-1., -2., -3.], [-4., -5., -6.]])
            d = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

            res3 = torch_expr1(c, d)
            self.assertTrue(torch.allclose(res3, torch.tensor([[0., 0., 0.], [0., 0., 0.]])))

            res3 = torch_expr3(c, d)
            self.assertTrue(torch.allclose(res3, torch.tensor([[-1., -2., -3.], [-4., -5., -6.]])))

        def test_zero_tensors(self):
            n = 3
            x = cp.Variable(n)
            uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
            u = UncertainParameter(n, uncertainty_set=uncertainty_set)

            expr = x + u
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression

            a = torch.zeros((2, n))
            b = torch.zeros((2, n))


            res = torch_expr(a, b)
            self.assertTrue(torch.allclose(res, torch.zeros((2, n))))

        def test_batched_addition(self):
            n = 2
            b = 3

            a_batch = torch.ones((b, n), dtype=torch.double)
            b_batch = torch.ones((b, n), dtype=torch.double)

            expected_result = torch.ones((b, n), dtype=torch.double) * 2

            x = cp.Variable(n)
            uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
            u = UncertainParameter(n, uncertainty_set=uncertainty_set)

            expr = x + u
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression

            res = torch_expr(a_batch, b_batch)

            self.assertTrue(torch.allclose(res, expected_result))

        def test_batched_addition_with_zero_tensors(self):
            n = 3
            b = 4
            zero_batch = torch.zeros((b, n), dtype=torch.double)
            expected_result = torch.zeros((b, n), dtype=torch.double)
            x = cp.Variable(n)
            uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
            u = UncertainParameter(n, uncertainty_set=uncertainty_set)
            expr = x + u
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression
            res = torch_expr(zero_batch, zero_batch)
            self.assertTrue(torch.allclose(res, expected_result))

        def test_batched_addition_diff_sizes(self):
            n = 3

            for b in [2, 4, 6]:
                a_batch = torch.tensor([[1., 2., 3.]] * b, dtype=torch.double)
                b_batch = torch.tensor([[1., 1., 1.]] * b, dtype=torch.double)
                expected_result = torch.tensor([[2., 3., 4.]] * b, dtype=torch.double)
                x = cp.Variable(n)
                uncertainty_set = Ellipsoidal(rho=5, p=1, lb=0, ub=10)
                u = UncertainParameter(n, uncertainty_set=uncertainty_set)
                expr = x + u
                expr = batchify(expr)
                torch_expr = TorchExpression(expr).torch_expression
                res = torch_expr(a_batch, b_batch)
                self.assertTrue(torch.allclose(res, expected_result))

class TestBatchedVstack(unittest.TestCase):
    def test_basic_vstack(self):
        n = 3
        x = cp.Variable(n)
        y = cp.Variable(n)

        expr1 = cp.vstack([x, y])
        expr1 = batchify(expr1)

        torch_expr1 = TorchExpression(expr1).torch_expression

        a = torch.tensor([[1., 2., 3.]])
        b = torch.tensor([[4., 5., 6.]])

        expected_result = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

        res1 = torch_expr1(a, b)
        self.assertTrue(torch.allclose(res1, expected_result))

    def test_diff_batch_sizes(self):
        n = 3
        for b in [2, 4, 6]:
            x_batch = torch.tensor([[1., 2., 3.]] * b, dtype=torch.double)
            y_batch = torch.tensor([[4., 5., 6.]] * b, dtype=torch.double)

            expected_result = torch.cat([x_batch, y_batch], dim=0)

            x = cp.Variable(n)
            y = cp.Variable(n)

            expr = cp.vstack([x, y])
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression

            res = torch_expr(x_batch, y_batch)
            self.assertTrue(torch.allclose(res, expected_result))

    def test_batched_vstack_zero_values(self):
        n = 3
        for b in [2, 4, 6]:
            x_batch = torch.zeros((b, n), dtype=torch.double)
            y_batch = torch.zeros((b, n), dtype=torch.double)

            expected_result = torch.cat([x_batch, y_batch], dim=0)

            x = cp.Variable(n)
            y = cp.Variable(n)

            expr = cp.vstack([x, y])
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression

            res = torch_expr(x_batch, y_batch)
            self.assertTrue(torch.allclose(res, expected_result))

    def test_large_batch(self):
        n = 3
        b = 10000  # Large batch size
        x_batch = torch.ones((b, n), dtype=torch.double)
        y_batch = torch.ones((b, n), dtype=torch.double) * 2

        expected_result = torch.cat([x_batch, y_batch], dim=0)

        x = cp.Variable(n)
        y = cp.Variable(n)

        expr = cp.vstack([x, y])
        expr = batchify(expr)
        torch_expr = TorchExpression(expr).torch_expression

        res = torch_expr(x_batch, y_batch)
        self.assertTrue(torch.allclose(res, expected_result))

    def test_zero_and_identity_matrices(self):
        n = 4
        x = torch.zeros((2, n), dtype=torch.double)
        y = torch.eye(2, n, dtype=torch.double)

        expected_result = torch.cat([x, y], dim=0)

        x_var = cp.Variable(n)
        y_var = cp.Variable(n)

        expr = cp.vstack([x_var, y_var])
        expr = batchify(expr)
        torch_expr = TorchExpression(expr).torch_expression

        res = torch_expr(x, y)
        self.assertTrue(torch.allclose(res, expected_result))

    def test_mixed_data_types(self):
        n = 3
        x = cp.Variable(n)
        y = cp.Variable(n)

        expr1 = cp.vstack([x, y])
        expr1 = batchify(expr1)

        torch_expr1 = TorchExpression(expr1).torch_expression

        a = torch.tensor([[1., 2., 3.]], dtype=torch.float32)
        b = torch.tensor([[4., 5., 6.]], dtype=torch.float64)

        expected_result = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)

        res1 = torch_expr1(a, b)
        self.assertTrue(torch.allclose(res1, expected_result, atol=1e-6))

class TestBatchedHStack(unittest.TestCase):
    def test_basic_hstack(self):
        n = 3
        m = 2

        x = cp.Variable((n, m))
        y = cp.Variable((n, m))

        expr1 = cp.hstack([x, y])
        expr1 = batchify(expr1)

        torch_expr1 = TorchExpression(expr1).torch_expression

        a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        b = torch.tensor([[7., 8., 9.], [10., 11., 12.]])

        expected_result = torch.tensor([[1., 2., 3., 7., 8., 9.], [4., 5., 6., 10., 11., 12.]])

        res1 = torch_expr1(a, b)
        self.assertTrue(torch.allclose(res1, expected_result))

    def test_different_batch_sizes(self):
        n = 3
        m = 2

        for b in [2, 4, 6]:
            x_batch = torch.tensor([[1., 2.]] * b, dtype=torch.double)
            y_batch = torch.tensor([[4., 5.]] * b, dtype=torch.double)
            expected_result = torch.cat([x_batch, y_batch], dim=1)

            x = cp.Variable((n, m))
            y = cp.Variable((n, m))
            expr = cp.hstack([x, y])
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression
            res = torch_expr(x_batch, y_batch)
            self.assertTrue(torch.allclose(res, expected_result))

    def test_zero_tensors(self):
        n = 3
        m = 2

        zero_tensor = torch.zeros((n, m), dtype=torch.double)
        x = cp.Variable((n, m))
        y = cp.Variable((n, m))
        expr = cp.hstack([x, y])
        expr = batchify(expr)
        torch_expr = TorchExpression(expr).torch_expression
        res = torch_expr(zero_tensor, zero_tensor)
        expected_result = torch.cat([zero_tensor, zero_tensor], dim=1)
        self.assertTrue(torch.allclose(res, expected_result))

    def test_identity_matrices(self):
        n = 2
        m = 2

        identity_tensor = torch.eye(m, dtype=torch.double).unsqueeze(0).repeat(n, 1, 1) # Batch of 1
        x = cp.Variable((n, m))
        y = cp.Variable((n, m))
        expr = cp.hstack([x, y])
        expr = batchify(expr)
        torch_expr = TorchExpression(expr).torch_expression
        res = torch_expr(identity_tensor.squeeze(0), identity_tensor.squeeze(0))
        expected_result = torch.cat([identity_tensor.squeeze(0), identity_tensor.squeeze(0)], dim=1)
        self.assertTrue(torch.allclose(res, expected_result))

    def test_mixed_values(self):
        for b in [2, 3]:
            x_batch = torch.tensor([[i, i + 1] for i in range(b)], dtype=torch.double)
            y_batch = torch.tensor([[i + 2, i + 3] for i in range(b)], dtype=torch.double)
            expected_result = torch.cat([x_batch, y_batch], dim=1)

            x = cp.Variable((b, 2))
            y = cp.Variable((b, 2))
            expr = cp.hstack([x, y])
            expr = batchify(expr)
            torch_expr = TorchExpression(expr).torch_expression
            res = torch_expr(x_batch, y_batch)
            self.assertTrue(torch.allclose(res, expected_result))

    def test_large_tensors(self):
        large_batch_size = 100
        n = 3

        x_batch = torch.ones((large_batch_size, n), dtype=torch.double)
        y_batch = torch.ones((large_batch_size, n), dtype=torch.double) * 2
        expected_result = torch.cat([x_batch, y_batch], dim=1)

        x = cp.Variable((large_batch_size, n))
        y = cp.Variable((large_batch_size, n))
        expr = cp.hstack([x, y])
        expr = batchify(expr)
        torch_expr = TorchExpression(expr).torch_expression
        res = torch_expr(x_batch, y_batch)
        self.assertTrue(torch.allclose(res, expected_result))
