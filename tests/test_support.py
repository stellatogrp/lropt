import unittest

import numpy as np

from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.box import Box
from lropt.uncertainty_sets.budget import Budget

# from lropt.robust_problem import RobustProblem
from lropt.uncertainty_sets.ellipsoidal import Ellipsoidal
from lropt.uncertainty_sets.mro import MRO
from lropt.uncertainty_sets.norm import Norm
from lropt.uncertainty_sets.polyhedral import Polyhedral
from lropt.uncertainty_sets.uncertainty_set import SUPPORT_TYPE

# from tests.settings import TESTS_ATOL as ATOL
# from tests.settings import TESTS_RTOL as RTOL


class TestSupport(unittest.TestCase):

    def setUp(self):
        """Setup basic problem"""
        self.n = 5
        np.random.seed(seed=12345)

    def _check_condition(self, uncertainty_class, ub, lb, eq, c, d, c_check_empty, d_check_empty,
                         c_check_existing, d_check_existing):
        #Empty self._c and self._d
        if uncertainty_class!=Polyhedral:
            uncertainty_set = uncertainty_class()
            uncertainty_set.add_support_type(ub, self.n, SUPPORT_TYPE.UPPER_BOUND)
            uncertainty_set.add_support_type(lb, self.n, SUPPORT_TYPE.LOWER_BOUND)
            uncertainty_set.add_support_type(eq, self.n, SUPPORT_TYPE.EQUALITY)
            self.assertTrue(np.all(uncertainty_set._c == c_check_empty))
            self.assertTrue(np.all(uncertainty_set._d == d_check_empty))

        #Concatentated version
        uncertainty_set = uncertainty_class(c=c, d=d)
        uncertainty_set.add_support_type(ub, self.n, SUPPORT_TYPE.UPPER_BOUND)
        uncertainty_set.add_support_type(lb, self.n, SUPPORT_TYPE.LOWER_BOUND)
        uncertainty_set.add_support_type(eq, self.n, SUPPORT_TYPE.EQUALITY)
        self.assertTrue(np.all(uncertainty_set._c == c_check_existing))
        self.assertTrue(np.all(uncertainty_set._d == d_check_existing))

    def test_all_none(self):
        c = np.eye(self.n)
        d = np.ones(self.n) 
        self._check_condition(uncertainty_class=Norm, ub=None, lb=None, eq=None, c=c, d=d,
                              c_check_empty=None, d_check_empty=None,
                              c_check_existing=c, d_check_existing=d)
    
    def test_vector_bounds(self):
        c = np.eye(self.n)
        d = np.ones(self.n)
        ub = np.ones(self.n)
        lb = np.zeros(self.n)
        eq = None
        c_check_empty = np.vstack([np.eye(self.n),-np.eye(self.n)])
        d_check_empty = np.hstack([ub,-lb])
        c_check_existing = np.vstack([c,c_check_empty])
        d_check_existing = np.hstack([d,d_check_empty])
        self._check_condition(uncertainty_class=Ellipsoidal, ub=ub, lb=lb, eq=eq, c=c, d=d,
                              c_check_empty=c_check_empty, d_check_empty=d_check_empty,
                              c_check_existing=c_check_existing, d_check_existing=d_check_existing)
    
    def test_scalars(self):
        c = np.random.randn(self.n, self.n)
        d = np.random.uniform(size=self.n)
        ub = 3
        lb = -np.ones(self.n)
        eq = 8
        c_check_empty = np.vstack([np.eye(self.n),-np.eye(self.n),np.ones(self.n),-np.ones(self.n)])
        d_check_empty = np.hstack([np.ones(self.n)*ub,-lb,eq,-eq])
        c_check_existing = np.vstack([c,c_check_empty])
        d_check_existing = np.hstack([d,d_check_empty])
        self._check_condition(uncertainty_class=Box, ub=ub, lb=lb, eq=eq, c=c, d=d,
                              c_check_empty=c_check_empty, d_check_empty=d_check_empty,
                              c_check_existing=c_check_existing, d_check_existing=d_check_existing)
        
    def test_vector_equality(self):
        c = np.random.randn(self.n, self.n)
        d = np.random.uniform(size=self.n)
        ub = None
        lb = -3
        eq = 8*np.ones(self.n)
        c_check_empty = np.vstack([-np.eye(self.n),np.eye(self.n),-np.eye(self.n)])
        d_check_empty = np.hstack([-np.ones(self.n)*lb,eq,-eq])
        c_check_existing = np.vstack([c,c_check_empty])
        d_check_existing = np.hstack([d,d_check_empty])
        self._check_condition(uncertainty_class=Polyhedral, ub=ub, lb=lb, eq=eq, c=c, d=d,
                              c_check_empty=c_check_empty, d_check_empty=d_check_empty,
                              c_check_existing=c_check_existing, d_check_existing=d_check_existing)
                  
    def _check_condition_param(self,uncertainty_class, ub, lb, eq, c, d,
                            c_check_empty, d_check_empty,
                            c_check_existing, d_check_existing):

        def _safe_create_uncertain_parameter(uncertainty_class, n, uncertainty_set):
            if uncertainty_class == MRO:
                data = np.random.rand(n, n)
                return UncertainParameter(n, uncertainty_set=uncertainty_set, data=data)
            else:
                return UncertainParameter(n, uncertainty_set=uncertainty_set)

        #Empty self._c and self._d
        if uncertainty_class!=Polyhedral:
            uncertainty_set = uncertainty_class(ub=ub, lb=lb, eq=eq)
            #u = UncertainParameter(self.n, uncertainty_set=uncertainty_set)
            u = _safe_create_uncertain_parameter(uncertainty_class, self.n, uncertainty_set)
            self.assertTrue(np.all(u.uncertainty_set._c == c_check_empty))
            self.assertTrue(np.all(u.uncertainty_set._d == d_check_empty))

        #Concatentated version
        uncertainty_set = uncertainty_class(c=c, d=d,ub=ub, lb=lb, eq=eq)
        #u = UncertainParameter(self.n, uncertainty_set=uncertainty_set)
        u = _safe_create_uncertain_parameter(uncertainty_class, self.n, uncertainty_set)
        self.assertTrue(np.all(u.uncertainty_set._c == c_check_existing))
        self.assertTrue(np.all(u.uncertainty_set._d == d_check_existing))

    def test_uncertain_param_none(self):
        c = np.eye(self.n)
        d = -np.ones(self.n) 
        self._check_condition_param(uncertainty_class=Budget, ub=None, lb=None, eq=None, c=c, d=d,
                              c_check_empty=None, d_check_empty=None,
                              c_check_existing=c, d_check_existing=d)

    def test_uncertain_param_scalars(self):
        c = np.random.randn(self.n, self.n)
        d = np.random.uniform(size=self.n)
        ub = 5
        lb = -3
        eq = 1
        c_check_empty = np.vstack([np.eye(self.n),-np.eye(self.n),np.ones(self.n),-np.ones(self.n)])
        d_check_empty = np.hstack([np.ones(self.n)*ub,-np.ones(self.n)*lb,eq,-eq])
        c_check_existing = np.vstack([c,c_check_empty])
        d_check_existing = np.hstack([d,d_check_empty])
        self._check_condition_param(uncertainty_class=Norm, ub=ub, lb=lb, eq=eq, c=c, d=d,
                              c_check_empty=c_check_empty, d_check_empty=d_check_empty,
                              c_check_existing=c_check_existing, d_check_existing=d_check_existing)
        
    def test_uncertain_param_scalar_box(self):
        c = np.random.randn(self.n, self.n)
        d = np.random.uniform(size=self.n)
        ub = 10
        lb = -0
        eq = 1
        c_check_empty = np.vstack([np.eye(self.n),-np.eye(self.n),np.ones(self.n),-np.ones(self.n)])
        d_check_empty = np.hstack([np.ones(self.n)*ub,-np.ones(self.n)*lb,eq,-eq])
        c_check_existing = np.vstack([c,c_check_empty])
        d_check_existing = np.hstack([d,d_check_empty])
        self._check_condition_param(uncertainty_class=MRO, ub=ub, lb=lb, eq=eq, c=c, d=d,
                              c_check_empty=c_check_empty, d_check_empty=d_check_empty,
                              c_check_existing=c_check_existing, d_check_existing=d_check_existing)