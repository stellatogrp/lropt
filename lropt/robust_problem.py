from abc import ABC
from inspect import signature
from typing import Optional

import numpy as np
import pandas as pd
import scipy as sc
import torch
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem
from cvxpy.reductions import Dcp2Cone, Qp2SymbolicQp
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.solving_chain import SolvingChain, construct_solving_chain
from cvxpylayers.torch import CvxpyLayer
from sklearn.model_selection import train_test_split

#from lropt.settings import DTYPE, EPS_LST_DEFAULT, LAYER_SOLVER, OPTIMIZERS, MRO_CASE
import lropt.settings as settings
from lropt.parameter import Parameter
from lropt.remove_uncertain.remove_uncertain import RemoveUncertainParameters
from lropt.shape_parameter import ShapeParameter
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.distribute_uncertain_params import Distribute_Uncertain_Params
from lropt.uncertain_canon.uncertain_chain import UncertainChain
from lropt.uncertainty_sets.mro import MRO


# TODO (Amit): Irina, please go over the functions and update the args descrption where necessary
class TrainLoopStats():
    """
    A class that contains useful statistics for each training iteration

    Attributes:
        step_num:
            Current iteration number
        tot_lagrangian:
            Lagrangian over training set
        testval:
            Evaluation function over test set
        test_vio:
            Probability of constraint violation over test set
        train_vio:
            Probability fo constraint violation over train set
        violation_test:
            Violation of learning constraint over test set
        violation_train:
            Violation of learning constraint over train set
    """
    # TODO (Amit): Can we change the names of the last 4 parameters? They're very confusing.
    # Response (Irina): Yes, we can. I'm not sure what good naming would be. Would prob_vio_test/train, vio_value_test/train work? Or something simpler.

    def __init__(self, step_num, train_flag=True):
        def __value_init__(self):
            """
            This is an internal function that either initiates a tensor or a list
            """
            return [] if not self.train_flag else torch.tensor(0., dtype=float)
        self.step_num = step_num
        self.train_flag = train_flag
        self.tot_lagrangian = __value_init__(self)
        self.testval = __value_init__(self)
        self.test_vio = __value_init__(self)
        self.train_vio = __value_init__(self)
        self.violation_test = __value_init__(self)
        self.violation_train = __value_init__(self)

    def update_stats(self, temp_lagrangian, num_ys, obj_validation, train_vio, test_vio,
                     var_vio, cvar_update):
        """
        This function updates the statistics after each training iteration
        """
        # TODO (Amit): Irina, can we please see which of these is needed
        # Response (Irina): These should all be needed
        self.tot_lagrangian += temp_lagrangian/num_ys
        if not self.train_flag:
            self.testval.append(obj_validation.item())
            self.test_vio.append(test_vio.item())
            self.train_vio.append(train_vio.item())
            self.violation_test.append(var_vio.item())
            self.violation_train.append(cvar_update.item())
        else:  # if self.train_flag
            self.testval += obj_validation.item()
            self.test_vio += test_vio.item()
            self.train_vio += train_vio.item()
            self.violation_test += var_vio.item()
            self.violation_train += cvar_update.item()

    def generate_row(self, num_ys, paramT_tch, lam, alpha, coverage, coverage2,
                     val_dset, eval_set, eps_tch1):
        """
        This function generates a new row with the statistics
        """
        # TODO (Amit): Why do we have Violation and Violations?
        # Also, why some fields are capitilized and some are not?
        # Response (Irina): That was bad naming. These should be the ones in the docstrings. Coverage can also be capitalized.
        row_dict = {
            "Lagrangian_val":   self.tot_lagrangian.item(),
            "Test_val":         self.testval/num_ys,
            "Violations":       self.test_vio/num_ys,
            "Violations_train": self.train_vio/num_ys,
            "Violation_test":   self.violation_test/num_ys,
            "Violation_train":  self.violation_train/num_ys,
            "coverage_train":   coverage.detach().numpy().item()/val_dset.shape[0],
            "coverage_test":    coverage2.detach().numpy().item()/eval_set.shape[0]
        }
        if not self.train_flag:
            # TODO (Amit): I commented these two fields because I believe
            # we no longer want them, please confirm.
            # Response (Irina): Yes.
            #row_dict["Eval_val"]    = totevallagrangian,
            #row_dict["Opt_val"]     = optval,
            row_dict["Eps"] = 1 / eps_tch1[0][0].detach().numpy().copy(),
        else:  # if self.train_flag
            row_dict["step"] = self.step_num,
            row_dict["A_norm"] = np.linalg.norm(
                paramT_tch.detach().numpy().copy()),
            row_dict["lam_list"] = lam.detach().numpy().copy(),
            row_dict["alpha"] = alpha.item(),
            row_dict["alphagrad"] = alpha.grad,
            row_dict["dfnorm"] = np.linalg.norm(paramT_tch.grad),
            row_dict["gradnorm"] = paramT_tch.grad,
        new_row = pd.Series(row_dict)
        return new_row


class GridStats():
    """
    This class contains useful information for grid search
    """

    def __init__(self):
        # TODO (Amit): originally was 9999999. Please confirm.
        self.minval = np.float("inf")
        # Response (Irina): Yes.
        self.var_vals = 0

    def update(self, train_stats, temp_lagrangian, eps_tch1, paramT_tch, var_values):
        """
        This function updates the best stats in the grid search.

        Args:
            train_stats:
                The train stats
            temp_lagrangian:
                Calculated lagrangian
            eps_tch1:
                Epsilon torch
            paramT_tch
                T torch
            var_values
                variance values
        """
        if train_stats.tot_lagrangian <= self.minval:
            self.minval = temp_lagrangian
            self.mineps = eps_tch1.clone()
            self.minT = paramT_tch.clone()
            self.var_vals = var_values


class RobustProblem(Problem):
    """Create a Robust Optimization Problem with uncertain variables"""

    def __init__(
        self, objective, constraints, objective_torch=None, constraints_torch=None,
        eval_torch=None, train_flag=True
    ):
        self._trained = False
        self._values = None
        self._numvars = 0
        super(RobustProblem, self).__init__(objective, constraints)
        self._trained = False
        self._values = None
        self.new_prob = None
        self.inverse_data = None
        self._init = None
        self.train_flag = train_flag

        self.num_ys = self.verify_y_parameters()
        self.f, self.g = objective_torch, constraints_torch
        self.eval = eval_torch if eval_torch is not None else objective_torch
        self.l, self.h, self.num_g = self.fg_to_lh(
            objective_torch, constraints_torch)

    @property
    def trained(self):
        return self._trained

    @property
    def param_values(self):
        return self._values

    def uncertain_parameters(self):
        """Find uncertain parameters"""
        return [v for v in self.parameters() if isinstance(v, UncertainParameter)]

    def y_parameters(self):
        """Find y parameters"""
        return [v for v in self.parameters() if isinstance(v, Parameter)]

    def shape_parameters(self, problem):
        return [v for v in problem.parameters() if isinstance(v, ShapeParameter)]

    def verify_y_parameters(self):
        y_parameters = self.y_parameters()
        num_ys = 1
        if len(y_parameters) > 0:
            num_ys = y_parameters[0].data.shape[0]
        for param in y_parameters:
            if param.data.shape[0] != num_ys:
                raise ValueError("shape inconsistency: num_ys")
        return num_ys

    # TODO (Amit): I think a lot of the following functions need refactoring,
    # I don't know what to change to
    # Also, which of H, G, J, N can be changed to x_func?
    # Response (Irina): Yes, these also need to be tested. I don't think these should be changed to x_func.

    def fg_to_lh(self, f_tch, g_tch):
        """
        Returns l and h function pointers. 
        Each of them takes a single x,y,u triplet (i.e. one instance of each)
        """
        if f_tch is None or g_tch is None:
            return None, None, None
        sig_f = signature(f_tch)
        arg_num = len(sig_f.parameters)
        true_arg_num = (
            len(self.variables()) + len(self.y_parameters()) +
            len(self.uncertain_parameters())
        )
        if arg_num != true_arg_num:
            raise ValueError(
                "incorrect number of user's pytorch function arguments")
        h_funcs = []
        for g in g_tch:
            if len(signature(g).parameters) != true_arg_num:
                raise ValueError(
                    "incorrect number of user's pytorch function arguments")

            def hg(vars, y_params, u_params, alpha, eta, kappa):
                return (
                    torch.maximum(
                        g(*vars, *y_params, *u_params) - alpha,
                        torch.tensor(0.0),
                    ) / eta + alpha - kappa)

            h_funcs.append(hg)

        l_func = f_tch
        return l_func, h_funcs, len(h_funcs)

    def lagrange_objective(self, vars, y_params_mat, u_params_mat):
        J = len(y_params_mat)
        N = len(u_params_mat)

        sum = torch.tensor(0.0, dtype=float)
        for i in range(N):
            for j in range(J):
                sum += self.l(*vars, *y_params_mat[j], *u_params_mat[i])
        expectation = sum / (J * N)
        return expectation

    def lagrange_constraint(self, vars, y_params_mat, u_params_mat, alpha, eta, kappa):
        J = len(y_params_mat)
        N = len(u_params_mat)

        num_g = len(self.h)
        H = torch.zeros(num_g, dtype=float)
        for k, h_k in enumerate(self.h):
            sum = torch.tensor(0.0, dtype=float)
            for i in range(N):
                for j in range(J):
                    sum += h_k(vars, y_params_mat[j],
                               u_params_mat[i], alpha, eta, kappa)
            h_k_expectation = sum / (J * N)
            H[k] = h_k_expectation
        return H

    def Eval(self, vars, y_params_mat, u_params_mat):
        J = len(y_params_mat)
        N = len(u_params_mat)

        sum = torch.tensor(0.0, dtype=float)
        for i in range(N):
            for j in range(J):
                sum += self.eval(*vars, *y_params_mat[j], *u_params_mat[i])
        return sum / (J * N)

    def prob_constr_violation(self, vars, y_params_mat, u_params_mat):
        num_g = len(self.g)
        J = len(y_params_mat)
        N = len(u_params_mat)
        G = torch.zeros((num_g, J, N), dtype=float)

        for k, g_k in enumerate(self.g):
            for i in range(N):
                for j in range(J):
                    G[k, j, i] = g_k(*vars, *y_params_mat[j], *u_params_mat[i])

        g_max = torch.max(G, dim=0)[0]
        g_max_violate = (g_max > 0).float()
        return torch.mean(g_max_violate)

    # helper function for intermediate version
    def _udata_to_lst(self, data):
        num_instances = data.shape[0]
        u_params_mat = []
        for i in range(num_instances):
            u_params_mat.append([data[i, :]])
        return u_params_mat

    def lagrange(self, vars, y_params_mat, u_params_mat, alpha, lam,
                 eta=settings.ETA_LAGRANGIAN_DEFAULT, kappa=settings.KAPPA_LAGRANGIAN_DEFAULT):
        # TODO (Amit): It seems to me that this function does two things: returns the lagrangian
        # and also evaluates. Shouldn't these be two separate functions?
        # Response (Irina): Yes, these should be two separate functions.
        F = self.lagrange_objective(vars, y_params_mat, u_params_mat)
        H = self.lagrange_constraint(
            vars, y_params_mat, u_params_mat, alpha, eta, kappa)
        eval = self.Eval(vars, y_params_mat, u_params_mat)
        prob_constr_violation = self.prob_constr_violation(
            vars, y_params_mat, u_params_mat)
        return F + lam @ H, eval, H, prob_constr_violation

    # create function for only remove_uncertain reduction
    def _construct_chain(
        self,
        solver: Optional[str] = None,
        gp: bool = False,
        enforce_dpp: bool = True,
        ignore_dpp: bool = False,
        solver_opts: Optional[dict] = None,
        canon_backend: str | None = None,
    ) -> SolvingChain:
        """
        Construct the chains required to reformulate and solve the problem.
        In particular, this function
        # finds the candidate solvers
        # constructs the solving chain that performs the
           numeric reductions and solves the problem.

        Args:

        solver : str, optional
            The solver to use. Defaults to ECOS.
        gp : bool, optional
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.
        enforce_dpp : bool, optional
            Whether to error on DPP violations.
        ignore_dpp : bool, optional
            When True, DPP problems will be treated as non-DPP,
            which may speed up compilation. Defaults to False.
        canon_backend : str, optional
            'CPP' (default) | 'SCIPY'
            Specifies which backend to use for canonicalization, which can affect
            compilation time. Defaults to None, i.e., selecting the default
            backend.
        solver_opts: dict, optional
            Additional arguments to pass to the solver.

        Returns:
            A solving chain
        """
        candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
        self._sort_candidate_solvers(candidate_solvers)
        solving_chain = construct_solving_chain(
            self,
            candidate_solvers,
            gp=gp,
            enforce_dpp=enforce_dpp,
            ignore_dpp=ignore_dpp,
            canon_backend=canon_backend,
            solver_opts=solver_opts,
        )
        #
        new_reductions = solving_chain.reductions
        if self.uncertain_parameters():
            # new_reductions = solving_chain.reductions
            # Find position of Dcp2Cone or Qp2SymbolicQp
            for idx in range(len(new_reductions)):
                if type(new_reductions[idx]) in [Dcp2Cone, Qp2SymbolicQp]:
                    # Insert RemoveUncertainParameters before those reductions
                    new_reductions.insert(idx, RemoveUncertainParameters())
                    break
        # return a chain instead (chain.apply, return the problem and inverse data)
        return SolvingChain(reductions=new_reductions)

    def _validate_uncertain_parameters(self):
        """
        This function checks if there are uncertain parameters.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError if there are no uncertain parameters
        """
        if self.uncertain_parameters() is None:
            raise ValueError("The problem has no uncertain parameters")

    def _validate_unc_set_T(self, unc_set):
        """
        This function checks if paramaterT is not empty.

        Args:
            unc_set
                Uncertainty set

        Returns:
            None.

        Raises:
            ValueError if there is no paramT in the uncertainty set
        """

        if unc_set.paramT is None:
            raise ValueError("unc_set.paramT is None")

    def _is_mro_set(self, unc_set):
        """
        This function returns whether we work on MRO set or not.

        Args:

        unc_set
            Uncertainty set

        Returns:

            Boolean result
        """

        return (type(unc_set) == MRO)

    def _canonize_problem(self, override=False):
        """
        This function canonizes a problem and saves it to self.new_prob

        Args:

        override 
            If True, will override current new_prob. If false and new_prob exists, does nothing.

        Returns:

        None
        """

        if (not override) and (self.new_prob):
            return

        # Creating uncertainty reduction and chain
        unc_reductions = []
        if type(self.objective) == Maximize:
            unc_reductions += [FlipObjective()]
        unc_reductions += [RemoveUncertainParameters()]
        newchain = UncertainChain(self, reductions=unc_reductions)

        # TODO (Bart): We should keep inverse data. We can use it to reconstruct the original
        # solution from new_prob. Are we sure we don't need it?
        # Apply the chain
        # prob is the problem without uncertainty
        # The second (unused) returned value is inverse_data
        self.new_prob, self.inverse_data = newchain.apply(self)

    def _gen_init(self, eps, train_set, init_eps, init_A):
        """
        This is an internal function that calculates init.
        Init means different things depending on eps
            it is an internal function not intended to be used publicly.

        Args:

        eps
            Boolean flag indicating if we train epsilon or not
        train_set
            The training set
        init_eps : float, optional
            The epsilon to initialize :math:`A` and :math:`b`, if passed. If not passed,
            :math:`A` will be initialized as the inverse square root of the
            covariance of the data, and b will be initialized as :math:`\bar{d}`.
        init_A
            TODO (Amit): Irina - please provide explanation
            The given initiation for the reshaping matrix A. If none is passed, it will be initiated as the covarience matrix of the provided data. 

        Returns:

        init
            np.array (NOT TENSOR)
        """

        if eps:
            return init_A if (init_A is not None) else np.eye(train_set.shape[1])

        cov_len_cond = (len(np.shape(np.cov(train_set.T))) >= 1)
        if (init_eps is None) and (init_A is None):
            if cov_len_cond:
                return sc.linalg.sqrtm(sc.linalg.inv(np.cov(train_set.T)))
            return np.array([[np.cov(train_set.T)]])

        scalar = 1/init_eps if init_eps else 1
        mat_shape = train_set.shape[1] if cov_len_cond else 1
        matrix = np.array(init_A) if (
            init_A is not None) else np.eye(mat_shape)
        return scalar * matrix

    def _init_torches(self, init_eps, init_A, init_b, init_alpha, train_set, eps_tch,
                      mro_set, unc_set):
        """
        This function Initializes and returns paramT_tch, paramb_tch, and alpha as tensors
        """

        eps = (eps_tch is not None)
        if self._init is None:
            self._init = self._gen_init(eps, train_set, init_eps, init_A)
        init_tensor = torch.tensor(
            self._init, requires_grad=self.train_flag, dtype=settings.DTYPE)
        paramb_tch = None
        case = self._calc_mro_case(eps_tch, mro_set, unc_set, init_A)

        if case == settings.MRO_CASE.NO_MRO:
            if init_b is not None:
                paramb_tch_data = np.array(init_b)
            else:
                paramb_tch_data = -self._init@np.mean(train_set, axis=0)
            paramb_tch = torch.tensor(paramb_tch_data, requires_grad=self.train_flag,
                                      dtype=settings.DTYPE)
            paramb_tch = eps_tch*paramb_tch if eps else paramb_tch
            paramT_tch = init_tensor
            if eps:
                paramT_tch *= self._init

        elif case == settings.MRO_CASE.DIFF_A_UNINIT:
            paramT_tch = eps_tch[0]*init_tensor
            for k_ind in range(1, unc_set._K):
                paramT_tch = torch.vstack(
                    (paramT_tch, eps_tch[k_ind]*self._init))

        elif case == settings.MRO_CASE.DIFF_A_INIT:
            paramT_tch = eps_tch[0]*torch.tensor(init_A[0:unc_set._m, 0:unc_set._m],
                                                 dtype=settings.DTYPE)
            for k_ind in range(1, unc_set._K):
                paramT_tch = torch.vstack(
                    (paramT_tch, eps_tch[k_ind] *
                        torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)*unc_set._m,
                                            0:unc_set._m],
                                     dtype=settings.DTYPE)))

        elif case == settings.MRO_CASE.SAME_A:
            paramT_tch = eps_tch*init_tensor

        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)
        return paramT_tch, paramb_tch, alpha, case

    def _split_dataset(self, unc_set, test_percentage, seed):
        """
        This function splits the uncertainty set into train and test sets
            and also creates torch tensors

        Args:

        unc_set
            Uncertainty set
        test_percentage
            Fraction of test percentage
        seed
            Random number generator seed

        Returns:

        train_set
            Training set
        test_set
            Testing set
        val_dset
            Training set torch tensor
        eval_set
            Testing set torch tensor
        """

        # Split the dataset into train_set and test, and create Tensors
        # Val is the training set, eval is the test set
        train_set, test_set = train_test_split(unc_set.data, test_size=int(
            unc_set.data.shape[0]*test_percentage), random_state=seed)
        val_dset = torch.tensor(
            train_set, requires_grad=self.train_flag, dtype=settings.DTYPE)
        eval_set = torch.tensor(
            test_set, requires_grad=self.train_flag, dtype=settings.DTYPE)

        return train_set, test_set, val_dset, eval_set

    def _update_iters(self, save_iters, T_iter, b_iter, paramT_tch, paramb_tch, mro_set):
        """
        This function updates T_iter and b_iter

        Args:

        save_iters
            Whether to save per iteration data or not
        T_iter
            List of per-iteration T
        b_iter
            List of per-iteration b
        paramT_tch
            Torch tensor of T
        paramb_tch
            Torch tensor of b
        mro_set
            Boolean flag set to True for MRO problem

        Returns:

        None
        """

        if not save_iters:
            return

        T_iter.append(paramT_tch.detach().numpy().copy())
        if not mro_set:
            b_iter.append(paramb_tch.detach().numpy().copy())

    def _set_train_varaibles(self, fixb, mro_set, init_A, unc_set, alpha, paramT_tch,
                             paramb_tch, eps_tch):
        """
        This function sets the variables to be trained in the outer level problem
        TODO (Amit): complete the docstrings
        """
        if eps_tch is not None:
            variables = [eps_tch, alpha]
            return paramT_tch, variables

        if fixb or mro_set:
            if mro_set and unc_set._uniqueA:
                if init_A is None:
                    paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                elif init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m):
                    paramT_tch = paramT_tch.repeat(unc_set._K, 1)
            paramT = paramT_tch.detach().numpy()
            paramT_tch = torch.tensor(
                paramT, requires_grad=self.train_flag, dtype=settings.DTYPE)
            variables = [paramT_tch, alpha]
        else:
            variables = [paramT_tch, paramb_tch, alpha]

        return paramT_tch, variables

    def _gen_new_lst(self, num_ys, paramT_tch, paramb_tch, mro_set, y_parameters):
        """
        This function generates a set of parameters for each y in the family of y's

        Args:

        num_ys
            Number of y's in the family
        paramT_tch
            Parameter T torch tensor
        paramb_tch
            Parameter b torch tensor
        mro_set
            Boolean flag set to True for MRO problem
        y_parameters
            Y parameters
        """
        # Save the parameters for each y in the family of y's
        newlst = {}
        for scene in range(num_ys):
            newlst[scene] = []
            for i in range(len(y_parameters)):
                newlst[scene].append(torch.tensor(
                    np.array(y_parameters[i].data[scene, :])
                    .astype(float), requires_grad=self.train_flag, dtype=settings.DTYPE))
            newlst[scene].append(paramT_tch)
            if not mro_set:
                newlst[scene].append(paramb_tch)
        return newlst

    def _gen_eps_tch(self, init_eps, unc_set, mro_set):
        """
        This function generates eps_tch

        Args:

        init_eps
            Initial epsilon
        unc_set
            Uncertainty set
        mro_set
            Boolean flag set to True for MRO problem
        """

        scalar = 1/np.array(init_eps) if init_eps else 1.0
        eps_tch = torch.tensor(
            scalar, requires_grad=self.train_flag, dtype=settings.DTYPE)

        if (not mro_set) or (not unc_set._uniqueA):
            return eps_tch

        if init_eps and eps_tch.shape != torch.Size([]):
            return eps_tch

        eps_tch = eps_tch.repeat(unc_set._K)
        eps_tch = eps_tch.detach().numpy()
        eps_tch = torch.tensor(
            eps_tch, requires_grad=self.train_flag, dtype=settings.DTYPE)
        return eps_tch

    def _calc_mro_case(self, eps_tch, mro_set, unc_set, init_A):
        """
        This function calculates the MRO_CASE of this problem

        Args:

        eps_tch
            Epsilon torch
        mro_set
            MRO Set
        unc_set
            Uncertainty set
        init_A
            Initialized A

        Returns:
            MRO case
        case
            The MRO_CASE

        """

        eps = (eps_tch is not None)
        case = settings.MRO_CASE.NO_MRO
        if (not eps) or (not mro_set):
            return case

        elif unc_set._uniqueA and eps:
            if init_A is None or init_A.shape[0] != (unc_set._K*unc_set._m):
                case = settings.MRO_CASE.DIFF_A_UNINIT
            else:
                case = settings.MRO_CASE.DIFF_A_INIT
        else:
            case = settings.MRO_CASE.SAME_A

        return case

    def _get_unc_set(self):
        """
        This function returns unc_set.data.

        Args:
            None

        Returns:
            uncertainty set

        Raises:
            ValueError if unc_set.data is None
        """

        unc_set = self.uncertain_parameters()[0].uncertainty_set
        if unc_set.data is None:
            raise ValueError("Cannot train without uncertainty set data")
        return unc_set

    def _calc_coverage(self, dset, paramT_tch, eps_tch1, paramb_tch):
        """
        This function calculates coverage.

        Args:
            dset:
                Dataset (train or test)
            paramT_tch:
                T torch
            eps_tch1:
                epsilon torch
            paramb_tch:
                b torch

        Returns:
            Coverage
        """
        coverage = 0
        for datind in range(dset.shape[0]):
            coverage += torch.where(
                torch.norm(paramT_tch @ dset[datind] +
                           eps_tch1[0][0] * paramb_tch)
                <= 1,
                1,
                0,
            )
        return coverage

    # TODO (Amit): Do we want all default values in settings.py, or only the numerical values?
    # Response (Irina): We can put all default values in settings.py
    def train(
        self,
        eps=False,
        fixb=False,
        num_iter=settings.NUM_ITER_DEFAULT,  # Used to be "step"
        lr=settings.LR_DEFAULT,
        scheduler=True,
        momentum=settings.MOMENTUM_DEFAULT,
        optimizer=settings.OPT_DEFAULT,
        init_eps=None,
        init_A=None,
        init_b=None,
        save_iters=False,
        seed=settings.SEED_DEFAULT,
        init_lam=settings.INIT_LAM_DEFAULT,
        init_alpha=settings.INIT_ALPHA_DEFAULT,
        # TODO (Bart): This should be Kappa and passed
        kappa=settings.KAPPA_DEFAULT,
        # (originall target_cvar)
        test_percentage=settings.TEST_PERCENTAGE_DEFAULT,
        ys=None,  # TODO (Amit): Can we remove this variable?
        # Response (Irina): Yes. In fact, num_ys can also be removed.
        num_ys=None,
        # TODO (Irina): step_y can also be renamed step_lam
        step_y=settings.STEP_Y_DEFAULT,
        batch_percentage=settings.BATCH_PERCENTAGE_DEFAULT,
        # TODO (Amit): Why is this not used? Can we remove it?
        # Response (Irina): The solver for the train is limited to ecos/scs, so yes we can remove it.
        solver: Optional[str] = None,
    ):
        r"""
        Trains the uncertainty set parameters to find optimal set w.r.t. lagrangian metric

        Parameters TODO (Amit): Irina - update all the variables
        -----------
        eps : bool, optional
           If True, train only epsilon, where :math:`A = \epsilon I, \
           b = \epsilon \bar{d}`, where :math:`\bar{d}` is the centroid of the
           training data. Default False.
        num_iter : int, optional
            The total number of gradient steps performed.
        lr : float, optional
            The learning rate of gradient descent.
        momentum: float between 0 and 1, optional
            The momentum for gradient descent.
        optimizer: str or letters, optional
            The optimizer to use tor the descent algorithm.
        init_eps : float, optional
            The epsilon to initialize :math:`A` and :math:`b`, if passed. If not passed,
            :math:`A` will be initialized as the inverse square root of the
            covariance of the data, and b will be initialized as :math:`\bar{d}`.
        init_A : numpy array, optional
            Initialization for the reshaping matrix, if passed. If not passed, :math:`A` will be initialized as the inverse square root of the
            covariance of the data. Default None. 
        init_b : numpy array, optional
            Initialization for the relocation vector, if passed. If not passed, b will be initialized as :math:`\bar{d}`. Default None. 
        init_alpha : float, optional
            The initial alpha value for the cvar constraint in the outer level problem. Default -0.01.
        init_lam : float, optional
            The initial lambda value for the outer level lagrangian function. Default 0.
        kappa : float, optional
            The target value for the outer level cvar constraint. Default -0.015.
        schedular : bool, optional
            Flag for whether or not to decrease the learning rate on plateau of the derivatives. 
        test_percentage : float, optional
            The percentage of data to use in the testing set. Default 0.2.
        seed : int, optional
            The seed to control the random state of the train-test data split.
        step_y : float, optional
            The step size for the lambda value updates in the outer level problem. Default 0.1. 
        batch_percentage : float, optional
            The percentage of data to use in each training step. Default 0.2.
        Returns:
        A pandas data frame with information on each :math:r`\epsilon` having the following columns:
            Test_val: float
                The out of sample objective value of the Robust Problem
            Lagrangian_val: float
                The value of the lagrangian function applied to the training data
            test_vio:
                Probability of constraint violation over test set
            train_vio:
                Probability fo constraint violation over train set
            violation_test:
                Violation of learning constraint over test set
            violation_train:
                Violation of learning constraint over train set
            Eps: float
                The :math:`\epsilon` value
            coverage_test: float
                The percentage of testing data covered by the uncertainty set
            coverage_train : float
                The percentage of training data covered by the uncertainty set
        """

        # Validity checks and initializations
        self.train_flag = True
        T_iter = []
        b_iter = []
        self._validate_uncertain_parameters()

        unc_set = self._get_unc_set()
        self._canonize_problem()
        self._validate_unc_set_T(unc_set)

        mro_set = self._is_mro_set(unc_set)
        df = pd.DataFrame(columns=["step", "Opt_val", "Eval_val", "Lagrangian_val",
                                   "Violations", "A_norm"])

        # setup train and test data
        #test_set is not used
        train_set, _, val_dset, eval_set = self._split_dataset(
            unc_set, test_percentage, seed)

        cvxpylayer = CvxpyLayer(self.new_prob, parameters=self.y_parameters()
                                + self.shape_parameters(self.new_prob), variables=self.variables())
        eps_tch = self._gen_eps_tch(
            self, init_eps, unc_set, mro_set) if eps else None
        paramT_tch, paramb_tch, alpha, _ = self._init_torches(init_eps, init_A, init_b,
                                                              init_alpha, train_set, eps_tch,
                                                              mro_set, unc_set)
        if not eps:
            self._update_iters(save_iters, T_iter, b_iter,
                               paramT_tch, paramb_tch, mro_set)
        paramT_tch, variables = self._set_train_varaibles(fixb, mro_set, init_A, unc_set, alpha,
                                                          paramT_tch, paramb_tch, eps_tch)
        opt = settings.OPTIMIZERS[optimizer](
            variables, lr=lr, momentum=momentum)
        scheduler = None
        if scheduler:
            scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=settings.PATIENCE)

        # y's and cvxpylayer begin
        y_parameters = self.y_parameters()
        num_ys = self.num_ys
        # TODO (Amit): Irina - is it intended that this happens only if not eps?
        # Response (Irina): I actually do not know what this is. It also does not seem to do anything.
        if not eps:
            self.new_prob.parameters()
        newlst = self._gen_new_lst(
            num_ys, paramT_tch, paramb_tch, mro_set, y_parameters)
        # train
        lam_list = init_lam * torch.ones((num_ys, self.num_g), dtype=float)
        lam = init_lam * torch.ones(self.num_g, dtype=float)

        # step is the number of iterations
        for step_num in range(num_iter):
            train_stats = TrainLoopStats(
                step_num=step_num, train_flag=self.train_flag)
            # Index to select what data the SGD takes
            random_int = np.random.randint(0, val_dset.shape[0],
                                           int(val_dset.shape[0]*batch_percentage))
            for scene in range(num_ys):
                if not mro_set:
                    newlst[scene][-1] = paramb_tch
                    newlst[scene][-2] = paramT_tch
                else:
                    paramT_tch, _, _, _ = self._init_torches(init_eps, init_A, init_b,
                                                             init_alpha, train_set,
                                                             eps_tch, mro_set, unc_set)
                    newlst[scene][-1] = paramT_tch
                # Solve the problem for specific y, returns the variables (x)
                var_values = cvxpylayer(*newlst[scene],
                                        solver_args=settings.LAYER_SOLVER)

                # temp_lagrangian     - Lagrangian of the Lagrangian problem (F(x)+lambda H(x))
                # obj                - Objective value of the problem (F(x))
                # cvar_update        - Amount of constraint violation (H(z))
                # train_vio         - Probability of violation

                # Training set:
                temp_lagrangian, obj, cvar_update, train_vio = self.lagrange(var_values,
                                                                             [newlst[scene][:-2]], self._udata_to_lst(
                                                                                 val_dset[random_int]),
                                                                             alpha, lam)
                # Testing set:
                evallagrangian, obj_validation, var_vio, test_vio = self.lagrange(var_values,
                                                                                  [newlst[scene][:-2]], self._udata_to_lst(
                                                                                      eval_set),
                                                                                  alpha, lam)
                # Update parameters for the next y
                lam_list[scene, :] = cvar_update.detach()
                train_stats.update_stats(temp_lagrangian, num_ys, obj_validation,
                                         train_vio, test_vio, var_vio, cvar_update)

            # calculate statistics over all y
            lam = torch.maximum(lam + step_y*(torch.mean(lam_list, axis=0)),
                                torch.zeros(self.num_g, dtype=float))
            train_stats.tot_lagrangian.backward()
            coverage = 0
            for datind in range(val_dset.shape[0]):
                coverage += torch.where(torch.norm(paramT_tch@val_dset[datind] + paramb_tch) <= 1,
                                        1, 0)
            coverage2 = 0
            for datind in range(eval_set.shape[0]):
                coverage2 += torch.where(torch.norm(paramT_tch@eval_set[datind] + paramb_tch) <= 1,
                                         1, 0)

            # BEFORE UPDTATING PANDAS DATAFRAME
            new_row = train_stats.generate_row(num_ys, paramT_tch, lam, alpha, coverage,
                                               coverage2, val_dset, eval_set, eps_tch)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            self._update_iters(save_iters, T_iter, b_iter,
                               paramT_tch, paramb_tch, mro_set)
            if step_num < num_iter - 1:
                opt.step()
                opt.zero_grad()
                if scheduler:
                    scheduler_.step(train_stats.tot_lagrangian)

        self._trained = True
        unc_set._trained = True
        unc_set.paramT.value = paramT_tch.detach().numpy().copy()
        if not mro_set:
            unc_set.paramb.value = paramb_tch.detach().numpy().copy()

        return_eps = eps_tch.detach().numpy().copy() if eps else 1
        return_parab_value = unc_set.paramb.value if mro_set else None
        return_b_iter = b_iter if mro_set else None
        return Result(self, self.new_prob, df, unc_set.paramT.value, return_parab_value,
                      return_eps, obj.item(), var_values, T_iter=T_iter, b_iter=return_b_iter)

    def grid(
        self,
        epslst=settings.EPS_LST_DEFAULT,
        seed=settings.SEED_DEFAULT,
        init_A=None,
        init_b=None,
        init_alpha=settings.INIT_ALPHA_DEFAULT,
        test_percentage=settings.TEST_PERCENTAGE_DEFAULT,
        ys=None,  # TODO (Amit): This is not used, can we delete it?
        num_ys=None,
        # TODO (Amit): This is not used, can we delete it?
        # Response (Irina): Yes.
        solver: Optional[str] = None,
    ):
        r"""
        Perform gridsearch to find optimal :math:`\epsilon`-ball around data.

        Args:

        epslst : np.array, optional
            The list of :math:`\epsilon` to iterate over. "Default np.logspace(-3, 1, 20)
        seed: int
            The seed to control the train test split. Default 1.
        solver: optional
            A solver to perform gradient-based learning

        Returns:

        A pandas data frame with information on each :math:`\epsilon` having the following columns:
            Opt_val: float
                The objective value of the Robust Problem
            Lagrangian_val: float
                The value of the lagrangian function applied to the training data
            Eval_val: float
                The value of the lagrangian function applied to the evaluation data
            Eps: float
                The epsilon value
        """

        self.train_flag = False
        self._validate_uncertain_parameters()

        unc_set = self._get_unc_set()
        self._canonize_problem()

        self._validate_unc_set_T(unc_set)
        df = pd.DataFrame(
            columns=["Opt_val", "Eval_val", "Lagrangian_val", "Eps"])
        mro_set = self._is_mro_set(unc_set)

        # setup train and test data
        train_set, _, val_dset, eval_set = self._split_dataset(
            unc_set, test_percentage, seed)

        # creviolationsate cvxpylayer
        cvxpylayer = CvxpyLayer(
            self.new_prob, parameters=self.new_prob.parameters(), variables=self.variables()
        )

        # paramlst = prob.parameters()
        y_parameters = self.y_parameters()
        num_ys = self.num_ys

        newlst = {}
        for scene in range(num_ys):
            newlst[scene] = []
            for i in range(len(y_parameters)):
                newlst[scene].append(
                    torch.tensor(
                        np.array(y_parameters[i].data[scene, :]).astype(float),
                        requires_grad=self.train_flag,
                        dtype=settings.DTYPE,
                    )
                )
            # TODO (Amit): Why do we always append 0? Is this a mistake?
            # It looks very similar to _gen_new_lst, and I wonder if they should be the same?
            # Response (Irina): Appending 0 is to make space for paramT and paramb. There should be a better way to do it. Yes, this is the same as gen new lst
            newlst[scene].append(0)
            if not mro_set:
                newlst[scene].append(0)

        grid_stats = GridStats()

        '''
        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)

        if init_A is not None:
            init = torch.tensor(init_A, requires_grad=self.train_flag, dtype=settings.DTYPE)
        else:
            init = torch.tensor(np.eye(train_set.shape[1]), requires_grad=self.train_flag,
                                dtype=settings.DTYPE)
        if init_b is not None:
            init_bval = torch.tensor(init_b, requires_grad=self.train_flag, dtype=settings.DTYPE)
        else:
            init_bval = torch.tensor(-np.mean(train_set, axis=0), requires_grad=self.train_flag,
                                        dtype=settings.DTYPE)
        '''

        # TODO (Amit): What is this 1000? (numbers have no business inside a function)
        # There's no training so there shouldn't be a lambda here.
        # Response (Irina): We have a lambda here because we still use the same evaluation functions as before, which needs a lambda. Though, we should only be interested in returning the eval value (out of sample objective with respect to the test set) and not the lagrangian value, so the value of lambda is inconsequential.

        lam = 1000 * torch.ones(self.num_g, dtype=float)

        # TODO (Amit): Is it true that epss is the same as init_eps in train? If yes, we will have
        # to change the name to be consistent
        # Response (Irina): Yes, functionally it should be the same.
        for epss in epslst:
            # TODO (Amit): I am a bit confused. I am not sure if _init_torches that is used in train
            # implemenets the same logic as in the commented block above.
            # Response (Irina): It should be the same logic as in train. I think this part was simplified (did not have the covariance part for passing None, as the current use cases always involved passing initA and initb).
            # TODO (Amit): Why is it called eps_tch1 and not eps_tch?
            # Response (Irina): It should be eps_tch. This was a relic of editing the code and changing things a while ago...
            # TODO (Amit): #Is this call correct? There are more conditions in this function
            # Also, why is it called eps_tch1 and not eps_tch?
            eps_tch1 = self._gen_eps_tch(epss, unc_set, mro_set)
            paramT_tch, paramb_tch, alpha, case = self._init_torches(epss, init_A, init_b,
                                                                     init_alpha, train_set, eps_tch1,
                                                                     mro_set, unc_set)

            train_stats = TrainLoopStats(
                step_num=np.NAN, train_flag=self.train_flag)
            for scene in range(num_ys):
                if not mro_set:
                    newlst[scene][-1] = paramb_tch
                    newlst[scene][-2] = paramT_tch
                else:
                    paramT_tch, _, _, _ = self._init_torches(epss, init_A, init_b,
                                                             init_alpha, train_set,
                                                             eps_tch1, mro_set, unc_set)
                    newlst[scene][-1] = paramT_tch

                var_values = cvxpylayer(
                    *newlst[scene], solver_args=settings.LAYER_SOLVER)

                # TODO (Bart): Is there a need to do lagrange here? There's no training.
                # TODO (Amit): There are some unused variables here. Please correct/delete.
                # Response (Irina): We don't need to do the lagrangian, but we do want the value and probability of the violations, and the test set evaluation value.
                temp_lagrangian, obj, cvar_update, train_vio = self.lagrange(
                    var_values,
                    [newlst[scene][:-2]],
                    # TODO (Amit): shouldn't be val_dset[random_int]?
                    self._udata_to_lst(val_dset),
                    alpha,
                    lam,
                )
                # temp_lagrangian, obj, train_vio,cvar_update = unc_set.lagrangian(
                #     *var_values, *newlst[scene][:-2], alpha = torch.tensor(
                #     init_alpha), data = val_dset)

                evallagrangian, obj_validation, var_vio, test_vio = self.lagrange(
                    var_values,
                    [newlst[scene][:-2]],
                    self._udata_to_lst(eval_set),
                    alpha,
                    lam,
                )
                # evallagrangian, obj_validation, test_vio, var_vio = unc_set.lagrangian(
                #     *var_values, *newlst[scene][:-2], alpha = torch.tensor(
                #     init_alpha), data = eval_set)

                train_stats.update_stats(temp_lagrangian, num_ys, obj_validation,
                                         train_vio, test_vio, var_vio, cvar_update)
            grid_stats.update(train_stats, temp_lagrangian,
                              eps_tch1, paramT_tch, var_values)

            # TODO (Amit): Shouldn't it be train_dset and test_dset (applies also to train)?
            # TODO (Amit): If so, update names from coverage/coverage2 (bad names)
            # TODO (Amit): Why do we sometimes use validation/evaluation/train/test?
            # Can we come up with consistent names?
            # TODO (Bart): why val_dset and eval_set? either dset or set.
            # Response (Irina): Yes, we should use train_dset and test_dset everywhere. We should not use validation/evaluation.
            coverage = self._calc_coverage(
                self, val_dset, paramT_tch, eps_tch1, paramb_tch)
            coverage2 = self._calc_coverage(
                self, eval_set, paramT_tch, eps_tch1, paramb_tch)

            new_row = train_stats.generate_row(num_ys, paramT_tch, lam, alpha, coverage, coverage2,
                                               val_dset, eval_set, eps_tch1)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        self._trained = True
        unc_set._trained = True

        if not mro_set:
            # TODO (Amit): Irina, can we go over these assignments and see if we can use
            # previously calculated torches? (I initialized to None to pass tests)
            # Response (Irina): Perhaps there should be a function that initializes only init and init_bval, instead of init_torches that already combines these with the epsilon values. Then init_torches can call that function to do the combination, and we can also use the init and init_bval here.
            init = None  # TODO (Amit): TEMPORARY ONLY, REMOVE
            init_bval = None  # TODO (Amit): TEMPORARY ONLY, REMOVE
            unc_set.paramT.value = (
                grid_stats.mineps * init).detach().numpy().copy()
            unc_set.paramb.value = (
                grid_stats.mineps[0] * init_bval).detach().numpy().copy()
        else:
            unc_set.paramT.value = grid_stats.minT.detach().numpy().copy()
        paramb_value = None if mro_set else unc_set.paramb.value
        return Result(
            self,
            self.new_prob,
            df,
            unc_set.paramT.value,
            paramb_value,
            grid_stats.mineps[0][0].detach().numpy().copy(),
            grid_stats.minval,
            grid_stats.var_vals,
        )

    def dualize_constraints(self):
        # TODO (Bart): This might be a redundant function, construct_chain should do that.
        if self.uncertain_parameters():
            unc_reductions = []
            if type(self.objective) == Maximize:
                unc_reductions += [FlipObjective()]

            unc_reductions += [Distribute_Uncertain_Params()]
            unc_reductions += [RemoveUncertainParameters()]
            newchain = UncertainChain(self, reductions=unc_reductions)
            prob, _ = newchain.apply(self)
            return prob
        return super(RobustProblem, self)

    def solve(self, solver: Optional[str] = None):
        # TODO (Bart): Need to invert the data to get the solution to the original problem
        if self.new_prob is not None:
            return self.new_prob.solve(solver=solver)
        elif self.uncertain_parameters():
            if self.uncertain_parameters()[0].uncertainty_set.data is not None:
                if not type(self.uncertain_parameters()[0].uncertainty_set) == MRO:
                    _ = self.train_set()
                    return self.new_prob.solve(solver=solver)
                elif self.uncertain_parameters()[0].uncertainty_set._train:
                    _ = self.train_set()
                    return self.new_prob.solve(solver=solver)
            prob = self.dualize_constraints()
            return prob.solve(solver=solver)
        return super(RobustProblem, self).solve()


class Result(ABC):
    def __init__(self, prob, probnew, df, T, b, eps, obj, x, T_iter=None, b_iter=None):
        self._reform_problem = probnew
        self._problem = prob
        self._df = df
        self._A = T
        self._b = b
        self._obj = obj
        self._x = x
        self._eps = eps
        self._T_iter = T_iter
        self._b_iter = b_iter

    @property
    def problem(self):
        return self._problem

    @property
    def df(self):
        return self._df

    @property
    def reform_problem(self):
        return self._reform_problem

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def eps(self):
        return self._eps

    @property
    def obj(self):
        return self._obj

    @property
    def var_values(self):
        return self._x

    @property
    def uncset_iters(self):
        return self._T_iter, self._b_iter
