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
        prob_violation_test:
            Probability of constraint violation over test set
        prob_violation_train:
            Probability fo constraint violation over train set
        violation_test:
            Violation of learning constraint over test set
        violation_train:
            Violation of learning constraint over train set
    """

    def __init__(self, step_num, train_flag=True):
        def __value_init__(self):
            """
            This is an internal function that either initiates a tensor or a list
            """
            return [] if not self.train_flag else torch.tensor(0., dtype=settings.DTYPE)
        self.step_num = step_num
        self.train_flag = train_flag
        self.tot_lagrangian = __value_init__(self)
        self.testval = __value_init__(self)
        self.trainval = __value_init__(self)
        self.prob_violation_test = __value_init__(self)
        self.prob_violation_train = __value_init__(self)
        self.violation_test = __value_init__(self)
        self.violation_train = __value_init__(self)

    def update_train_stats(self, temp_lagrangian, obj, prob_violation_train, train_constraint):
        """
        This function updates the statistics after each training iteration
        """
        self.tot_lagrangian = temp_lagrangian
        # if not self.train_flag:
        #     self.testval.append(obj_test.item())
        #     self.prob_violation_test.append(prob_violation_test.item())
        #     self.prob_violation_train.append(prob_violation_train.item())
        #     self.violation_test.append(var_vio.item())
        #     self.violation_train.append(train_constraint.item())
        # else:  # if self.train_flag
        self.trainval = obj.item()
        self.prob_violation_train = prob_violation_train.item()
        self.violation_train = train_constraint.item()

    def update_test_stats(self, obj_test, prob_violation_test, var_vio):
        """
        This function updates the statistics after each training iteration
        """
        # if not self.train_flag:
        #     self.testval.append(obj_test.item())
        #     self.prob_violation_test.append(prob_violation_test.item())
        #     self.prob_violation_train.append(prob_violation_train.item())
        #     self.violation_test.append(var_vio.item())
        #     self.violation_train.append(train_constraint.item())
        # else:  # if self.train_flag
        self.testval = obj_test.item()
        self.prob_violation_test = prob_violation_test.item()
        self.violation_test = var_vio.item()

    # TODO (Amit): Check why train_set is not used
    def generate_train_row(self, a_tch, lam, alpha):
        """
        This function generates a new row with the statistics
        """
        row_dict = {
            "Lagrangian_val":   self.tot_lagrangian.item(),
            "Train_val":         self.trainval,
            "Probability_violations_train": self.prob_violation_train,
            "Violation_train":  self.violation_train,
        }
        row_dict["step"] = self.step_num,
        row_dict["A_norm"] = np.linalg.norm(
            a_tch.detach().numpy().copy()),
        row_dict["lam_list"] = lam.detach().numpy().copy(),
        row_dict["alpha"] = alpha.item(),
        row_dict["alphagrad"] = alpha.grad,
        row_dict["dfnorm"] = np.linalg.norm(a_tch.grad),
        row_dict["gradnorm"] = a_tch.grad,
        new_row = pd.Series(row_dict)
        return new_row

    def generate_test_row(self, calc_coverage, a_tch, b_tch, alpha, test_tch, eps_tch):
        """
        This function generates a new row with the statistics
        """
        coverage_test = calc_coverage(
            test_tch, a_tch, b_tch)
        row_dict = {
            "Test_val":         self.testval,
            "Probability_violations_test":       self.prob_violation_test,
            "Violation_test":   self.violation_test,
            "Coverage_test":    coverage_test.detach().numpy().item()
        }
        row_dict["step"] = self.step_num,
        if not self.train_flag:
            row_dict["Eps"] = 1 / eps_tch[0][0].detach().numpy().copy()
        new_row = pd.Series(row_dict)
        return new_row


class GridStats():
    """
    This class contains useful information for grid search
    """

    def __init__(self):
        self.minval = np.float("inf")
        self.var_vals = 0

    def update(self, train_stats, obj, eps_tch, a_tch, var_values):
        """
        This function updates the best stats in the grid search.

        Args:
            train_stats:
                The train stats
            obj:
                Calculated test objective
            eps_tch:
                Epsilon torch
            a_tch
                A torch
            var_values
                variance values
        """
        if train_stats.testval <= self.minval:
            self.minval = obj
            self.mineps = eps_tch.clone()
            self.minT = a_tch.clone()
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

            def hg(vars, y_params, u_params, alpha, eta):
                return (
                    torch.maximum(
                        g(*vars, *y_params, u_params) - alpha,
                        torch.tensor(0.0, dtype=settings.DTYPE,
                                     requires_grad=self.train_flag),
                    ) / eta)

            h_funcs.append(hg)

        l_func = f_tch
        return l_func, h_funcs, len(h_funcs)

    def _eval_input(self, eval_func, vars, y_params_mat, u_params_mat, star_flag, *args):
        """
        This function takes decision varaibles, y's, and u's,
            evaluates them and averages them on a given function.

        Args:
            eval_func:
                The function used for evaluation.
            vars:
                Contains all the variables we optimize over (J in total)
            y_params_mat:  
                Contains all the y's (J in total)
            u_params_mat:
                Contains all the u data points (N in total)
            star_flag:
                If True, will expand vars and y_params_mat

        Returns:
            The average among all evaluated J x N pairs
        """
        J = len(y_params_mat[0])
        init_val = 0
        
        for j in range(J):
            if star_flag:
                init_val += eval_func(*[valuex[j] for valuex in vars],
                                        *[valuey[j] for valuey in y_params_mat], u_params_mat, *args)
            else:
                init_val += eval_func([valuex[j] for valuex in vars],
                                        [valuey[j] for valuey in y_params_mat], u_params_mat, *args)
                                    
        init_val /= J
        return init_val.mean()

    def train_objective(self, vars, y_params_mat, u_params_mat):
        return self._eval_input(eval_func=self.l, vars=vars, y_params_mat=y_params_mat,
                                u_params_mat=u_params_mat, star_flag=True)

    def train_constraint(self, vars, y_params_mat, u_params_mat, alpha, eta, kappa):
        num_g = len(self.h)
        J = len(y_params_mat[0])
        N = len(u_params_mat)
        H = torch.zeros(num_g, dtype=settings.DTYPE)
        for k, h_k in enumerate(self.h):
            init_val = self._eval_input(h_k, vars, y_params_mat, u_params_mat, False, alpha, eta)
            h_k_expectation = init_val + alpha - kappa
            H[k] = h_k_expectation
        return H

    def evaluation_metric(self, vars, y_params_mat, u_params_mat):
        if (self.eval is None):
            return 0
        return self._eval_input(eval_func=self.eval, vars=vars, y_params_mat=y_params_mat,
                                u_params_mat=u_params_mat, star_flag=True)

    def prob_constr_violation(self, vars, y_params_mat, u_params_mat):
        num_g = len(self.g)
        J = len(y_params_mat[0])
        N = len(u_params_mat)
        G = torch.zeros((num_g, J, N), dtype=settings.DTYPE)

        for k, g_k in enumerate(self.g):
            for i in range(N):
                for j in range(J):
                    G[k, j, i] = g_k(*[valuex[j] for valuex in vars],
                                     *[valuey[j] for valuey in y_params_mat], u_params_mat[i])

        g_max = torch.max(G, dim=0)[0]
        g_max_violate = (g_max > 0).float()
        return torch.mean(g_max_violate)

    # helper function for intermediate version
    def _udata_to_lst(self, data, batch_size):
        num_instances = data.shape[0]
        random_int = np.random.randint(0, num_instances,
                                       int(num_instances*batch_size))
        # u_params_mat = []
        # for i in range(num_instances):
        #     u_params_mat.append([data[i, :]])
        # return u_params_mat
        return torch.tensor(data[random_int], requires_grad=self.train_flag, dtype=settings.DTYPE)

    def lagrangian(self, vars, y_params_mat, u_params_mat, alpha, lam,
                   eta=settings.ETA_LAGRANGIAN_DEFAULT, kappa=settings.KAPPA_LAGRANGIAN_DEFAULT):
        F = self.train_objective(vars, y_params_mat, u_params_mat)
        H = self.train_constraint(
            vars, y_params_mat, u_params_mat, alpha, eta, kappa)
        return F + lam @ H, H.detach()

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
            ValueError if there is no a in the uncertainty set
        """

        if unc_set.a is None:
            raise ValueError("unc_set.a is None")

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
            The given initiation for the reshaping matrix A.
            If none is passed, it will be initiated as the covarience matrix of the provided data.

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
        This function Initializes and returns a_tch, b_tch, and alpha as tensors
        """

        eps = (eps_tch is not None)
        if self._init is None:
            self._init = self._gen_init(eps, train_set, init_eps, init_A)
        init_tensor = torch.tensor(
            self._init, requires_grad=self.train_flag, dtype=settings.DTYPE)
        b_tch = None
        case = self._calc_mro_case(eps_tch, mro_set, unc_set, init_A)

        if case == settings.MRO_CASE.NO_MRO:
            if init_b is not None:
                b_tch_data = np.array(init_b)
            else:
                b_tch_data = -self._init@np.mean(train_set, axis=0)
            b_tch = torch.tensor(b_tch_data, requires_grad=self.train_flag,
                                 dtype=settings.DTYPE)
            b_tch = eps_tch*b_tch if eps else b_tch
            a_tch = init_tensor
            if eps:
                a_tch *= self._init

        elif case == settings.MRO_CASE.DIFF_A_UNINIT:
            a_tch = eps_tch[0]*init_tensor
            for k_ind in range(1, unc_set._K):
                a_tch = torch.vstack(
                    (a_tch, eps_tch[k_ind]*self._init))

        elif case == settings.MRO_CASE.DIFF_A_INIT:
            a_tch = eps_tch[0]*torch.tensor(init_A[0:unc_set._m, 0:unc_set._m],
                                            dtype=settings.DTYPE)
            for k_ind in range(1, unc_set._K):
                a_tch = torch.vstack(
                    (a_tch, eps_tch[k_ind] *
                        torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)*unc_set._m,
                                            0:unc_set._m],
                                     dtype=settings.DTYPE)))

        elif case == settings.MRO_CASE.SAME_A:
            a_tch = eps_tch*init_tensor
            if unc_set._uniqueA:
                if init_A is None:
                    a_tch = a_tch.repeat(unc_set._K, 1)
                elif init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m):
                    a_tch = a_tch.repeat(unc_set._K, 1)
            a = a_tch.detach().numpy()
            a_tch = torch.tensor(
                a, requires_grad=self.train_flag, dtype=settings.DTYPE)
        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)
        return a_tch, b_tch, alpha, case

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
        train_set
            Training set torch tensor
        test_set
            Testing set torch tensor
        """

        # Split the dataset into train_set and test, and create Tensors
        # Val is the training set, eval is the test set
        train_set, test_set = train_test_split(unc_set.data, test_size=int(
            unc_set.data.shape[0]*test_percentage), random_state=seed)
        train_tch = torch.tensor(
            train_set, requires_grad=self.train_flag, dtype=settings.DTYPE)
        test_tch = torch.tensor(
            test_set, requires_grad=self.train_flag, dtype=settings.DTYPE)

        return train_set, test_set, train_tch, test_tch

    def _update_iters(self, save_history, a_history, b_history, a_tch, b_tch, mro_set):
        """
        This function updates a_history and b_history

        Args:

        save_history
            Whether to save per iteration data or not
        a_history
            List of per-iteration T
        b_history
            List of per-iteration b
        a_tch
            Torch tensor of A
        b_tch
            Torch tensor of b
        mro_set
            Boolean flag set to True for MRO problem

        Returns:

        None
        """

        if not save_history:
            return

        a_history.append(a_tch.detach().numpy().copy())
        if not mro_set:
            b_history.append(b_tch.detach().numpy().copy())

    def _set_train_variables(self, fixb, mro_set, alpha, a_tch,
                             b_tch, eps_tch):
        """
        This function sets the variables to be trained in the outer level problem.
        TODO (Amit): complete the docstrings (to Irina)
        """
        if eps_tch is not None:
            variables = [eps_tch, alpha]
            return variables

        if fixb or mro_set:
            variables = [a_tch, alpha]
        else:
            variables = [a_tch, b_tch, alpha]

        return variables

    def _gen_y_batch(self, num_ys, y_parameters, batch_size):
        """
        This function generates a set of parameters for each y in the family of y's

        Args:

        num_ys
            Number of y's in the family
        a_tch
            Parameter A torch tensor
        b_tch
            Parameter b torch tensor
        mro_set
            Boolean flag set to True for MRO problem
        y_parameters
            Y parameters
        """
        # Save the parameters for each y in the family of y's
        # y_batch = {}
        # for sample in range(num_ys):
        #     y_batch[sample] = []
        #     for i in range(len(y_parameters)):
        #         y_batch[sample].append(torch.tensor(
        #             np.array(y_parameters[i].data[sample, :])
        #             .astype(float), requires_grad=self.train_flag, dtype=settings.DTYPE))
        #     y_batch[sample].append(a_tch)
        # if not mro_set:
        #     y_batch[sample].append(b_tch)
        random_int = np.random.randint(0, num_ys,
                                       int(num_ys*batch_size))
        y_tchs = []
        for i in range(len(y_parameters)):
            y_tchs.append(torch.tensor(
                y_parameters[i].data[random_int], requires_grad=self.train_flag,
                dtype=settings.DTYPE))
        return y_tchs

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

        # Irina: uniqueA and not eps - goes to sameA
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

    def _calc_coverage(self, dset, a_tch, b_tch):
        """
        This function calculates coverage.

        Args:
            dset:
                Dataset (train or test)
            a_tch:
                A torch
            b_tch:
                b torch

        Returns:
            Coverage
        """
        coverage = 0
        for datind in range(dset.shape[0]):
            coverage += torch.where(
                torch.norm(a_tch @ dset[datind] +
                           b_tch)
                <= 1,
                1,
                0,
            )
        return coverage/dset.shape[0]

    def train(
        self,
        eps=settings.EPS_DEFAULT,
        fixb=settings.FIXB_DEFAULT,
        num_iter=settings.NUM_ITER_DEFAULT,  # Used to be "step"
        lr=settings.LR_DEFAULT,
        scheduler=settings.SCHEDULER_DEFAULT,
        momentum=settings.MOMENTUM_DEFAULT,
        optimizer=settings.OPT_DEFAULT,
        init_eps=settings.INIT_EPS_DEFAULT,
        init_A=settings.INIT_A_DEFAULT,
        init_b=settings.INIT_B_DEFAULT,
        save_history=settings.SAVE_HISTORY_DEFAULT,
        seed=settings.SEED_DEFAULT,
        init_lam=settings.INIT_LAM_DEFAULT,
        init_alpha=settings.INIT_ALPHA_DEFAULT,
        kappa=settings.KAPPA_DEFAULT,  # (originall target_cvar)
        test_percentage=settings.TEST_PERCENTAGE_DEFAULT,
        step_lam=settings.STEP_LAM_DEFAULT,
        u_batch_percentage=settings.U_BATCH_PERCENTAGE_DEFAULT,
        y_batch_percentage=settings.Y_BATCH_PERCENTAGE_DEFAULT,
        solver_args=settings.LAYER_SOLVER,
    ):
        r"""
        Trains the uncertainty set parameters to find optimal set w.r.t. lagrangian metric

        Parameters TODO (Amit): Irina - update all the variables
        -----------
        eps : bool, optional
           If True, train only epsilon, where :math:`A = \epsilon I, \
           b = \epsilon \bar{d}`, where :math:`\bar{d}` is the centroid of the
           training data. 
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
            Initialization for the reshaping matrix, if passed.
            If not passed, :math:`A` will be initialized as the inverse square root of the
            covariance of the data. 
        init_b : numpy array, optional
            Initialization for the relocation vector, if passed.
            If not passed, b will be initialized as :math:`\bar{d}`.
        init_alpha : float, optional
            The initial alpha value for the cvar constraint in the outer level problem.
        init_lam : float, optional
            The initial lambda value for the outer level lagrangian function.
        kappa : float, optional
            The target value for the outer level cvar constraint.
        schedular : bool, optional
            Flag for whether or not to decrease the learning rate on plateau of the derivatives. 
        test_percentage : float, optional
            The percentage of data to use in the testing set. 
        seed : int, optional
            The seed to control the random state of the train-test data split.
        step_lam : float, optional
            The step size for the lambda value updates in the outer level problem. 
        batch_percentage : float, optional
            The percentage of data to use in each training step.
        Returns:
        A pandas data frame with information on each :math:r`\epsilon` having the following columns:
            Test_val: float
                The out of sample objective value of the Robust Problem
            Lagrangian_val: float
                The value of the lagrangian function applied to the training data
            prob_violation_test:
                Probability of constraint violation over test set
            prob_violation_train:
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
        a_history = []
        b_history = []
        self._validate_uncertain_parameters()

        unc_set = self._get_unc_set()
        self._canonize_problem()
        self._validate_unc_set_T(unc_set)

        mro_set = self._is_mro_set(unc_set)
        df = pd.DataFrame(columns=["step"])
        df_test = pd.DataFrame(columns=["step"])

        # setup train and test data
        #test_set is not used
        train_set, _, train_tch, test_tch = self._split_dataset(
            unc_set, test_percentage, seed)

        cvxpylayer = CvxpyLayer(self.new_prob, parameters=self.y_parameters()
                                + self.shape_parameters(self.new_prob), variables=self.variables())
        eps_tch = self._gen_eps_tch(
            self, init_eps, unc_set, mro_set) if eps else None
        a_tch, b_tch, alpha, _ = self._init_torches(init_eps, init_A, init_b,
                                                    init_alpha, train_set, eps_tch,
                                                    mro_set, unc_set)
        if not eps:
            self._update_iters(save_history, a_history, b_history,
                               a_tch, b_tch, mro_set)

        variables = self._set_train_variables(fixb, mro_set, alpha,
                                              a_tch, b_tch, eps_tch)
        opt = settings.OPTIMIZERS[optimizer](
            variables, lr=lr, momentum=momentum)
        if scheduler:
            scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=settings.PATIENCE)

        # y's and cvxpylayer begin
        y_parameters = self.y_parameters()
        num_ys = self.num_ys
        lam = init_lam * torch.ones(self.num_g, dtype=settings.DTYPE)

        # use multiple initial points and training. pick lowest eval loss
        for step_num in range(num_iter):
            train_stats = TrainLoopStats(
                step_num=step_num, train_flag=self.train_flag)

            # generate batched y and u
            y_batch = self._gen_y_batch(
                num_ys, y_parameters, y_batch_percentage)

            u_batch = self._udata_to_lst(train_set, u_batch_percentage)

            if mro_set:
                a_tch, _, _, _ = self._init_torches(init_eps, init_A, init_b,
                                                    init_alpha, train_set,
                                                    eps_tch, mro_set, unc_set)
                var_values = cvxpylayer(*y_batch, a_tch,
                                        solver_args=solver_args)
            else:
                var_values = cvxpylayer(*y_batch, a_tch, b_tch,
                                        solver_args=solver_args)

            obj = self.evaluation_metric(var_values, y_batch, u_batch)
            prob_violation_train = self.prob_constr_violation(
                var_values,
                y_batch, u_batch)
            temp_lagrangian, train_constraint_value = self.lagrangian(
                var_values,
                y_batch,
                u_batch,
                alpha,
                lam,
                kappa=kappa)
            temp_lagrangian.backward()

            train_stats.update_train_stats(temp_lagrangian.detach().numpy(
            ).copy(), obj, prob_violation_train, train_constraint_value)

            lam = torch.maximum(lam + step_lam*train_constraint_value,
                                torch.zeros(self.num_g, dtype=settings.DTYPE))

            new_row = train_stats.generate_train_row(a_tch, lam, alpha)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

            self._update_iters(save_history, a_history, b_history,
                               a_tch, b_tch, mro_set)

            #TODO (Amit): 10 should probably be an input or a constant. 10 is also a bit small.
            if step_num % 10 == 0:
                y_batch = self._gen_y_batch(
                    num_ys, y_parameters, 1)
                if mro_set:
                    var_values = cvxpylayer(*y_batch, a_tch,
                                            solver_args=solver_args)
                else:
                    var_values = cvxpylayer(*y_batch, a_tch, b_tch,
                                            solver_args=solver_args)

                with torch.no_grad():
                    obj_test = self.evaluation_metric(
                        var_values, y_batch, test_tch)
                    prob_violation_test = self.prob_constr_violation(
                        var_values, y_batch, test_tch)
                    _, var_vio = self.lagrangian(
                        var_values, y_batch, test_tch, alpha, lam, kappa=kappa)

                train_stats.update_test_stats(
                    obj_test, prob_violation_test, var_vio)
                new_row = train_stats.generate_test_row(
                    self._calc_coverage, a_tch, b_tch, alpha, test_tch, eps_tch)
                df_test = pd.concat(
                    [df_test, new_row.to_frame().T], ignore_index=True)

            if step_num < num_iter - 1:
                opt.step()
                opt.zero_grad()
                if scheduler:
                    # DELETE HERE this doesn't break
                    scheduler_.step(temp_lagrangian)

        self._trained = True
        unc_set._trained = True
        unc_set.a.value = a_tch.detach().numpy().copy()
        if not mro_set:
            unc_set.b.value = b_tch.detach().numpy().copy()

        return_eps = eps_tch.detach().numpy().copy() if eps else 1
        return_b_value = unc_set.b.value if not mro_set else None
        return_b_history = b_history if not mro_set else None
        return Result(self, self.new_prob, df, df_test, unc_set.a.value, return_b_value,
                      return_eps, obj.item(), var_values, a_history=a_history,
                      b_history=return_b_history)

    def grid(
        self,
        epslst=settings.EPS_LST_DEFAULT,
        seed=settings.SEED_DEFAULT,
        init_A=settings.INIT_A_DEFAULT,
        init_b=settings.INIT_B_DEFAULT,
        init_alpha=settings.INIT_ALPHA_DEFAULT,
        test_percentage=settings.TEST_PERCENTAGE_DEFAULT,
        num_ys=settings.NUM_YS_DEFAULT,
        solver_args=settings.LAYER_SOLVER,
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
            columns=["Eps"])
        mro_set = self._is_mro_set(unc_set)

        # setup train and test data
        train_set, _, train_tch, test_tch = self._split_dataset(
            unc_set, test_percentage, seed)

        # create cvxpylayer
        cvxpylayer = CvxpyLayer(
            self.new_prob, parameters=self.new_prob.parameters(), variables=self.variables()
        )

        # use all y's
        y_parameters = self.y_parameters()
        num_ys = self.num_ys
        y_batch = self._gen_y_batch(num_ys, y_parameters, 1)
        grid_stats = GridStats()

        lam = 1000 * torch.ones(self.num_g, dtype=settings.DTYPE)
        # initialize torches
        eps_tch = self._gen_eps_tch(1, unc_set, False)
        a_tch_init, b_tch_init, alpha, _ = self._init_torches(1, init_A, init_b,
                                                              init_alpha, train_set, eps_tch,
                                                              False, unc_set)
        for init_eps in epslst:
            eps_tch = torch.tensor(
                1/init_eps, requires_grad=self.train_flag, dtype=settings.DTYPE)
            if mro_set:
                var_values = cvxpylayer(*y_batch, eps_tch*a_tch_init,
                                        solver_args=solver_args)
            else:
                var_values = cvxpylayer(*y_batch, eps_tch*a_tch_init, eps_tch*b_tch_init,
                                        solver_args=solver_args)

            train_stats = TrainLoopStats(
                step_num=np.NAN, train_flag=self.train_flag)
            with torch.no_grad():
                obj_test = self.evaluation_metric(
                    var_values, y_batch, test_tch)
                prob_violation_test = self.prob_constr_violation(
                    var_values, y_batch, test_tch)
                _, var_vio = self.lagrangian(
                    var_values,
                    y_batch,
                    test_tch,
                    alpha,
                    lam,
                )

            train_stats.update_test_stats(obj_test, prob_violation_test,
                                          var_vio)
            grid_stats.update(train_stats, obj_test,
                              eps_tch, eps_tch*a_tch_init, var_values)

            new_row = train_stats.generate_test_row(
                self._calc_coverage, eps_tch*a_tch_init, eps_tch*b_tch_init,  alpha, test_tch,
                eps_tch)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        self._trained = True
        unc_set._trained = True

        if not mro_set:
            unc_set.a.value = (
                grid_stats.mineps * a_tch_init).detach().numpy().copy()
            unc_set.b.value = (
                grid_stats.mineps * b_tch_init).detach().numpy().copy()
        else:
            unc_set.a.value = grid_stats.minT.detach().numpy().copy()
        b_value = None if mro_set else unc_set.b.value
        return Result(
            self,
            self.new_prob,
            df,
            None,
            unc_set.a.value,
            b_value,
            grid_stats.mineps.detach().numpy().copy(),
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
                    _ = self.train()
                    return self.new_prob.solve(solver=solver)
                elif self.uncertain_parameters()[0].uncertainty_set._train:
                    _ = self.train()
                    return self.new_prob.solve(solver=solver)
            prob = self.dualize_constraints()
            return prob.solve(solver=solver)
        return super(RobustProblem, self).solve()


class Result(ABC):
    def __init__(self, prob, probnew, df, df_test, T, b, eps, obj, x, a_history=None,
                 b_history=None):
        self._reform_problem = probnew
        self._problem = prob
        self._df = df
        self._df_test = df_test
        self._A = T
        self._b = b
        self._obj = obj
        self._x = x
        self._eps = eps
        self._a_history = a_history
        self._b_history = b_history

    @property
    def problem(self):
        return self._problem

    @property
    def df(self):
        return self._df

    @property
    def df_test(self):
        return self._df_test

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
        return self._a_history, self._b_history
