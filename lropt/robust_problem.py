import warnings
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sc
import torch
from cvxpy import Parameter as OrigParameter
from cvxpy import error
from cvxpy import settings as s
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.leaf import Leaf
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import INF_OR_UNB_MESSAGE
from cvxpylayers.torch import CvxpyLayer
from joblib import Parallel, delayed

# from pathos.multiprocessing import ProcessPool as Pool
import lropt.settings as settings
from lropt import utils
from lropt.batch import batchify
from lropt.parameter import Parameter
from lropt.shape_parameter import EpsParameter, ShapeParameter
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.remove_uncertainty import RemoveUncertainty
from lropt.uncertainty_sets.mro import MRO

torch.manual_seed(0)
_COL_WIDTH = 79
_HEADER = (
    '='*_COL_WIDTH +
    '\n' +
    ('CVXPY').center(_COL_WIDTH) +
    '\n' +
    ('v' + cvxtypes.version()).center(_COL_WIDTH) +
    '\n' +
    '='*_COL_WIDTH
)
_COMPILATION_STR = (
    '-'*_COL_WIDTH +
    '\n' +
    ('Compilation').center(_COL_WIDTH) +
    '\n' +
    '-'*_COL_WIDTH
)

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

    def __init__(self, step_num, train_flag=True, num_g_total=1):
        def __value_init__(self):
            """
            This is an internal function that either initiates a tensor or a list
            """
            return torch.tensor(0., dtype=settings.DTYPE)
            # return [] if not self.train_flag else torch.tensor(0., dtype=settings.DTYPE)
        self.step_num = step_num
        self.train_flag = train_flag
        self.tot_lagrangian = __value_init__(self)
        self.testval = __value_init__(self)
        self.upper_testval = __value_init__(self)
        self.lower_testval = __value_init__(self)
        self.trainval = __value_init__(self)
        self.prob_violation_test = __value_init__(self)
        self.prob_violation_train = __value_init__(self)
        self.violation_test = __value_init__(self)
        self.violation_train = __value_init__(self)
        self.num_g_total = num_g_total

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
        self.trainval = obj[1].item()
        self.prob_violation_train = prob_violation_train.detach().numpy()
        self.violation_train = sum(train_constraint).item()/self.num_g_total

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
        self.lower_testval = obj_test[0].item()
        self.testval = obj_test[1].item()
        self.upper_testval = obj_test[2].item()
        self.prob_violation_test = prob_violation_test.detach().numpy()
        self.violation_test = sum(var_vio.detach().numpy())/self.num_g_total

    def generate_train_row(self, a_tch, eps_tch, lam, mu, alpha, slack):
        """
        This function generates a new row with the statistics
        """
        row_dict = {
            "Lagrangian_val":   self.tot_lagrangian.item(),
            "Train_val":         self.trainval,
            "Probability_violations_train": self.prob_violation_train,
            "Violations_train":  self.violation_train,
            "Avg_prob_train": np.mean(self.prob_violation_train)
        }
        row_dict["step"] = self.step_num
        row_dict["A_norm"] = np.linalg.norm(
            a_tch.detach().numpy().copy())
        row_dict["lam_list"] = lam.detach().numpy().copy()
        row_dict["mu"] = mu
        row_dict["alpha"] = alpha.item()
        row_dict["slack"] = slack.detach().numpy().copy()
        row_dict["alphagrad"] = alpha.grad
        row_dict["dfnorm"] = np.linalg.norm(a_tch.grad)
        row_dict["gradnorm"] = a_tch.grad
        row_dict["Eps"] = eps_tch.detach().numpy().copy()
        new_row = pd.Series(row_dict)
        return new_row

    def generate_test_row(self, calc_coverage, a_tch, b_tch,
                        alpha, test_tch, eps_tch, uncset, var_values= None):
        """
        This function generates a new row with the statistics
        """
        coverage_test = calc_coverage(
            test_tch, a_tch, b_tch, uncset._rho)
        row_dict = {
            "Test_val":         self.testval,
            "Lower_test": self.lower_testval,
            "Upper_test": self.upper_testval,
            "Probability_violations_test":       self.prob_violation_test,
            "Violations_test":   self.violation_test,
            "Coverage_test":    coverage_test.detach().numpy().item(),
            "Avg_prob_test": np.mean(self.prob_violation_test),
            "var_values": var_values,
            "Eps": eps_tch.detach().numpy().copy()
        }
        row_dict["step"] = self.step_num,
        if not self.train_flag:
            row_dict["Train_val"] = self.trainval
            row_dict["Probability_violations_train"] = self.prob_violation_train
            row_dict["Violations_train"] = self.violation_train
            row_dict["Avg_prob_train"] = np.mean(self.prob_violation_train)
        new_row = pd.Series(row_dict)
        return new_row


class GridStats():
    """
    This class contains useful information for grid search
    """

    def __init__(self):
        self.minval = float("inf")
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
            self.minval = obj[1]
            self.mineps = eps_tch.clone()
            self.minT = a_tch.clone()
            self.var_vals = var_values


class RobustProblem(Problem):
    """Create a Robust Optimization Problem with uncertain variables"""

    _EVAL_INPUT_CASE = Enum("_EVAL_INPUT_CASE", "MEAN EVALMEAN MAX")

    def __init__(
        self, objective, constraints,
        eval_exp=None, train_flag=True, cons_data = None
    ):
        self._trained = False
        self._values = None
        self._numvars = 0
        super(RobustProblem, self).__init__(objective, constraints)
        self._trained = False
        self._values = None
        self.prob_no_uncertainty = None
        self.inverse_data = None
        self.uncertain_chain = None
        self._init = None
        self.train_flag = train_flag
        self._solution = None
        self._status = None
        self._cons_data = cons_data

        self.num_ys = self.verify_y_parameters()
        self._store_variables_parameters()
        self.f, _ = self._gen_torch_exp(objective.expr)
        self.g = []
        self.g_shapes = []
        self.num_g_total = 0
        for constraint in constraints:
            g, has_uncertain_parameters = self._gen_torch_exp(constraint)
            if has_uncertain_parameters:
                self.g.append(g)
                if len(constraint.shape) >=1:
                    self.g_shapes.append(constraint.shape[0])
                    self.num_g_total += constraint.shape[0]
                else:
                    self.g_shapes.append(1)
                    self.num_g_total += 1
        if eval_exp is None:
            self.eval = self.f
        else:
            self.eval, _ = self._gen_torch_exp(eval_exp)
        self.fg_to_lh()

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

    def orig_yparams(self):
        """Find cvxpy y parameters"""
        return [v for v in self.parameters() if isinstance(v, OrigParameter)
                 and not (isinstance(v,Parameter) or
                           isinstance(v,UncertainParameter))]

    def rho_mult_param(self,problem):
        return [v for v in problem.parameters() if isinstance(v, EpsParameter)]

    def gen_y_orig(self,yparams):
            if yparams is not None:
                return [torch.tensor(y.value,
                        dtype=settings.DTYPE,
                        requires_grad=self.train_flag) for y in yparams]
            else:
                return []

    def gen_rho_mult_tch(self,rhoparams):
        if rhoparams is not None:
            return [torch.tensor(rho.value,
                        dtype=settings.DTYPE,
                        requires_grad=self.train_flag) for rho in rhoparams]

    def shape_parameters(self, problem):
        return [v for v in problem.parameters() if isinstance(v, ShapeParameter)]

    def verify_y_parameters(self):
        """
        This function verifies that y and u are in the correct diemsnions.
        """

        y_parameters = self.y_parameters()
        u_parameters = self.uncertain_parameters()
        num_ys = 1
        if len(y_parameters) > 0:
            num_ys = y_parameters[0].data.shape[0]
        #Check that both y and u dimensions are okay
        for params in [y_parameters, u_parameters]:
            for param in params:
                #Fetch the current shape - different from Parameter and UncertainParameter
                if params is y_parameters:
                    curr_shape = param.data.shape[0]
                else:
                    #Skip the check if there is no data
                    if param.uncertainty_set.data is None:
                        continue
                    #Skip the check if _train==False (MRO without training)
                    train_mro = getattr(param.uncertainty_set, "_train", True)
                    if not train_mro:
                        continue
                    curr_shape = param.uncertainty_set.data.shape[0]
                if curr_shape != num_ys:
                    raise ValueError(f"shape inconsistency: expected num_ys={num_ys}, "
                                     f"but got {curr_shape}.")
        return num_ys

    def gen_unique_y(self,y_batch):
        # get unique y's
        y_batch_array = [np.array(ele) for ele in y_batch]
        all_indices = [np.unique(ele,axis=0, return_index=True)[1] for ele in y_batch_array]
        unique_indices = np.unique(np.concatenate(all_indices))
        num_unique_indices = len(unique_indices)
        y_unique = [torch.tensor(ele, dtype=settings.DTYPE)[unique_indices] \
                    for ele in y_batch_array]
        y_unique_array = [ele[unique_indices] for ele in y_batch_array]
        return y_batch_array, num_unique_indices, y_unique, y_unique_array

    def gen_new_var_values(self, num_unique_indices,
                y_unique_array, var_values, batch_int, y_batch_array):
        # create dictionary from unique y's to var_values
        y_to_var_values_dict = {}
        for i in range(num_unique_indices):
            y_to_var_values_dict[tuple(tuple(v[i].flatten())\
                    for v in y_unique_array)] = [v[i] for v in var_values]
        # initialize new var_values
        shapes = [torch.tensor(v.shape) for v in var_values]
        for i in range(len(shapes)):
            shapes[i][0] = batch_int
        new_var_values = [torch.zeros(*shape, dtype=settings.DTYPE) for shape in shapes]

        # populate new_var_values using the dictionary
        for i in range(batch_int):
            values_list = y_to_var_values_dict[tuple(tuple(v[i].flatten())\
                    for v in y_batch_array)]
            for j in range(len(var_values)):
                new_var_values[j][i] = values_list[j]
        return new_var_values

    def fg_to_lh(self):
        """
        Returns l and h function pointers.
        Each of them takes a single x,y,u triplet (i.e. one instance of each)
        """
        # TODO (Amit): Change this function name to a better name
        h_funcs = []
        for g in self.g:
            def hg(*args, **kwargs):
                return (torch.maximum(g(*args) - kwargs["alpha"],
                        torch.tensor(0.0, dtype=settings.DTYPE,
                                     requires_grad=self.train_flag))/kwargs["eta"])

            h_funcs.append(hg)

        self.h = h_funcs
        self.num_g = len(h_funcs)

    #BATCHED
    def _eval_input_b(self, batch_int,eval_func, eval_args, items_to_sample, init_val,
                    eval_input_case, quantiles, **kwargs):
        """
        This function takes decision varaibles, y's, and u's,
            evaluates them and averages them on a given function.

        Args: TODO (Amit): Finish this documentation
            eval_func:
                The function used for evaluation.

        Returns:
            The average among all evaluated J x N pairs
        """

        if eval_input_case != RobustProblem._EVAL_INPUT_CASE.MAX:
            curr_result = eval_func(*eval_args, **kwargs)
        if eval_input_case == RobustProblem._EVAL_INPUT_CASE.MEAN:
            init_val = curr_result
            init_val = torch.mean(init_val,axis=0)
        elif eval_input_case == RobustProblem._EVAL_INPUT_CASE.EVALMEAN:
            init_val = curr_result
            bot_q, top_q = quantiles
            init_val_lower = torch.quantile(init_val, bot_q, axis=0)
            init_val_mean = torch.mean(init_val,axis=0)
            init_val_upper = torch.quantile(init_val, top_q,axis=0)
            return (init_val_lower, init_val_mean, init_val_upper)
        elif eval_input_case == RobustProblem._EVAL_INPUT_CASE.MAX:
            # We want to see if there's a violation: either 1 from previous iterations,
            # or new positive value from now
            # curr_result = (curr_result > 1e-4).float()
            # init_val += curr_result
            # make a setting/variable
            # for j in range(batch_int):
            #     curr_eval_args = _sample_args(eval_args, j, items_to_sample)
            init_val = eval_func(*eval_args, **kwargs)
            if len(init_val.shape) > 1:
                init_val = init_val.T
            init_val = (init_val > settings.TOLERANCE_DEFAULT).float()
        return init_val

    #SERIAL VERSION - TODO - DELETE WHEN _EVAL_INPUT IS VERIFIED TO WORK WELL
    def _eval_input(self, batch_int,eval_func, eval_args, items_to_sample, init_val,
                    eval_input_case, quantiles, **kwargs):
        """
        This function takes decision varaibles, y's, and u's,
            evaluates them and averages them on a given function.

        Args: TODO (Amit): Finish this documentation
            eval_func:
                The function used for evaluation.

        Returns:
            The average among all evaluated J x N pairs
        """
        def _sample_args(eval_args, sample_ind, items_to_sample):
            res = []
            for ind, eval_arg in enumerate(eval_args):
                curr_arg = eval_arg[sample_ind]
                res.append(curr_arg)
                # if ind in items_to_sample:
                #     curr_arg = eval_arg[sample_ind]
                #     # I removed the star_flag, see if makes problems
                #     res.append(curr_arg)
                # else:
                #     res.append(eval_arg)
            return res
        curr_result = {}
        if eval_input_case != RobustProblem._EVAL_INPUT_CASE.MAX:
            for j in range(batch_int):
                curr_eval_args = _sample_args(eval_args, j, items_to_sample)
                curr_result[j] = eval_func(*curr_eval_args, **kwargs)
        if eval_input_case == RobustProblem._EVAL_INPUT_CASE.MEAN:
            init_val = torch.vstack([curr_result[v] for v in curr_result])
            # init_val /= self.num_ys
            init_val = torch.mean(init_val,axis=0)
        elif eval_input_case == RobustProblem._EVAL_INPUT_CASE.EVALMEAN:
            init_val = torch.vstack([curr_result[v] for v in curr_result])
            # init_val /= self.num_ys
            bot_q, top_q = quantiles
            init_val_lower = torch.quantile(init_val, bot_q, axis=0)
            init_val_mean = torch.mean(init_val,axis=0)
            init_val_upper = torch.quantile(init_val, top_q,axis=0)
            return (init_val_lower, init_val_mean, init_val_upper)
        elif eval_input_case == RobustProblem._EVAL_INPUT_CASE.MAX:
            # We want to see if there's a violation: either 1 from previous iterations,
            # or new positive value from now
            # curr_result = (curr_result > 1e-4).float()
            # init_val += curr_result
            # make a setting/variable
            for j in range(batch_int):
                curr_eval_args = _sample_args(eval_args, j, items_to_sample)
                init_val[:,j] = eval_func(*curr_eval_args, **kwargs)
            # init_val = eval_func(*eval_args, **kwargs)
            init_val = (init_val > settings.TOLERANCE_DEFAULT).float()
        return init_val

    def train_objective(self, batch_int, eval_args, items_to_sample):
        return self._eval_input(batch_int,eval_func=self.f, eval_args=eval_args,
                                items_to_sample=items_to_sample, init_val=0,
                                eval_input_case=RobustProblem._EVAL_INPUT_CASE.MEAN, quantiles=None)

    def train_constraint(self, batch_int,eval_args, items_to_sample, alpha, slack, eta, kappa):
        H = torch.zeros(self.num_g_total, dtype=settings.DTYPE)
        for k, h_k in enumerate(self.h):
            init_val = self._eval_input(batch_int,h_k, eval_args, items_to_sample, 0,
                                        RobustProblem._EVAL_INPUT_CASE.MEAN, quantiles=None,
                                        alpha=alpha, eta=eta)
            # init_val = self._eval_input(
            #     h_k, vars, y_params_mat, u_params_mat, 0, RobustProblem._EVAL_INPUT_CASE.MEAN,
            #     False, None, alpha, eta)
            h_k_expectation = init_val + alpha - kappa + \
                slack[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])]
            H[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])] \
                = h_k_expectation
        return H

    def evaluation_metric(self, batch_int, eval_args, items_to_sample, quantiles):
        if (self.eval is None):
            return 0

        return self._eval_input(batch_int,eval_func=self.eval, eval_args=eval_args,
                                items_to_sample=items_to_sample, init_val=0,
                                eval_input_case=RobustProblem._EVAL_INPUT_CASE.EVALMEAN,
                                quantiles=quantiles)

    def prob_constr_violation(self, batch_int, eval_args, items_to_sample, num_us):
        """
        TODO (Amit): Irina, please complete the docstring
        """
        G = torch.zeros((self.num_g_total, num_us),
                        dtype=settings.DTYPE)
        ind=0
        for k, g_k in enumerate(self.g):
            G[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])] = \
            self._eval_input(batch_int, eval_func=g_k, eval_args=eval_args,
                                 items_to_sample=items_to_sample,
                                 init_val=\
                        G[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])],
            eval_input_case=RobustProblem._EVAL_INPUT_CASE.MAX, quantiles=None)
            ind +=1

        # G_max = torch.max(G,dim=0)[0]
        # torch.mean(G_max)
        # G.view(len(self.g),-1).mean(axis=1)
        return G.mean(axis=1)

    # helper function for intermediate version
    # def _udata_to_lst(self, data, batch_size, num_ys, y_parameters):
    #     num_instances = data.shape[0]
    #     batch_int = max(1,min(int(
    #         num_instances*batch_size),10))
    #     random_int = np.random.choice(num_instances, batch_int, replace=False)

    #     # u_params_mat = []
    #     # for i in range(num_instances):
    #     #     u_params_mat.append([data[i, :]])
    #     # return u_params_mat
    #     y_tchs = []
    #     for i in range(len(y_parameters)):
    #         y_tchs.append(torch.tensor(
    #             y_parameters[i].data[random_int], requires_grad=self.train_flag,
    #             dtype=settings.DTYPE))

    #     return batch_int, y_tchs, torch.tensor(data[random_int],
    #                     requires_grad=self.train_flag, dtype=settings.DTYPE)

    def lagrangian(self, batch_int,eval_args, items_to_sample, alpha, slack, lam, mu,
                   eta=settings.ETA_LAGRANGIAN_DEFAULT, kappa=settings.KAPPA_LAGRANGIAN_DEFAULT):
        F = self.train_objective(
            batch_int,eval_args=eval_args, items_to_sample=items_to_sample)
        H = self.train_constraint(batch_int,eval_args=eval_args, items_to_sample=items_to_sample,
                                  alpha=alpha, slack=slack, eta=eta, kappa=kappa)
        return F + lam @ H + (mu/2)*(torch.linalg.norm(H)**2), H.detach()

    # create function for only remove_uncertain reduction
    # def _construct_chain(
    #     self,
    #     solver: Optional[str] = None,
    #     gp: bool = False,
    #     enforce_dpp: bool = True,
    #     ignore_dpp: bool = False,
    #     solver_opts: Optional[dict] = None,
    #     canon_backend: str | None = None,
    # ) -> SolvingChain:
    #     """
    #     Construct the chains required to reformulate and solve the problem.
    #     In particular, this function
    #     # finds the candidate solvers
    #     # constructs the solving chain that performs the
    #        numeric reductions and solves the problem.

    #     Args:

    #     solver : str, optional
    #         The solver to use. Defaults to ECOS.
    #     gp : bool, optional
    #         If True, the problem is parsed as a Disciplined Geometric Program
    #         instead of as a Disciplined Convex Program.
    #     enforce_dpp : bool, optional
    #         Whether to error on DPP violations.
    #     ignore_dpp : bool, optional
    #         When True, DPP problems will be treated as non-DPP,
    #         which may speed up compilation. Defaults to False.
    #     canon_backend : str, optional
    #         'CPP' (default) | 'SCIPY'
    #         Specifies which backend to use for canonicalization, which can affect
    #         compilation time. Defaults to None, i.e., selecting the default
    #         backend.
    #     solver_opts: dict, optional
    #         Additional arguments to pass to the solver.

    #     Returns:
    #         A solving chain
    #     """
    #     candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
    #     self._sort_candidate_solvers(candidate_solvers)
    #     solving_chain = construct_solving_chain(
    #         self,
    #         candidate_solvers,
    #         gp=gp,
    #         enforce_dpp=enforce_dpp,
    #         ignore_dpp=ignore_dpp,
    #         canon_backend=canon_backend,
    #         solver_opts=solver_opts,
    #     )
    #     #
    #     new_reductions = solving_chain.reductions
    #     if self.uncertain_parameters():
    #         # new_reductions = solving_chain.reductions
    #         # Find position of Dcp2Cone or Qp2SymbolicQp
    #         for idx in range(len(new_reductions)):
    #             if type(new_reductions[idx]) in [Dcp2Cone, Qp2SymbolicQp]:
    #                 # Insert Uncertain_Canonicalization before those reductions
    #                 new_reductions.insert(idx, Uncertain_Canonicalization())
    #                 break
    #     # return a chain instead (chain.apply, return the problem and inverse data)
    #     return SolvingChain(reductions=new_reductions)

    def unpack(self, solution) -> None:
        """Updates the problem state given a Solution.

        Updates problem.status, problem.value and value of primal and dual
        variables. If solution.status is in cvxpy.settins.ERROR, this method
        is a no-op.

        Arguments
        _________
        solution : cvxpy.Solution
            A Solution object.

        Raises
        ------
        ValueError
            If the solution object has an invalid status
        """
        if solution.status in s.SOLUTION_PRESENT:
            for v in self.variables():
                v.save_value(solution.primal_vars[v.id])
            for c in self.constraints:
                if c.id in solution.dual_vars:
                    c.save_dual_value(solution.dual_vars[c.id])
            # Eliminate confusion of problem.value versus objective.value.
            # self.objective.value = solution.opt_val
            self._value = solution.opt_val

        elif solution.status in s.INF_OR_UNB:
            for v in self.variables():
                v.save_value(None)
            for constr in self.constraints:
                for dv in constr.dual_variables:
                    dv.save_value(None)
            self._value = solution.opt_val
        else:
            raise ValueError("Cannot unpack invalid solution: %s" % solution)

        self._status = solution.status
        self._solution = solution

    def unpack_results_unc(self, solution, chain, inverse_data,solvername) -> None:
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Arguments
        _________
        solution : object
            The solution returned by applying the chain to the problem
            and invoking the solver on the resulting data.
        chain : SolvingChain
            A solving chain that was used to solve the problem.
        inverse_data : list
            The inverse data returned by applying the chain to the problem.

        Raises
        ------
        cvxpy.error.SolverError
            If the solver failed
        """

        solution = chain.invert(solution, inverse_data)
        if solution.status in s.INACCURATE:
            warnings.warn(
                "Solution may be inaccurate. Try another solver, "
                "adjusting the solver settings, or solve with "
                "verbose=True for more information."
            )
        if solution.status == s.INFEASIBLE_OR_UNBOUNDED:
            warnings.warn(INF_OR_UNB_MESSAGE)
        if solution.status in s.ERROR:
            raise error.SolverError(
                    f"Solver {solvername} failed. "
                    "Try another solver, or solve with verbose=True for more "
                    "information.")
        self.unpack(solution)
        self._solver_stats = SolverStats.from_dict(self._solution.attr, solvername)

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

    def _gen_init(self, train_shape, train_set, init_eps, init_A):
        """
        This is an internal function that calculates init.
        Init means different things depending on eps
            it is an internal function not intended to be used publicly.

        Args:

        train_shape
            Boolean flag indicating if we train a/b or not
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

        #if
        # if not train_shape:
        #     scalar = 1
        #     # return init_A if (init_A is not None) else np.eye(train_set.shape[1])
        cov_len_cond = (len(np.shape(np.cov(train_set.T))) >= 1)
        if (init_eps is None) and (init_A is None):
            if cov_len_cond:
                return sc.linalg.sqrtm(np.cov(train_set.T))
            return np.array([[np.cov(train_set.T)]])

        # scalar = init_eps if init_eps else 1
        # scalar = scalar if train_shape else 1
        mat_shape = train_set.shape[1] if cov_len_cond else 1
        matrix = np.array(init_A) if (
            init_A is not None) else np.eye(mat_shape)
        return matrix

    def _init_torches(self, init_eps, init_A, init_b, init_alpha,\
                       train_set, train_shape):
        """
        This function Initializes and returns a_tch, b_tch, and alpha as tensors
        """
        self._init = self._gen_init(train_shape, train_set, init_eps, init_A)
        init_tensor = torch.tensor(
            self._init, requires_grad=self.train_flag, dtype=settings.DTYPE)
        b_tch = None
        # case = self._calc_mro_case(eps_tch, mro_set, unc_set, init_A)

        # if case == settings.MRO_CASE.NO_MRO:
        if init_b is not None:
            b_tch_data = np.array(init_b)
        else:
            b_tch_data = -np.mean(train_set, axis=0)
        b_tch = torch.tensor(b_tch_data, requires_grad=self.train_flag,
                                dtype=settings.DTYPE)
        a_tch = init_tensor

        # elif case == settings.MRO_CASE.DIFF_A_UNINIT:
        #     a_tch = eps_tch[0]*init_tensor
        #     for k_ind in range(1, unc_set._K):
        #         a_tch = torch.vstack(
        #             (a_tch, eps_tch[k_ind]*self._init))

        # elif case == settings.MRO_CASE.DIFF_A_INIT:
        #     a_tch = eps_tch[0]*torch.tensor(init_A[0:unc_set._m, 0:unc_set._m],
        #                                     dtype=settings.DTYPE)
        #     for k_ind in range(1, unc_set._K):
        #         a_tch = torch.vstack(
        #             (a_tch, eps_tch[k_ind] *
        #                 torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)*unc_set._m,
        #                                     0:unc_set._m],
        #                              dtype=settings.DTYPE)))

        # elif case == settings.MRO_CASE.SAME_A:
        #     a_tch = eps_tch*init_tensor
        #     if unc_set._uniqueA:
        #         if init_A is None:
        #             a_tch = a_tch.repeat(unc_set._K, 1)
        #         elif init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m):
        #             a_tch = a_tch.repeat(unc_set._K, 1)
        #     a = a_tch.detach().numpy()
        #     a_tch = torch.tensor(
        #         a, requires_grad=self.train_flag, dtype=settings.DTYPE)
        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)
        slack = torch.zeros(
            self.num_g_total, requires_grad=self.train_flag, dtype=settings.DTYPE)
        return a_tch, b_tch, alpha, slack

    def _split_dataset(self, unc_set, y_orig_parameters, y_parameters, test_percentage, seed):
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
        np.random.seed(seed)
        num_test = max(1, int(unc_set.data.shape[0]*test_percentage))
        test_indices = np.random.choice(unc_set.data.shape[0],
                                      num_test, replace=False)
        train_indices = [i for i in range(unc_set.data.shape[0]) if i not in test_indices]

        unc_train_set = np.array([unc_set.data[i] for i in train_indices])
        unc_test_set = np.array([unc_set.data[i] for i in test_indices])
        unc_train_tch = torch.tensor(
            unc_set.data[train_indices], requires_grad=self.train_flag, dtype=settings.DTYPE)
        unc_test_tch = torch.tensor(
            unc_set.data[test_indices], requires_grad=self.train_flag, dtype=settings.DTYPE)

        y_train_tchs = []
        y_test_tchs = []
        # if y_orig_parameters is not None:
        #     for i in range(len(y_orig_parameters)):
        #         y_train_tchs.append(torch.vstack([torch.tensor(
        #         y_orig_parameters[i].value, requires_grad=self.train_flag,
        #         dtype=settings.DTYPE)]*len(train_indices)))
        #         y_test_tchs.append(torch.vstack([torch.tensor(
        #             y_orig_parameters[i].value, requires_grad=self.train_flag,
        #             dtype=settings.DTYPE)]*len(test_indices)))

        for i in range(len(y_parameters)):
            y_train_tchs.append(torch.tensor(
                y_parameters[i].data[train_indices], requires_grad=self.train_flag,
                dtype=settings.DTYPE))
            y_test_tchs.append(torch.tensor(
                y_parameters[i].data[test_indices], requires_grad=self.train_flag,
                dtype=settings.DTYPE))

        return unc_train_set, unc_test_set, unc_train_tch, unc_test_tch, y_train_tchs, y_test_tchs

    def _update_iters(self, save_history, a_history, b_history, a_tch, b_tch, eps_tch):
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
        eps = eps_tch.detach().numpy().copy()
        a_history.append(eps*a_tch.detach().numpy().copy())
        b_history.append(b_tch.detach().numpy().copy())

    def _set_train_variables(self, fixb, alpha, slack, a_tch,
                             b_tch, eps_tch, train_size):
        """
        This function sets the variables to be trained in the outer level problem.
        TODO (Amit): complete the docstrings (to Irina)
        """
        if train_size:
            variables = [eps_tch, alpha, slack]
            return variables

        if fixb:
            variables = [a_tch, alpha, slack]
        else:
            variables = [a_tch, b_tch, alpha, slack]

        return variables

    def _gen_batch(self, num_ys, y_parameters, u_data, batch_size, max_size=10000, min_size=1):
        """
        This function generates a set of parameters for each y and u in the family of y's and u's

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
        u_data
            Paired uncertainty dataset
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

        batch_int = max(min(int(num_ys*batch_size),max_size),min_size)
        random_int = np.random.choice(
            num_ys, batch_int, replace=False)
        y_tchs = []
        for i in range(len(y_parameters)):
            y_tchs.append(y_parameters[i].data[random_int])

        u_tch = torch.tensor(u_data[random_int], requires_grad=self.train_flag,
                             dtype=settings.DTYPE)

        return batch_int, y_tchs, u_tch

    def _gen_eps_tch(self, init_eps):
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

        scalar = init_eps if init_eps else 1.0
        eps_tch = torch.tensor(
            scalar, requires_grad=self.train_flag, dtype=settings.DTYPE)

        # if (not mro_set):
        #     return eps_tch

        # if init_eps and eps_tch.shape != torch.Size([]):
        #     return eps_tch

        # eps_tch = eps_tch.repeat(unc_set._K)
        # eps_tch = eps_tch.detach().numpy()
        # eps_tch = torch.tensor(
        #     eps_tch, requires_grad=self.train_flag, dtype=settings.DTYPE)
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

    def _calc_coverage(self, dset, a_tch, b_tch, rho=1):
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
                torch.norm((a_tch.T@torch.linalg.inv(a_tch@a_tch.T)) @ (dset[datind]-
                           b_tch))
                <= rho,
                1,
                0,
            )
        return coverage/dset.shape[0]

    def _store_variables_parameters(self):
        """
        This is an internal function that generates a dictionary of all the variables and parameters
        of the problem from the objective and the constraints.
        The dictionary's keys are the indeces (in order which they are discovered), and the values
        are the variables or the parameters
        """
        def update_vars_params(expr: Expression | cp.constraints.constraint.Constraint,
                               vars_params: dict):
            """
            This function updates vars_params with all the varaibles and params found in expr.
            """
            def safe_check_in_dict(var, vars_dict):
                """
                This function checks if var is in vars_dict.
                For some reason var in vars_dict fails so we check manually
                """
                for value in vars_dict.values():
                    if (var is value):
                        return True
                return False

            _, vars_dict = expr.gen_torch_exp()
            for var_param in vars_dict.vars_dict:
                if safe_check_in_dict(var_param, vars_params):
                    continue
                vars_params[len(vars_params)] = var_param

        vars_params = dict()
        update_vars_params(expr=self.objective.expr, vars_params=vars_params)
        for constraint in self.constraints:
            update_vars_params(expr=constraint, vars_params=vars_params)
        self.vars_params = vars_params

    def _gen_torch_exp(self, expr: Expression):
        """
        This function generates a torch expression to be used by RobustProblem from an expression
        and a vars_dict generated by any cvxpy expression. Also returns a variable indicating
        if this toch_exp has uncertain parameters or not.
        """

        def gen_args_inds_to_pass(vars_params, vars_dict):
            """
            This is a helper function that generates a dictionary from a variable/parameter index
            in vars_params (a dictionary that contains all the problem's variables/parameters)
            to vars_dict (a dictionary that contains all the expression's variables/parameters)
            """
            args_inds_to_pass = dict()
            for global_ind, var_param in vars_params.items():
                if var_param not in vars_dict.vars_dict.keys():
                    continue
                args_inds_to_pass[global_ind] = vars_dict.vars_dict[var_param]
            return args_inds_to_pass

        def wrapped_function(torch_exp, args_inds_to_pass, *args):
            """
            This is the function that wraps the torch expression.

            Args:
                torch_exp:
                    A function (torch expression)
                args_inds_to_pass:
                    A dictionary from index in *args to the args that will be passed.
                    Note that len(args) > len(args_inds_to_pass) is possible.
                *args
                    The arguments of torch_exp
            """

            def _safe_increase_axis(expr: Expression, arg_to_orig_axis: dict) -> None:
                """
                This is an internal function that increases expr.axis by 1 if it is not negative.
                It is needed because we add a new dimension that is reserved for batching, and when
                CVXPY atoms are created, they are unaware of that.
                The increase happens only if batch mode is recognized.
                """

                #Recursively increase the axis of the expression
                for arg in expr.args:
                    if isinstance(arg, Leaf):
                        arg_to_orig_axis[arg] = False
                        continue
                    _safe_increase_axis(arg, arg_to_orig_axis)

                if not hasattr(expr, "axis"):
                    arg_to_orig_axis[expr] = False
                    return
                original_axis = expr.axis
                arg_to_orig_axis[expr] = original_axis

                #If axis=None is equivalent to 0. This is needed to make sure numeric functions
                #do not flatten the inputs.
                if expr.axis is None:
                    expr.axis = 0

                if expr.axis>=0:
                    expr.axis += 1

            def _restore_original_axis(expr: Expression, arg_to_orig_axis: dict) -> None:
                """
                This is an internal function restores the original axis to the expression and all
                of its sub expressions.
                """
                for arg in expr.args:
                    #Recursively restore original axis of the subexpressions
                    _restore_original_axis(arg, arg_to_orig_axis)
                    #Restore the original axis of this expression
                original_axis = arg_to_orig_axis[expr]
                if original_axis is not False:
                    expr.axis = original_axis

            args_to_pass = [None]*len(args_inds_to_pass)
            for key, value in args_inds_to_pass.items():
                args_to_pass[value] = args[key]

            #To make sure batched inputs are processed correctly, we need to update expr.axis
            #(if applicable). It is important to revert it back to the original value when done,
            #hence we save original_axis.
            # expr = torch_exp.args[0]
            # arg_to_orig_axis = {} #Expression (arg) -> original axis dictionary
            # _safe_increase_axis(expr, arg_to_orig_axis)
            res = torch_exp(*args_to_pass)
            #Revert to the original axis if applicable. Note: None is a valid axis (unlike False).
            # _restore_original_axis(expr, arg_to_orig_axis)
            return res

        # vars_dict contains a dictionary from variable/param -> index in *args (for the expression)
        expr = batchify(expr)
        torch_exp, vars_dict = expr.gen_torch_exp()


        # Create a dictionary from index -> variable/param (for the problem)
        args_inds_to_pass = gen_args_inds_to_pass(self.vars_params, vars_dict)

        return partial(wrapped_function, torch_exp, args_inds_to_pass), \
            vars_dict.has_type_in_keys(UncertainParameter)

    def _order_args(self, var_values, y_batch, u_batch):
        """
        This function orders var_values, y_batch, and u_batch according to the order in vars_params.
        """
        args = []
        # self.vars_params is a dictionary, hence unsorted. Need to iterate over it in order
        ind_dict = {
            Variable: 0,
            Parameter: 0,
            UncertainParameter: 0,
        }
        args_dict = {
            Variable: var_values,
            Parameter: y_batch,
            UncertainParameter: u_batch,
        }
        item_to_sample = []
        for i in range(len(self.vars_params)):
            curr_type = type(self.vars_params[i])
            if curr_type == OrigParameter:
                continue
            # This checks for list/tuple or not, to support the fact that currently
            # u_batch is not a list. Irina said in the future this might change.

            # If list or tuple: append the next element
            if isinstance(args_dict[curr_type], tuple) or isinstance(args_dict[curr_type], list):
                append_item = args_dict[curr_type][ind_dict[curr_type]]
                ind_dict[curr_type] += 1
            # If not list-like (e.g. a tensor), append it
            else:
                append_item = args_dict[curr_type]
            args.append(append_item)
            if curr_type != UncertainParameter:
                item_to_sample.append(i)
        return args, item_to_sample

    def _train_loop(self, init_num, **kwargs):
        if kwargs['random_init'] and kwargs['train_shape']:
            if init_num >= 1:
                np.random.seed(kwargs['seed']+init_num)
                shape = kwargs['unc_set']._a.shape
                kwargs['init_A'] = np.random.rand(shape[0],shape[1])
                    #  + 0.01*np.eye(kwargs['u_size'])
                kwargs['init_b'] = np.mean(kwargs['train_set'], axis=0)
        a_history = []
        b_history = []
        df = pd.DataFrame(columns=["step"])
        df_test = pd.DataFrame(columns=["step"])

        eps_tch = self._gen_eps_tch(kwargs['init_eps'])
        a_tch, b_tch, \
            alpha, slack \
            = self._init_torches(kwargs['init_eps'], kwargs['init_A'],
                                 kwargs['init_b'], kwargs['init_alpha'],
                                 kwargs['train_set'], kwargs['train_shape'])

        self._update_iters(kwargs['save_history'], a_history, b_history,
                               a_tch, b_tch, eps_tch)

        variables = self._set_train_variables(kwargs['fixb'], alpha,
                                              slack,
                                              a_tch, b_tch,eps_tch,kwargs["trained_shape"])
        if kwargs['optimizer'] == "SGD":
            opt = settings.OPTIMIZERS[kwargs['optimizer']](
                variables, lr=kwargs['lr'], momentum=kwargs['momentum'])
        else:
            opt = settings.OPTIMIZERS[kwargs['optimizer']](
                variables, lr=kwargs['lr'])
        if kwargs['scheduler']:
            scheduler_ = torch.optim.lr_scheduler.StepLR(
                opt, step_size=kwargs['lr_step_size'], gamma=kwargs['lr_gamma'])
        # y's and cvxpylayer begin
        self.y_parameters()
        num_ys = kwargs["y_train_tch"][0].shape[0]
        lam = kwargs['init_lam'] * torch.ones(self.num_g_total, dtype=settings.DTYPE)
        mu = kwargs['init_mu']
        # use multiple initial points and training. pick lowest eval loss
        # if kwargs["position"]:
        #     p_bar = tqdm(
        #         range(kwargs['num_iter']),
        #         desc=f"run {init_num}: test value N/A, violations N/A",
        #         position=init_num)
        # else:
        #     p_bar = tqdm(
        #         range(kwargs['num_iter']), desc=f"run {init_num}: test value N/A, violations N/A")
        curr_cvar = np.inf
        for step_num in range(kwargs['num_iter']):
            train_stats = TrainLoopStats(
                step_num=step_num, train_flag=self.train_flag, num_g_total=self.num_g_total)

            # generate batched y and u
            # y_batch = self._gen_y_batch(
            #     num_ys, y_parameters, kwargs['y_batch_percentage'])

            batch_int, y_batch, u_batch = self._gen_batch(num_ys,
                    kwargs['y_train_tch'], kwargs['train_set'],
                    kwargs['batch_percentage'], max_size=kwargs["max_batch_size"])
            # slack = torch.maximum(slack, torch.tensor(0.))

            var_values = kwargs['cvxpylayer'](eps_tch, \
                                    *kwargs['y_orig_tch'], \
                *y_batch, a_tch, b_tch,solver_args=kwargs['solver_args'])

            eval_args, items_to_sample = self._order_args(var_values=var_values,
                                                          y_batch=y_batch, u_batch=u_batch)
            temp_lagrangian, train_constraint_value = self.lagrangian(batch_int, eval_args,
                                                                      items_to_sample,
                                                                      alpha, slack, lam,
                                                                      mu,
                                                                      eta=kwargs['eta'],
                                                                      kappa=kwargs['kappa'])
            temp_lagrangian.backward()
            with torch.no_grad():
                obj = self.evaluation_metric(
                    batch_int, eval_args, items_to_sample, kwargs['quantiles'])
                prob_violation_train = self.prob_constr_violation(batch_int,
                  eval_args, items_to_sample,num_us=len(u_batch))

            train_stats.update_train_stats(temp_lagrangian.detach().numpy(
            ).copy(), obj, prob_violation_train, train_constraint_value)

            # lam = torch.maximum(lam + step_lam*train_constraint_value,
            #                    torch.zeros(self.num_g, dtype=settings.DTYPE))

            if step_num % kwargs['aug_lag_update_interval'] == 0:
                if torch.norm(train_constraint_value) <= \
                    kwargs['lambda_update_threshold']*curr_cvar:
                    curr_cvar= torch.norm(train_constraint_value)
                    lam = lam + torch.minimum(mu*train_constraint_value,
                                        kwargs['lambda_update_max']*\
                                            torch.ones(self.num_g_total,
                                                        dtype=settings.DTYPE))
                else:
                    mu = kwargs['mu_multiplier']*mu

            new_row = train_stats.generate_train_row(
                a_tch, eps_tch, lam, mu, alpha, slack)
            df = pd.concat(
                [df, new_row.to_frame().T], ignore_index=True)

            self._update_iters(kwargs['save_history'], a_history, b_history,
                               a_tch, b_tch, eps_tch)

            if step_num % kwargs['test_frequency'] == 0:
                batch_int, y_batch, u_batch = self._gen_batch(
                    kwargs["y_test_tch"][0].shape[0],
                    kwargs['y_test_tch'], kwargs['test_set'], 1,max_size=kwargs["max_batch_size"])
                var_values = kwargs['cvxpylayer'](
                    eps_tch,*kwargs['y_orig_tch'],\
                        *y_batch, a_tch, b_tch,
                        solver_args=kwargs['solver_args'])

                with torch.no_grad():
                    # test_u = kwargs['test_tch']
                    test_args, test_to_sample = self._order_args(var_values=var_values,
                                                                 y_batch=y_batch, u_batch=u_batch)
                    obj_test = self.evaluation_metric(batch_int,test_args, test_to_sample,
                                                      kwargs['quantiles'])
                    prob_violation_test = self.prob_constr_violation(batch_int,
                            test_args, test_to_sample,num_us=len(u_batch))
                    _, var_vio = self.lagrangian(batch_int,test_args,
                                                 test_to_sample, alpha, slack,
                                                   lam, mu,
                                                 eta=kwargs['eta'],
                                                 kappa=kwargs['kappa'])
                # p_bar.set_description(
                #     f"run {init_num}:"
                #     f" test value {round(obj_test[1].item(),3)}"
                #     f", violations {round(sum(prob_violation_test).item()/self.num_g,3)}")
                train_stats.update_test_stats(
                    obj_test, prob_violation_test, var_vio)
                new_row = train_stats.generate_test_row(
                    self._calc_coverage, a_tch, b_tch, alpha,
                    u_batch, eps_tch, kwargs['unc_set'], var_values)
                df_test = pd.concat(
                    [df_test, new_row.to_frame().T], ignore_index=True)


            if step_num < kwargs['num_iter'] - 1:
                opt.step()
                opt.zero_grad()
                with torch.no_grad():
                    newval = torch.clamp(slack, min=0., max=torch.inf)
                    slack.copy_(newval)
                    if kwargs['trained_shape']:
                        neweps_tch = torch.clamp(eps_tch, min=0.001)
                        eps_tch.copy_(neweps_tch)
                if kwargs['scheduler']:
                    scheduler_.step()

        if sum(var_vio.detach().numpy())/self.num_g_total <= kwargs["kappa"]:
            fin_val = obj_test[1].item()
        else:
            fin_val = obj_test[1].item() + 10*abs(sum(var_vio.detach().numpy()))
        a_val = a_tch.detach().numpy().copy()
        b_val = b_tch.detach().numpy().copy()
        eps_val = eps_tch.detach().numpy().copy() if kwargs['trained_shape'] else 1
        param_vals = (a_val, b_val, eps_val, obj_test[1].item())
        # tqdm.write("Testing objective: {}".format(obj_test[1].item()))
        # tqdm.write("Probability of constraint violation: {}".format(
        #            prob_violation_test))
        return df, df_test, a_history, b_history, \
            param_vals, fin_val, var_values, mu

    def train(
        self,
        train_size=settings.TRAIN_SIZE_DEFAULT,
        train_shape=settings.TRAIN_SHAPE_DEFAULT,
        fixb=settings.FIXB_DEFAULT,
        num_iter=settings.NUM_ITER_DEFAULT,  # Used to be "step"
        num_iter_size = settings.NUM_ITER_SIZE_DEFAULT,
        lr=settings.LR_DEFAULT,
        lr_size = settings.LR_SIZE_DEFAULT,
        scheduler=settings.SCHEDULER_STEPLR_DEFAULT,
        momentum=settings.MOMENTUM_DEFAULT,
        optimizer=settings.OPT_DEFAULT,
        init_eps=settings.INIT_EPS_DEFAULT,
        init_A=settings.INIT_A_DEFAULT,
        init_b=settings.INIT_B_DEFAULT,
        save_history=settings.SAVE_HISTORY_DEFAULT,
        seed=settings.SEED_DEFAULT,
        init_lam=settings.INIT_LAM_DEFAULT,
        init_mu=settings.INIT_MU_DEFAULT,
        mu_multiplier=settings.MU_MULTIPLIER_DEFAULT,
        init_alpha=settings.INIT_ALPHA_DEFAULT,
        eta=settings.ETA_LAGRANGIAN_DEFAULT,
        kappa=settings.KAPPA_DEFAULT,  # (originall target_cvar)
        random_init=settings.RANDOM_INIT_DEFAULT,
        num_random_init=settings.NUM_RANDOM_INIT_DEFAULT,
        test_frequency=settings.TEST_FREQUENCY_DEFAULT,
        test_percentage=settings.TEST_PERCENTAGE_DEFAULT,
        step_lam=settings.STEP_LAM_DEFAULT,
        batch_percentage=settings.BATCH_PERCENTAGE_DEFAULT,
        solver_args=settings.LAYER_SOLVER,
        n_jobs=settings.N_JOBS,
        quantiles=settings.QUANTILES,
        lr_step_size=settings.LR_STEP_SIZE,
        lr_gamma=settings.LR_GAMMA,
        position=settings.POSITION,
        parallel=settings.PARALLEL,
        aug_lag_update_interval = settings.UPDATE_INTERVAL,
        lambda_update_threshold = settings.LAMBDA_UPDATE_THRESHOLD,
        lambda_update_max = settings.LAMBDA_UPDATE_MAX,
        max_batch_size = settings.MAX_BATCH_SIZE
    ):
        r"""
        Trains the uncertainty set parameters to find optimal set w.r.t. lagrangian metric

        Parameters TODO (Amit): Irina - update all the variables
        -----------
        train_size : bool, optional
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

        self._validate_uncertain_parameters()

        unc_set = self._get_unc_set()
        self.remove_uncertainty(override=True)
        self._validate_unc_set_T(unc_set)
        unc_train_set, unc_test_set, unc_train_tch, unc_test_tch, \
            y_train_tchs, y_test_tchs = self._split_dataset(
            unc_set, self.orig_yparams(), self.y_parameters(), test_percentage, seed)
        u_size = unc_train_set.shape[1]
        self._is_mro_set(unc_set)
        y_orig_torches = self.gen_y_orig(self.orig_yparams())
        rho_mult_params = self.rho_mult_param(self.prob_no_uncertainty)
        rho_mult_tch = self.gen_rho_mult_tch(rho_mult_params)

        cvxpylayer = CvxpyLayer(self.prob_no_uncertainty,
                                parameters=rho_mult_params + \
                                self.orig_yparams() + self.y_parameters()
                                + self.shape_parameters(self.prob_no_uncertainty),
                                variables=self.variables())
        num_random_init = num_random_init if random_init else 1
        num_random_init = num_random_init if train_shape else 1
        kwargs = {"train_size": train_size,
                   "trained_shape": not train_shape, "train_shape": train_shape,
                  "init_A": init_A, "init_b": init_b,
                  "init_eps": init_eps, "unc_set": unc_set,
                  "random_init": random_init,
                  "seed": seed, "u_size": u_size, "train_set": unc_train_set,
                  "init_alpha": init_alpha, "save_history": save_history,
                  "fixb": fixb, "optimizer": optimizer,
                  "lr": lr, "momentum": momentum,
                  "scheduler": scheduler, "init_lam":
                  init_lam, "init_mu": init_mu,
                  "num_iter": num_iter,
                  "batch_percentage": batch_percentage,
                  "cvxpylayer": cvxpylayer, "solver_args": solver_args,
                  "kappa": kappa, "test_frequency": test_frequency,
                  "test_set": unc_test_set, "mu_multiplier": mu_multiplier,
                  "quantiles": quantiles, "lr_step_size": lr_step_size,
                  "lr_gamma": lr_gamma, "eta": eta,
                    "position": position, "test_percentage": test_percentage,
                    "y_train_tch": y_train_tchs, "y_test_tch": y_test_tchs,
                    "y_orig_tch": y_orig_torches, "rho_mult_tch": rho_mult_tch,
                    "aug_lag_update_interval": aug_lag_update_interval,
                    "lambda_update_threshold":lambda_update_threshold,
                    "lambda_update_max":lambda_update_max,
                     "max_batch_size":max_batch_size }

        # Debugging code - one iteration
        # res = self._train_loop(0, **kwargs)
        # Debugging code - serial
        if not parallel:
            res = []
            for init_num in range(num_random_init):
                res.append(self._train_loop(init_num, **kwargs))
        # n_jobs = utils.get_n_processes() if parallel else 1
        # pool_obj = Pool(processes=n_jobs)
        # loop_fn = partial(self._train_loop, **kwargs)
        # res = pool_obj.map(loop_fn, range(num_random_init))
        # Joblib version
        else:
            n_jobs = utils.get_n_processes() if parallel else 1
            res = Parallel(n_jobs=n_jobs)(delayed(self._train_loop)(
                init_num, **kwargs) for init_num in range(num_random_init))
        df, df_test, a_history, b_history, param_vals, \
            fin_val, var_values, mu_val = zip(*res)
        index_chosen = np.argmin(np.array(fin_val))
        self._trained = True
        unc_set._trained = True
        unc_set.a.value = param_vals[index_chosen][0]
        unc_set.b.value = param_vals[index_chosen][1]
        return_eps = param_vals[index_chosen][2]

        if train_shape and train_size:
            kwargs["trained_shape"] = True
            kwargs["train_shape"] = False
            kwargs["init_A"] = unc_set.a.value
            kwargs["init_b"] = unc_set.b.value
            # kwargs["init_eps"] = 1
            kwargs["random_init"] = False
            kwargs["lr"] = lr_size if lr_size else lr
            kwargs["num_iter"] = num_iter_size if num_iter_size else num_iter
            kwargs["init_mu"] = mu_val[index_chosen]
            if not parallel:
                res = []
                for init_num in range(num_random_init):
                    res.append(self._train_loop(init_num, **kwargs))
            else:
                res = Parallel(n_jobs=n_jobs)(delayed(self._train_loop)(
                init_num, **kwargs) for init_num in range(num_random_init))
            df_s, df_test_s, a_history_s, b_history_s, param_vals_s, \
                fin_val_s, var_values_s, mu_s = zip(*res)
            return_eps = param_vals_s[0][2]
            return_df = pd.concat([df[index_chosen],df_s[0]])
            return_df_test = pd.concat([df_test[index_chosen], df_test_s[0]])
            return_a_history = a_history[index_chosen] + a_history_s[0]
            return_b_history = b_history[index_chosen] + b_history_s[0]
            return Result(self, self.prob_no_uncertainty, return_df,
                      return_df_test, unc_set.a.value,
                      unc_set.b.value,
                      return_eps, param_vals[0][3],
                      var_values[index_chosen] + var_values_s[0],
                      a_history=return_a_history,
                      b_history=return_b_history)
        return Result(self, self.prob_no_uncertainty, df[index_chosen],
                      df_test[index_chosen], unc_set.a.value,
                      unc_set.b.value,
                      return_eps, param_vals[index_chosen][3],
                      var_values[index_chosen],
                      a_history=a_history[index_chosen],
                      b_history=b_history[index_chosen])

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
        quantiles=settings.QUANTILES,
        newdata = settings.NEWDATA_DEFAULT,
        eta = settings.ETA_LAGRANGIAN_DEFAULT
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
        self.remove_uncertainty(override=True)
        mro_set = self._is_mro_set(unc_set)

        self._validate_unc_set_T(unc_set)
        df = pd.DataFrame(
            columns=["Eps"])
        unc_train_set, unc_test_set, unc_train_tch, unc_test_tch,\
            y_train_tchs, y_test_tchs = self._split_dataset(
            unc_set, self.orig_yparams(), self.y_parameters(), test_percentage, seed)
        y_orig_torches = self.gen_y_orig(self.orig_yparams())
        rho_mult_params = self.rho_mult_param(self.prob_no_uncertainty)
        rho_mult_tch = self.gen_rho_mult_tch(rho_mult_params)

        if newdata is not None:
            train_set, y_set = newdata
            u_batch = torch.tensor(train_set, requires_grad=self.train_flag, dtype=settings.DTYPE)
            if not isinstance(y_set, list):
                y_batch = [torch.tensor(y_set, requires_grad=self.train_flag, dtype=settings.DTYPE)]
            else:
                y_batch = [torch.tensor(y, requires_grad=self.train_flag,
                                        dtype=settings.DTYPE) for y in y_set]
            batch_int = train_set.shape[0]

        else:
        # setup train and test data
            # use all y's
            batch_int, y_batch, u_batch= self._gen_batch(unc_test_set.shape[0],
                                y_test_tchs, unc_test_set,1)

        batch_int_train, y_batch_train, u_batch_train = \
                self._gen_batch(unc_train_set.shape[0],
                                y_train_tchs, unc_train_set,1)
        # create cvxpylayer

        cvxpylayer = CvxpyLayer(self.prob_no_uncertainty,
                                parameters=rho_mult_params + \
                                self.orig_yparams() + self.y_parameters() +
                                self.shape_parameters(self.prob_no_uncertainty),
                                variables=self.variables())


        grid_stats = GridStats()

        lam = 1000 * torch.ones(self.num_g_total, dtype=settings.DTYPE)
        # initialize torches
        eps_tch = self._gen_eps_tch(1)
        a_tch_init, b_tch_init, alpha, slack = self._init_torches(1,
            init_A, init_b,init_alpha, unc_train_set, True)

        y_batch_array, num_unique_indices, y_unique, \
            y_unique_array = self.gen_unique_y(y_batch)

        y_batch_array_t, num_unique_indices_t, y_unique_t, \
            y_unique_array_t = self.gen_unique_y(y_batch_train)

        for init_eps in epslst:
            eps_tch = torch.tensor(
                init_eps, requires_grad=self.train_flag, dtype=settings.DTYPE)

            if mro_set:
                unc_set._rho_mult.value = init_eps
                rho_mult_tch = self.gen_rho_mult_tch(rho_mult_params)
                var_values = cvxpylayer(*rho_mult_tch, *y_orig_torches,
                                        *y_unique, a_tch_init,b_tch_init,
                                        solver_args=solver_args)
                var_values_t = cvxpylayer(*rho_mult_tch, *y_orig_torches,
                                        *y_unique_t, a_tch_init,b_tch_init,
                                        solver_args=solver_args)
            else:
                var_values = cvxpylayer(*rho_mult_tch, *y_orig_torches,
                                         *y_unique, eps_tch*a_tch_init,
                                           b_tch_init,
                                        solver_args=solver_args)
                var_values_t = cvxpylayer(*rho_mult_tch, *y_orig_torches,
                                         *y_unique_t, eps_tch*a_tch_init,
                                           b_tch_init,
                                        solver_args=solver_args)

            new_var_values = self.gen_new_var_values(num_unique_indices,
                    y_unique_array, var_values, batch_int, y_batch_array)
            new_var_values_t = self.gen_new_var_values(num_unique_indices_t,
                     y_unique_array_t, var_values_t, batch_int_train,
                       y_batch_array_t)

            train_stats = TrainLoopStats(
                step_num=np.NAN, train_flag=self.train_flag, num_g_total=self.num_g_total)
            with torch.no_grad():
                test_args, test_to_sample = self._order_args(
                    var_values=new_var_values, y_batch=y_batch, u_batch=u_batch)
                obj_test = self.evaluation_metric(batch_int,
                    test_args, test_to_sample, quantiles)
                prob_violation_test = self.prob_constr_violation(batch_int,
                    test_args, test_to_sample,num_us=batch_int)
                _, var_vio = self.lagrangian(batch_int,
                    test_args, test_to_sample,
                    alpha,
                    slack,
                    lam,
                    1, eta
                )

                test_args_t, test_to_sample_t = self._order_args(
                    var_values=new_var_values_t,
                y_batch=y_batch_train, u_batch=u_batch_train)
                obj_train = self.evaluation_metric(batch_int_train,
                    test_args_t, test_to_sample_t, quantiles)
                prob_violation_train = \
                    self.prob_constr_violation(batch_int_train,
                    test_args_t, test_to_sample_t,num_us=batch_int_train)
                _, var_vio_train = self.lagrangian(batch_int_train,
                    test_args_t, test_to_sample_t,
                    alpha,
                    slack,
                    lam,
                    1, eta
                )

            train_stats.update_test_stats(obj_test, prob_violation_test,
                                          var_vio)
            train_stats.update_train_stats(None, obj_train,
                                prob_violation_train,var_vio_train)
            grid_stats.update(train_stats, obj_test,
                              eps_tch, eps_tch*a_tch_init, var_values[1])

            new_row = train_stats.generate_test_row(
                self._calc_coverage, eps_tch*a_tch_init,b_tch_init, alpha, unc_test_tch,
                eps_tch, unc_set, var_values[1])
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        self._trained = True
        unc_set._trained = True

        unc_set.a.value = (
            grid_stats.mineps * a_tch_init).detach().numpy().copy()
        unc_set.b.value = (
            b_tch_init).detach().numpy().copy()
        # else:
        #     unc_set.a.value = grid_stats.minT.detach().numpy().copy()
        b_value = unc_set.b.value
        return Result(
            self,
            self.prob_no_uncertainty,
            df,
            None,
            unc_set.a.value,
            b_value,
            grid_stats.mineps.detach().numpy().copy(),
            grid_stats.minval,
            grid_stats.var_vals,
        )

    def remove_uncertainty(self,override = False):
        """
        This function canonizes a problem and saves it to self.prob_no_uncertainty

        Args:

        override
            If True, will override current prob_no_uncertainty.
            If False and prob_no_uncertainty exists, does nothing.

        Returns:

        None
        """
        from lropt.uncertain_canon.uncertain_canonicalization import UncertainCanonicalization
        if (not override) and (self.prob_no_uncertainty):
            return
        if self.uncertain_parameters():
            unc_reductions = []
            if type(self.objective) == Maximize:
                unc_reductions += [FlipObjective()]
            # unc_reductions += [RemoveUncertainty()]
            unc_reductions += [UncertainCanonicalization(),RemoveUncertainty()]
            newchain = Chain(self, reductions=unc_reductions)
            self.prob_no_uncertainty, self.inverse_data = newchain.apply(self)
            self.uncertain_chain = newchain

    def solve(self,
               solver: str = None,
               warm_start: bool = True,
               verbose: bool = False,
               gp: bool = False,
               qcp: bool = False,
               requires_grad: bool = False,
               enforce_dpp: bool = False,
               ignore_dpp: bool = False,
               canon_backend: str | None = None,
               **kwargs):
        """
        This function solves the robust problem, and dualizes it first if it has
        not been dualized

        Returns: the solution to the original problem
        """
        if self.uncertain_parameters():
            solver_func = self._helper_solve
            if self.prob_no_uncertainty is None:
                # if no data is passed, no training is needed
                if self.uncertain_parameters()[0].uncertainty_set.data is None:
                    self.remove_uncertainty()
                else:
                    # if not MRO set and not trained
                    if not type(self.uncertain_parameters()[0].uncertainty_set) == MRO:
                        _ = self.train()
                        for y in self.y_parameters():
                            y.value = y.data[0]
                    # if MRO set and training needed
                    elif self.uncertain_parameters()[0].uncertainty_set._train:
                        _ = self.train()
                        for y in self.y_parameters():
                            y.value = y.data[0]
                    else:
                        # if MRO set and no training needed
                        self.remove_uncertainty()
        else:
            solver_func = super(RobustProblem, self).solve
        solver_func(solver=solver, warm_start=warm_start, verbose=verbose, gp=gp, qcp=qcp,
                                            requires_grad=requires_grad, enforce_dpp=enforce_dpp,
                                            ignore_dpp=ignore_dpp, canon_backend=canon_backend,
                                            **kwargs)

    def _helper_solve(self,
                solver: str = None,
                warm_start: bool = True,
                verbose: bool = False,
                gp: bool = False,
                qcp: bool = False,
                requires_grad: bool = False,
                enforce_dpp: bool = False,
                ignore_dpp: bool = False,
                canon_backend: str | None = None,
                **kwargs):
        """
        This function solves the dualized robust problem

        Returns: the solution to the original problem
        """
        prob = self.prob_no_uncertainty
        for y in prob.parameters():
            if y.value is None:
                y.value = y.data[0]
        inverse_data = self.inverse_data
        uncertain_chain = self.uncertain_chain
        prob.solve(solver,warm_start,verbose,gp,qcp,requires_grad,enforce_dpp,ignore_dpp,canon_backend,**kwargs)
        solvername = prob.solver_stats.solver_name
        solution = prob._solution
        self.unpack_results_unc(solution, uncertain_chain, inverse_data,solvername)
        return self.value

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

@dataclass
class SolverStats:
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    solver_name : str
        The name of the solver.
    solve_time : double
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : double
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    extra_stats : object
        Extra statistics specific to the solver; these statistics are typically
        returned directly from the solver, without modification by CVXPY.
        This object may be a dict, or a custom Python object.
    """

    solver_name: str
    solve_time: Optional[float] = None
    setup_time: Optional[float] = None
    num_iters: Optional[int] = None
    extra_stats: Optional[dict] = None

    @classmethod
    def from_dict(cls, attr: dict, solver_name: str) -> "SolverStats":
        """Construct a SolverStats object from a dictionary of attributes.

        Parameters
        ----------
        attr : dict
            A dictionary of attributes returned by the solver.
        solver_name : str
            The name of the solver.

        Returns
        -------
        SolverStats
            A SolverStats object.
        """
        return cls(
            solver_name,
            solve_time=attr.get(s.SOLVE_TIME),
            setup_time=attr.get(s.SETUP_TIME),
            num_iters=attr.get(s.NUM_ITERS),
            extra_stats=attr.get(s.EXTRA_STATS),
        )
