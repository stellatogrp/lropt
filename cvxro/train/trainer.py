import gc
from abc import ABC

import numpy as np
import pandas as pd
import scipy as sc
import torch
import torch.nn as nn
from cvxpy import Parameter as OrigParameter
from cvxpylayers.torch import CvxpyLayer
from joblib import Parallel, delayed

import cvxro.train.settings as s
from cvxro import RobustProblem
from cvxro.parameter import ContextParameter, ShapeParameter, SizeParameter
from cvxro.train.predictors.covpred import CovPredictor
from cvxro.train.predictors.linear import LinearPredictor
from cvxro.train.settings import DEFAULT_SETTINGS as DS

# Import settings directly to use throughout the file
from cvxro.train.simulator import DefaultSimulator
from cvxro.train.utils import (
    EVAL_INPUT_CASE,
    eval_input,
    eval_prob_constr_violation,
    get_n_processes,
    reduce_step_size,
    restore_step_size,
    take_step,
    undo_step,
)
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.scenario import Scenario
from cvxro.utils import unique_list
from cvxro.violation_checker.utils import CONSTRAINT_STATUS
from cvxro.violation_checker.violation_checker import ViolationChecker


# Logistic regression model defined externally for reuse and gradient flow
class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class Trainer:
    """Create a class to handle training"""

    def __init__(self, problem: RobustProblem):
        if not (self.count_unq_uncertain_param(problem) == 1):
            raise ValueError("Must have a single uncertain parameter " + "for training")
        torch.set_default_dtype(torch.double)
        self.unc_set = self.uncertain_parameters(problem)[0].uncertainty_set
        self._validate_unc_set_T()

        problem.remove_uncertainty()
        self.orig_problem = problem
        self.problem_canon = problem.problem_canon
        self.problem_no_unc = problem.problem_no_unc

        self.f = self.problem_canon.f
        self.g = self.problem_canon.g
        self.h = self.problem_canon.h
        self.g_shapes = self.problem_canon.g_shapes
        self.num_g_total = self.problem_canon.num_g_total
        self.eval = self.problem_canon.eval

        self._x_parameters = self.x_parameters(self.orig_problem)
        self._cp_parameters = self.cp_params(self.orig_problem)
        self._shape_parameters = self.shape_parameters(self.problem_no_unc)
        self._rho_mult_parameter = self.rho_mult_param(self.problem_no_unc)

        self.train_flag = True

        self.u_train_tch = None
        self.u_test_tch = None
        self.u_train_set = None
        self.u_test_set = None
        self.x_train_tch = None
        self.x_test_tch = None
        self._x_tchs_init = True

    def monte_carlo(
        self,
        rho_tch,
        alpha,
        a_tch,
        b_tch,
        seed ):
        """This function calls the loss and constraint function, and returns
        an error if an infeasibility is encountered. This is infeasibility
        dependent on the testing data-set.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # remove for loop, set batch size to trials
        cost, constraint_cost, x_hist, z_hist, constraint_status, \
            avg_cost, prob_vio, u_hist ,cvar_cost,in_sample_cost, worst_cost = (
            self.loss_and_constraints(
                a_tch=a_tch,
                b_tch=b_tch,
                seed=seed,
                rho_tch=rho_tch,
                alpha=alpha,
                batch_size = self.settings.test_batch_size          )
        )
        if constraint_status is CONSTRAINT_STATUS.INFEASIBLE:
            # raise InfeasibleConstraintException(
            #     "Found an infeasible constraint during a call to monte_carlo."
            #     + "Possibly an infeasible uncertainty set initialization."
            #     + "Or infeasibility encountered in the testing set"
            # )
            print("Infeasible init")
        return cost, constraint_cost, x_hist, z_hist, avg_cost, \
            prob_vio, u_hist, cvar_cost, in_sample_cost, worst_cost

    def loss_and_constraints(
        self,
        rho_tch,
        alpha,
        a_tch,
        b_tch,
        seed,
        batch_size
    ):
        """
        This function propagates the system state, calculates the costs,
        and checks feasibility
        Args:
            rho_tch
                size of the uncertainty set. A torch tensor.
            alpha
                cvar variable for the constraint cost.
            a_tch, b_tch
                the initialized size and shape parameters
            seed
                RNG seed.
        Returns:
            cost
                objective function cost, averaged across time steps
            constraint_cost
                cost of constraint violation, averaged across time steps
            x_hist
                the list of all systems states over time
            z_hist
                the list of all decisions over time
            constraints_status
                whether any constraint was infeasible
            avg_cost
                evaluation function cost, averaged across time steps
            prob_vio
                probability of constraint violation, averaged across time steps
            u_0
                uncertainty data for the single-stage case
        """
        torch.manual_seed(seed)

        if self._multistage:
            u_0 = 0
            x_0 = self.simulator.init_state(batch_size, seed,
                                            **self.settings.kwargs_simulator)
        else:
            batch_int, x_0, u_0 = self.simulator.init_state(batch_size, seed,
                                                            **self.settings.kwargs_simulator)
            self.settings.kwargs_simulator["batch_int"] = batch_int

        if not isinstance(x_0, list):
            x_0 = [x_0]
        x_0 = [x.clone().detach() for x in x_0]
        if self.settings.contextual:
            a_tch, b_tch, radius = self.create_predictor_tensors(x_0)

        cost = 0.0
        constraint_cost = 0.0
        cvar_cost = 0.0
        worst_cost = 0.0
        in_sample_cost = 0.0
        if self._default_simulator:
            avg_cost = torch.tensor([0, 0, 0], dtype=s.DTYPE)
        else:
            avg_cost = 0.0
        prob_vio = 0.0
        x_t = x_0
        x_hist = [[xval.detach().numpy().copy() for xval in x_t.copy()]]
        z_hist = []
        for t in range(self.settings.time_horizon):
            z_t = self.cvxpylayer(rho_tch, *self.cp_param_tch, *x_t, a_tch, b_tch,
                                  solver_args=self.settings.solver_args)
            constraints_status = self.violation_checker.check_constraints(
                z_batch=z_t,
                rho_mult_parameter=self._rho_mult_parameter,
                rho_tch=rho_tch,
                cp_parameters=self._cp_parameters,
                cp_param_tch=self.cp_param_tch,
                x_parameters=self._x_parameters,
                x_batch=x_t,
                shape_parameters=self._shape_parameters,
                shape_torches=[a_tch, b_tch],
            )
            z_t = self._reduce_variables(z_t)
            if not self._multistage:
                eval_args = self.order_args(z_t, x_t, u_0)
                self.settings.kwargs_simulator["eval_args"] = eval_args
                self.settings.kwargs_simulator["u_train"] = u_0

            x_t = self.simulator.simulate(x_t, z_t, **self.settings.kwargs_simulator)
            in_sample_cost += self.simulator.in_sample_obj(
                x_t, z_t, **self.settings.kwargs_simulator)
            cvar_cur = self.simulator.stage_cost_cvar(x_t, z_t, **self.settings.kwargs_simulator)
            cvar_cost += cvar_cur[0]
            worst_cost += cvar_cur[1]
            cost += self.simulator.stage_cost(x_t, z_t, **self.settings.kwargs_simulator)
            avg_cost += self.simulator.stage_cost_avg(x_t, z_t, **self.settings.kwargs_simulator)
            if self.settings.kwargs_simulator is not None:
                constraint_kwargs = self.settings.kwargs_simulator.copy()
            else:
                constraint_kwargs = {}

            # TODO (bart): this is not ideal since we are copying the kwargs
            constraint_kwargs["alpha"] = alpha
            if self.settings.constraint_cvar:
                constraint_cost += self.simulator.constraint_cost(x_t, z_t, **constraint_kwargs)
            elif self.settings.delage_coverage:
                input_tensors = self.create_input_tensors(x_t)
                coverage_loss, _ = self.dist_loss(a_tch,b_tch,radius,input_tensors,u_0)
                constraint_cost += coverage_loss


            prob_vio += self.simulator.prob_constr_violation(x_t, z_t,
                                                             **self.settings.kwargs_simulator)
            x_hist.append([xval.detach().numpy().copy() for xval in x_t])
            z_hist.append(
                [z.detach() for z in z_t] if isinstance(z_t, list) else z_t.detach()
            )
            if self.settings.contextual:
                a_tch, b_tch,radius = self.create_predictor_tensors(x_t)
            self._a_tch = a_tch
            self._b_tch = b_tch
            self._cur_x = x_t
            self._cur_u = u_0
        return cost, constraint_cost, x_hist, z_hist, \
            constraints_status, avg_cost, prob_vio, u_0, \
                cvar_cost, in_sample_cost, worst_cost

    def _validate_unc_set_T(self):
        """
        This function checks if paramaterT is not empty.

        Args:
            unc_set
                Uncertainty set

        Returns:
            None.

        Raises:
            ValueError if there is no a in the uncertainty set, or there is no uncertain data
        """
        if self.unc_set.data is None:
            raise ValueError("Cannot train without uncertainty set data")
        if self.unc_set.a is None:
            raise ValueError("unc_set.a is None")
        if isinstance(self.unc_set, Scenario):
            raise ValueError("Cannot train a scenario set")

    def count_unq_uncertain_param(self, problem) -> int:
        unc_params = self.uncertain_parameters(problem)
        return len(unique_list(unc_params))

    def uncertain_parameters(self, problem):
        """Find uncertain (u) parameters"""
        return [v for v in problem.parameters() if isinstance(v, UncertainParameter)]

    def x_parameters(self, problem):
        """Find context (x) parameters"""
        return [v for v in problem.parameters() if isinstance(v, ContextParameter)]

    def cp_params(self, problem):
        """Find cvxpy (noncontextual) parameters"""
        return [
            v
            for v in problem.parameters()
            if isinstance(v, OrigParameter)
            and not (isinstance(v, ContextParameter) or isinstance(v, UncertainParameter))
        ]

    def rho_mult_param(self, problem):
        """Find the rho multiplier parameter"""
        return [v for v in problem.parameters() if isinstance(v, SizeParameter)]

    def gen_rho_mult_tch(self, rhoparams=[]):
        """Generate the torch of the rho multiplier value, placed in a list"""
        return [
            torch.tensor(rho.value, dtype=s.DTYPE, requires_grad=self.train_flag)
            for rho in rhoparams
        ][0]

    def shape_parameters(self, problem):
        """Get the reshaping parameters a and b"""
        return [v for v in problem.parameters() if isinstance(v, ShapeParameter)]

    def create_cvxpylayer(self, parameters=None, variables=None) -> CvxpyLayer:
        """Create cvxpylayers.
        Default parameter order: rho multiplier, cvxpy parameters, context parameters, a, b.
        Default variable order: the variables of problem_canon"""
        if parameters is None:
            new_parameters = (
                self._rho_mult_parameter
                + self._cp_parameters
                + self._x_parameters
                + self._shape_parameters
            )
        else:
            assert isinstance(parameters, list)
            self._x_tchs_init = False
            self._x_parameters = parameters
            new_parameters = self._rho_mult_parameter + parameters + self._shape_parameters
        if variables is None:
            # TODO: Clean up if works properly
            # variables = self.problem_no_unc.variables()
            variables = self.problem_canon.variables()
        cvxpylayer = CvxpyLayer(
            self.problem_no_unc,
            parameters=new_parameters,
            variables=self.problem_no_unc.variables(),
        )
        self._reduced_variables = variables
        return cvxpylayer

    def x_parameter_shapes(self, x_params):
        """Get the size of all y parameters"""
        return [v.size for v in x_params]

    def initialize_predictor_dims(self):
        """Find the dimensions of the linear model"""
        x_endind = self.x_endind
        x_shapes = self.x_parameter_shapes(self._x_parameters)
        a_shape = self.unc_set._a.shape
        b_shape = self.unc_set._b.shape
        if x_endind:
            in_shape = x_endind
        else:
            in_shape = sum(x_shapes)
        # out_shape = int(a_shape[0] * a_shape[1] + b_shape[0])
        return in_shape,  a_shape[0] * a_shape[1], b_shape[0]

    def create_predictor_tensors(self,x_batch):
        """Create the tensors of a's and b's using the trained linear model"""
        a_shape = self.unc_set._a.shape
        b_shape = self.unc_set._b.shape
        input_tensors = self.create_input_tensors(x_batch)
        a_tch, b_tch, radius = self.settings.predictor.forward(
            input_tensors,a_shape,b_shape,self.train_flag)
        return a_tch, b_tch, radius

    def create_input_tensors(self,x_batch):
        x_endind = self.x_endind
        x_batch = [torch.flatten(x, start_dim=1) for x in x_batch]
        if x_endind:
            input_tensors = torch.hstack(x_batch)[:, :x_endind]
        else:
            input_tensors = torch.hstack(x_batch)
        return input_tensors

    def create_cp_param_tch(self, num):
        """
        This function creates tensors for the cvxpy parameters

        Args:
        num
            batch size

        Returns:

        cp_param_tchs
            The list of torches of the cvxpy parameters,
            repeated with batch_size as the first dimension
        """
        cp_param_tchs = []
        for param in self._cp_parameters:
            param_tch = torch.tensor(param.value, dtype=s.DTYPE, requires_grad=self.train_flag)
            if num == 0:
                cp_param_tchs.append(param_tch)
                continue
            param_tch_dim = param_tch.dim()
            if param_tch_dim == 0:
                shape = num
            elif param_tch_dim == 1:
                shape = (num, 1)
            elif param_tch_dim == 2:
                shape = (num, 1, 1)
            else:
                raise ValueError("Maximum dimension of parameters is 2")
            cp_param_tchs.append(param_tch.repeat(shape))
        return cp_param_tchs

    def _split_dataset(self,
                       test_percentage=DS.test_percentage,
                       validate_percentage = DS.validate_percentage, seed=0):
        """
        This function splits the uncertainty set into train and test sets
            and also creates torch tensors

        Args:
        test_percentage
            Fraction of data to place into the test set.
        seed
            Random number generator seed

        Returns:

        unc_train_set
            Training set for u
        unc_test_set
            Testing set for u
        unc_train_tch
            Training set for u, torch tensor
        unc_test_tch
            Testing set for u, torch tensor
        x_train_tchs
            Training set for x's, torch tensor
        x_test_tchs
            Testing set for x's, torch tensor
        cp_param_tchs
            list of cvxpy param's, torch tensor
        """

        # Split the dataset into train_set and test, and create Tensors
        np.random.seed(seed)
        assert (test_percentage + validate_percentage) < 1
        n_data = self.settings.data.shape[0]
        num_test = max(1, int(n_data * test_percentage))
        num_validate = max(1, int(n_data * validate_percentage))
        num_train = int(n_data - num_test - num_validate)

        # If the user provided explicit indices in settings.indices_dict, validate and use them
        indices_dict = getattr(self.settings, "indices_dict", None)
        if indices_dict is not None:
            # Expected keys optionally: 'train', 'test', 'validate'
            # Convert to numpy arrays of ints when present
            provided = {}
            for key in ("train", "test", "validate"):
                if key in indices_dict and indices_dict[key] is not None:
                    provided[key] = np.asarray(indices_dict[key], dtype=int)

            # If the dict was empty, fall back to random selection
            if not provided:
                indices_dict = None
            else:
                # Validate indices are within bounds
                for key, arr in provided.items():
                    if np.any(arr < 0) or np.any(arr >= n_data):
                        raise ValueError(f"indices_dict['{key}'] contains out-of-bounds indices")

                # Check for overlap among provided indices
                all_provided = np.concatenate(
                    [provided[k] for k in provided]) if provided \
                        else np.array([], dtype=int)
                if all_provided.size != 0 and np.unique(all_provided).size != all_provided.size:
                    raise ValueError("Overlapping indices found in settings.indices_dict")

                # Assign provided indices (some may be missing)
                test_indices = provided.get("test", None)
                validate_indices = provided.get("validate", None)
                train_indices = provided.get("train", None)

                # Infer missing splits deterministically from remaining indices
                used = np.array([], dtype=int)
                if test_indices is not None:
                    used = np.concatenate([used, np.asarray(test_indices, dtype=int)])
                if validate_indices is not None:
                    used = np.concatenate([used, np.asarray(validate_indices, dtype=int)])
                if train_indices is not None:
                    used = np.concatenate([used, np.asarray(train_indices, dtype=int)])
                used = np.unique(used) if used.size else np.array([], dtype=int)

                remaining = np.array([i for i in range(n_data) if i not in used], dtype=int)
                # If train missing, put remaining into train
                if train_indices is None:
                    train_indices = remaining
                    remaining = np.array([], dtype=int)
                # If test missing, put remaining into test
                if test_indices is None and remaining.size:
                    test_indices = remaining
                    remaining = np.array([], dtype=int)
                # If validate missing, put any remaining into validate
                if validate_indices is None and remaining.size:
                    validate_indices = remaining
                    remaining = np.array([], dtype=int)

                # Ensure all indices are numpy arrays
                test_indices = np.asarray(
                    test_indices, dtype=int) if test_indices \
                        is not None else np.array([], dtype=int)
                validate_indices = np.asarray(
                    validate_indices, dtype=int) if \
                        validate_indices is not None else np.array(
                            [], dtype=int)
                train_indices = np.asarray(
                    train_indices, dtype=int) if \
                        train_indices is not None else np.array([], dtype=int)

                # Update counts to reflect provided/derived indices
                num_test = int(test_indices.size)
                num_validate = int(validate_indices.size)
                num_train = int(train_indices.size)

        if indices_dict is None:
            # No indices_dict provided; do random selection
            test_and_validate_indices = np.random.choice(
                n_data, num_test + num_validate, replace=False)
            test_indices = test_and_validate_indices[:num_test]
            validate_indices = test_and_validate_indices[num_test:]
            train_indices = np.array([i for i in range(
                n_data) if i not in test_and_validate_indices], dtype=int)

        unc_train_set = np.array([self.settings.data[i] for i in train_indices])
        unc_validate_set = np.array(
            [self.settings.data[i] for i in validate_indices])
        unc_test_set = np.array([self.settings.data[i] for i in test_indices])
        unc_train_tch = torch.tensor(
            self.settings.data[train_indices], requires_grad=self.train_flag, dtype=s.DTYPE
        )
        unc_test_tch = torch.tensor(
            self.settings.data[test_indices], requires_grad=self.train_flag, dtype=s.DTYPE
        )
        unc_validate_tch = torch.tensor(
            self.settings.data[validate_indices], requires_grad=self.train_flag, dtype=s.DTYPE
        )

        cp_param_tchs = []
        x_train_tchs = []
        x_test_tchs = []
        x_validate_tchs = []

        if self._x_tchs_init:
            cp_param_tchs = self.create_cp_param_tch(0)
            for i in range(len(self._x_parameters)):
                x_train_tchs.append(
                    torch.tensor(
                        self._x_parameters[i].data[train_indices],
                        requires_grad=self.train_flag,
                        dtype=s.DTYPE,
                    )
                )
                x_test_tchs.append(
                    torch.tensor(
                        self._x_parameters[i].data[test_indices],
                        requires_grad=self.train_flag,
                        dtype=s.DTYPE,
                    )
                )
                x_validate_tchs.append(
                    torch.tensor(
                        self._x_parameters[i].data[validate_indices],
                        requires_grad=self.train_flag,
                        dtype=s.DTYPE,
                    )
                )

        self.u_train_tch = unc_train_tch
        self.u_test_tch = unc_test_tch
        self.u_validate_tch = unc_validate_tch
        self.u_train_set = unc_train_set
        self.u_test_set = unc_test_set
        self.u_validate_set = unc_validate_set
        self.x_train_tch = x_train_tchs
        self.x_test_tch = x_test_tchs
        self.x_validate_tch = x_validate_tchs
        self.train_size = num_train
        self.test_size = num_test
        self.validate_size = num_validate
        self.cp_param_tch = cp_param_tchs
        if self._multistage and isinstance(\
            self.settings.predictor,CovPredictor):
            if (self._init_uncertain_parameter is None) or (self._init_context is None):
                raise ValueError(
                    "You must provide init_uncertain_param and \
                                  init_context in the trainer settings\
                                    when using the covariance predictor"
                )
            self.u_train_tch = torch.tensor(
                self._init_uncertain_parameter, requires_grad=True, dtype=s.DTYPE
            )
            self.x_train_tch = self._init_context
        elif self._multistage and isinstance(\
            self.settings.predictor,LinearPredictor):
            if self.settings.predictor.predict:
                if (self._init_uncertain_parameter is None):
                    raise ValueError(
                    "You must provide init_uncertain_param when using the mean linear predictor"
                )
                self.u_train_tch = torch.tensor(
                self._init_uncertain_parameter, requires_grad=True, dtype=s.DTYPE
            )
                self.x_train_tch = self.settings.simulator.init_state(
                    self._init_uncertain_parameter.shape[0],
                      self.settings.seed,
                                            **self.settings.kwargs_simulator)

        # return (
        #     unc_train_set,
        #     unc_test_set,
        #     unc_train_tch,
        #     unc_test_tch,
        #     x_train_tchs,
        #     x_test_tchs,
        #     cp_param_tchs,
        # )

    def _gen_batch(self, num_xs, x_data, u_data,
                   batch_percentage, max_size=10000, min_size=1,seed=0):
        """
        This function generates a list of torches for each x and u
        for all context parameters x and uncertain parameter u

        Args:
        num_xs
            Total number of samples of x. Default: size of training set.
        x_data
            context datasets.
        u_data
            uncertainty dataset.
        batch_percentage
            percentage of total number of samples to place in the batch
        max_size
            maximum number of samples in a batch
        min_size
            minimum number of samples in a batch
        """
        np.random.seed(seed)
        batch_int = max(min(int(num_xs * batch_percentage), max_size), min_size)
        random_int = np.random.choice(num_xs, batch_int, replace=False)
        x_tchs = []
        for i in range(len(x_data)):
            x_tchs.append(x_data[i].data[random_int])

        u_tch = torch.tensor(u_data[random_int], requires_grad=self.train_flag, dtype=s.DTYPE)

        return batch_int, x_tchs, u_tch

    def _gen_rho_tch(self, init_rho):
        """
        This function generates rho_tch

        Args:

        init_rho
            Initial rho (radius of uncertainty set)
        unc_set
            Uncertainty set
        mro_set
            Boolean flag set to True for MRO problem
        """
        scalar = init_rho if init_rho else 1.0
        rho_tch = torch.tensor(scalar, requires_grad=self.train_flag, dtype=s.DTYPE)

        return rho_tch

    # TODO(bart): why calling this name? Do we ever use it not for init_A?
    def _gen_init(self, train_set, init_A):
        """
        This is an internal function that calculates init.
        Init means different things depending on rho
            it is an internal function not intended to be used publicly.

        Args:

        train_shape
            Boolean flag indicating if we train a/b or not
        train_set
            The training set
        init_A
            The given initiation for the reshaping matrix A.
            If none is passed, it will be initiated as the covariance matrix of the provided data.

        Returns:

        init
            np.array (NOT TENSOR)
        """

        cov_len_cond = train_set.size > 0 and len(np.shape(np.cov(train_set.T))) >= 1
        if init_A is None:
            if cov_len_cond:
                return sc.linalg.sqrtm(np.cov(train_set.T))
            return np.array([[np.cov(train_set.T)]])

        mat_shape = train_set.shape[1] if cov_len_cond else 1
        matrix = np.array(init_A) if (init_A is not None) else np.eye(mat_shape)
        return matrix

    def _init_torches(self, init_A, init_b, init_alpha, train_set):
        """
        This function Initializes and returns a_tch, b_tch, and alpha as tensors.
        It also initializes alpha as 0
        """
        # train_set = train_set.detach().numpy()
        self._init = self._gen_init(train_set, init_A)
        init_tensor = torch.tensor(self._init, requires_grad=self.train_flag, dtype=s.DTYPE)
        b_tch = None

        if init_b is not None:
            b_tch_data = np.array(init_b)
        else:
            b_tch_data = np.mean(train_set, axis=0)
        b_tch = torch.tensor(b_tch_data, requires_grad=self.train_flag, dtype=s.DTYPE)
        a_tch = init_tensor

        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)
        return a_tch, b_tch, alpha

    def _update_iters(self, save_history, a_history, b_history, rho_history, a_tch, b_tch, rho_tch):
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
        rho = rho_tch.detach().numpy().copy()
        rho_history.append(rho)
        a_history.append(a_tch.detach().numpy().copy())
        b_history.append(b_tch.detach().numpy().copy())

    def _set_train_variables(
        self, fixb, alpha, a_tch, b_tch, rho_tch, train_size, contextual, model
    ):
        """
        This function sets the variables to be trained in the outer level problem.

        Args:

        fixb
            Whether to hold b constant or not when training
        alpha
            Torch tensor of alpha for CVaR
        a_tch
            Torch tensor of A
        b_tch
            Torch tensor of b
        rho_tch
            Torch tensor of rho
        contexual
            Whether or not the set is contextual
        model
            The contextual model

        Returns:

        The list of variables to train using pytorch
        """
        if train_size:
            variables = [rho_tch, alpha]
        elif fixb:
            variables = [rho_tch, a_tch, alpha]
        elif contextual:
            variables = [alpha]
            variables.extend(list(model.parameters()))
        else:
            variables = [rho_tch, a_tch, b_tch, alpha]

        return variables

    def _calc_coverage(
        self, dset, a_tch, b_tch, rho=1, p=2, contextual=False, y_set=None
    ):
        """
        This function calculates coverage.

        Args:
            dset:
                Dataset (train or test)
            a_tch:
                A torch
            b_tch:
                b torch
            rho:
                The radius of the set
            p:
                The order of the norm
            contextual:
                Whether or not the set is contextual

        Returns:
            Coverage
        """
        if self._multistage:
            return 0
        coverage = 0
        if contextual:
            a_tch, b_tch, radius = self.create_predictor_tensors(
                [torch.tensor(y) for y in y_set[-1]]
            )
            for i in range(dset.shape[0]):
                coverage += torch.where(
                    torch.norm(
                        (a_tch[i].T @ torch.linalg.inv(a_tch[i] @ a_tch[i].T))
                        @ (dset[i] - b_tch[i]),
                        p,
                    )
                    <= rho,
                    1,
                    0,
                )
        else:
            for datind in range(dset.shape[0]):
                coverage += torch.where(
                    torch.norm(
                        (a_tch.T @ torch.linalg.inv(a_tch @ a_tch.T)) @ (dset[datind] - b_tch), p
                    )
                    <= rho,
                    1,
                    0,
                )
        return (coverage / dset.shape[0]).detach().numpy().item()

    def order_args(self, z_batch, x_batch, u_batch):
        """
        This function orders z_batch (decisions), x_batch (context), and
        u_batch (uncertainty) according to the order in vars_params.
        """
        problem = self.problem_canon
        return problem.order_args(z_batch, x_batch, u_batch)

    def train_objective(self, batch_int, eval_args):
        """
        This function evaluates the expectation of the objective function over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the objective function
        Returns:
            The average among all evaluated J x N pairs
        """
        return eval_input(
            batch_int,
            eval_func=self.f,
            eval_args=eval_args,
            init_val=0,
            eval_input_case=EVAL_INPUT_CASE.MEAN,
            quantiles=None,
            eta_target= None
        )

    def train_constraint(self, batch_int, eval_args, alpha, eta, kappa):
        """
        This function evaluates the expectation of the CVaR
        constraint functions over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the constraint functions
            alpha:
                The alpha of the CVaR constraint
            kappa:
                The target CVaR threshold
        Returns:
            The average among all evaluated J x N pairs
        """
        H = torch.zeros(self.num_g_total, dtype=s.DTYPE)
        for k, h_k in enumerate(self.h):
            init_val = eval_input(
                batch_int,
                h_k,
                eval_args,
                0,
                EVAL_INPUT_CASE.MEAN,
                quantiles=None,
                alpha=alpha,
                eta=eta,
                eta_target = None
            )
            h_k_expectation = (
                init_val
                + alpha
                - kappa            )
            H[sum(self.g_shapes[:k]) : sum(self.g_shapes[: (k + 1)])] = h_k_expectation
        return H

    def evaluation_metric(self, batch_int, eval_args, quantiles):
        """
        This function evaluates the evaluation metric over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the evaluation function
            quantiles:
                The upper and lowerx quantiles to be returned.
        Returns:
            The average among all evaluated J x N pairs
        """
        if self.eval is None:
            return 0

        return eval_input(
            batch_int,
            eval_func=self.eval,
            eval_args=eval_args,
            init_val=0,
            eval_input_case=EVAL_INPUT_CASE.EVALMEAN,
            quantiles=quantiles,
            serial_flag=True,
            eta_target= None
        )

    def evaluation_cvar(self, batch_int, eval_args, eta = None):
        """
        This function evaluates the evaluation CVaR over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the evaluation function
            eta:
                The eta value for the CVaR.
        Returns:
            The CVaR of the evaluation function
        """
        if self.eval is None:
            return 0
        if eta is None:
            eta = self.settings.target_eta
        return eval_input(
            batch_int,
            eval_func=self.eval,
            eval_args=eval_args,
            init_val=0,
            eval_input_case=EVAL_INPUT_CASE.CVAR,
            serial_flag=True,
            quantiles = None,
            eta_target = eta
        )

    def dist_loss(self,cho,mean,radius, x, y):
        convergence_threshold=1e-4
        max_epochs=500
        lr=1e-2
        cov_inv = torch.matmul(torch.linalg.inv(torch.transpose(cho, 1, 2)), torch.linalg.inv(cho))


        def compute_distance(y_train, mean, cov_inv):
            def dist(u, v, c):
                diff = u - v
                m = torch.matmul(torch.matmul(diff.T, c), diff)
                return torch.sqrt(m)
            dist_list = torch.empty(len(y_train))
            for i in range(len(y_train)):
                x = y_train[i]
                m = mean[i]
                c = cov_inv[i]  #*(radius**2)
                dis = dist(x,m,c)
                # dis_div_ratio = dis/ r

                # temp = (1/(radius*radius))*torch.matmul(c[i].T, x - m)
                # dis = torch.linalg.norm(temp,  ord=2)
                dist_list[i] = dis   #dis_div_ratio

            return dist_list

        mahalanobis_dist = compute_distance(y, mean, cov_inv)
        assign_list = (mahalanobis_dist <= radius).float().unsqueeze(-1)

        log_regr = LogisticRegression(x.shape[1])
        criterion = nn.BCELoss()
        optimizer_conf = torch.optim.Adam(log_regr.parameters(), lr=lr)

        prev_loss = float('inf')
        for epoch in range(int(max_epochs)):
            optimizer_conf.zero_grad()
            predictions = log_regr(x)
            loss_conf = criterion(predictions.float(), assign_list.float())
            loss_conf.backward(create_graph=True)
            optimizer_conf.step()

            if abs(prev_loss - loss_conf.item()) < convergence_threshold:
                break
            prev_loss = loss_conf.item()

        predicted_prob = log_regr(x)


        conditional_loss = torch.mean((predicted_prob - (1 - 0.9)) ** 2)

        penalty = nn.LeakyReLU()(0.9 - assign_list.mean())
        loss_final = conditional_loss + 10 * penalty

        return loss_final, assign_list.mean().item()

    def prob_constr_violation(self, batch_int, eval_args):
        """
        This function evaluates the probability of constraint violation
        of all uncertain constraints over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the constraints
        Returns:
            The average among all evaluated J x N pairs
        """
        return eval_prob_constr_violation(
            g=self.g, g_shapes=self.g_shapes, batch_int=batch_int, eval_args=eval_args
        )

    def lagrangian(
        self,
        batch_int,
        eval_args,
        alpha,
        lam,
        mu,
        eta=DS.eta,
        kappa=DS.kappa,
    ):
        """
        The function evaluates generates the augmented lagrangian of
        the problem, over the batched set.
        Args:
            batch_int:x
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the problems (variables, parameters)
            alpha:
                The alpha of the CVaR constraint
            lam:
                The lagrangian multiplier
            mu:
                The penalty multiplier
            eta:
                The eta value for the CVaR constraint
            kappa:
                The target CVaR threshold

        Returns:
            The average among all evaluated J x N pairs
        """
        F = self.train_objective(batch_int, eval_args=eval_args)
        H = self.train_constraint(
            batch_int, eval_args=eval_args, alpha=alpha,eta=eta, kappa=kappa
        )
        return F + lam @ H + (mu / 2) * (torch.linalg.norm(H) ** 2), H.detach()

    def _reduce_variables(self, z_batch: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        This helper function reduces z_batch whose len is len(self.problem_no_unc.variables())
        to len(_reduced_variables) (default: len(self.problem_canon.variables())).
        It returns the reduced list of tensors, where only the tensors that correspond to variables
        of self.problem_canon.variables() are preserved.
        """
        res = [None] * len(self._reduced_variables)
        for problem_no_unc_ind, problem_no_unc_var in enumerate(self.problem_no_unc.variables()):
            for problem_canon_ind, problem_canon_var in enumerate(self._reduced_variables):
                if problem_no_unc_var.id == problem_canon_var.id:
                    res[problem_canon_ind] = z_batch[problem_no_unc_ind]
                    break
        return res

    def _line_search(self, step_num: int, a_tch: torch.Tensor, b_tch: torch.Tensor,
                     rho_tch: torch.Tensor, alpha: torch.Tensor,
                     lam: torch.Tensor, mu: float,
                     opt: torch.optim.Optimizer, prev_states: list,
                     seed_num: int,prev_fin_cost: torch.Tensor) \
                        -> tuple[CONSTRAINT_STATUS, torch.Tensor, torch.Tensor, torch.Tensor,
                                 torch.Tensor]:
        """
        If line_search is True (add new settings param), reduce the step size
        by line_search_mult (add a new settings param) until the new lagrangian
        value is at least as low as the previous lagrangian value, or until
        max_iter_line search in which we proceed with the last step size
        (don't fail, continue on, should not return a time_out error unlike
        if infeasibility is encountered)
        Args:
            step_num (int):

            a_tch (torch.Tensor):

            b_tch (torch.Tensor):

            rho_tch (torch.Tensor):

            alpha (torch.Tensor):

            lam (torch.Tensor):

            mu (float):

            opt (torch.optim.Optimizer):

            prev_states (list):

        Returns:

            constraint_status (CONSTRAINT_STATUS):
                Are all the constraints satisfied?

            fin_cost

            avg_cost (torch.Tensor):

            prob_violation_train (torch.Tensor):

            constr_cost (torch.Tensor):
        """
        # In the first epoch we try only once
        current_iter_line_search = 1 if step_num == 0 else self._max_iter_line_search + 1
        for violation_counter in range(current_iter_line_search):
            self._validate_flag = False
            self._test_flag = False
            cost, constr_cost, _, _, constraint_status, \
                avg_cost, prob_violation_train, _ ,cvar_cost, \
                    in_sample_cost, worst_cost= (
                self.loss_and_constraints(
                    a_tch=a_tch,
                    b_tch=b_tch,
                    seed=self.settings.seed + seed_num + 1,
                    rho_tch=rho_tch,
                    alpha=alpha,
                    batch_size = self.settings.batch_size
                )
            )

            if not self._default_simulator:
                avg_cost = avg_cost.repeat(3)

            if self.settings.constraint_cvar:
                if self.num_g_total > 1:
                    fin_cost = (
                        cost + lam @ torch.maximum(
                            constr_cost,torch.zeros(self.num_g_total)) + (
                                mu / 2) * (torch.linalg.norm(
                                    torch.maximum(constr_cost,
                                                    torch.zeros(self.num_g_total))) ** 2)
                    )
                else:
                    fin_cost = cost + lam * torch.maximum(
                        constr_cost,torch.zeros(1)) + (
                            mu / 2) * (torch.maximum(
                                constr_cost,torch.zeros(1))**2)
            elif self.settings.delage_coverage:
                fin_cost = (1-self.settings.coverage_gamma)*cost + \
                    self.settings.coverage_gamma*constr_cost
            else:
                fin_cost = cost

            if self.settings.line_search:
                search_condition = fin_cost <= self.settings.line_search_threshold*prev_fin_cost
            else:
                search_condition = True
            if constraint_status is CONSTRAINT_STATUS.FEASIBLE and search_condition:
                restore_step_size(opt, num_steps=violation_counter,
                                  step_mult = self.settings.line_search_mult)
                opt.zero_grad()
                fin_cost.backward()
                prev_fin_cost = fin_cost.clone().detach()
                break
            elif constraint_status is CONSTRAINT_STATUS.INFEASIBLE and (step_num != 0):
                undo_step(opt=opt,state=prev_states)
                reduce_step_size(opt=opt,step_mult = self.settings.line_search_mult)
                prev_states = take_step(opt=opt,
                                        rho_tch=rho_tch, scheduler=None,
                                        update_state = False,
                                        prev_states = prev_states)
            elif constraint_status is CONSTRAINT_STATUS.FEASIBLE \
                and violation_counter == current_iter_line_search -1:
                restore_step_size(opt, num_steps=violation_counter)
                opt.zero_grad()
                fin_cost.backward()
                prev_fin_cost = fin_cost.clone().detach()
                break
        return constraint_status, fin_cost, avg_cost, \
            prob_violation_train, constr_cost, cvar_cost, in_sample_cost, worst_cost

    def _train_loop(self, init_num):
        if self.settings.random_init and self.settings.train_shape:
            if init_num >= 1:
                np.random.seed(self.settings.seed + init_num)
                shape = self.unc_set._a.shape
                self.settings.init_A = np.random.rand(shape[0], shape[1])
                self.settings.init_b = np.mean(self.u_train_set, axis=0)
        a_history = []
        b_history = []
        rho_history = []
        rows = []
        rows_test = []
        rows_validate = []

        rho_tch = self._gen_rho_tch(self.settings.init_rho)
        a_tch, b_tch, alpha = self._init_torches(
            self.settings.init_A, self.settings.init_b, self.settings.init_alpha, self.u_train_set
        )

        self._update_iters(
            self.settings.save_history, a_history, b_history, rho_history, a_tch, b_tch, rho_tch
        )

        if self.settings.contextual:
            if self.settings.initialize_predictor:
                self.settings.predictor.initialize(a_tch,b_tch,self)


        variables = self._set_train_variables(
            self.settings.fixb,
            alpha,
            a_tch,
            b_tch,
            rho_tch,
            self.settings.trained_shape,
            self.settings.contextual,
            self.settings.predictor,
        )
        if self.settings.optimizer == "SGD":
            opt = s.OPTIMIZERS[self.settings.optimizer](
                variables, lr=self.settings.lr, momentum=self.settings.momentum
            )
        else:
            opt = s.OPTIMIZERS[self.settings.optimizer](variables,
                                                         lr=self.settings.lr,
                                                         weight_decay = 1e-1)

        if self.settings.scheduler:
            scheduler_ = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.settings.lr_step_size, gamma=self.settings.lr_gamma
            )
        else:
            scheduler_ = None
        # y's and cvxpylayer begin
        lam = self.settings.init_lam * torch.ones(self.num_g_total, dtype=s.DTYPE)
        mu = self.settings.init_mu
        seed_num = 0
        curr_cvar = np.inf
        prev_fin_cost = np.inf
        if self._default_simulator:
            self.settings.kwargs_simulator = {
                "trainer": self,
            }
        prev_states = []
        for step_num in range(self.settings.num_iter):
            if step_num > 0:
                prev_states = take_step(opt=opt, rho_tch=rho_tch, scheduler=scheduler_)
            train_stats = TrainLoopStats(
                step_num=step_num, train_flag=self.train_flag, num_g_total=self.num_g_total
            )

            torch.manual_seed(self.settings.seed + step_num)
            constraint_status, fin_cost, avg_cost, \
                prob_violation_train, constr_cost, cvar_cost, in_sample_cost,worst_cost = \
                self._line_search(step_num=step_num, a_tch=a_tch, b_tch=b_tch, rho_tch=rho_tch,
                                  alpha=alpha, lam=lam, mu=mu, opt=opt,
                                  prev_states=prev_states,
                                  seed_num = seed_num,
                                  prev_fin_cost = prev_fin_cost)
            if constraint_status is CONSTRAINT_STATUS.INFEASIBLE:
                if step_num == 0:
                    exception_message = "Infeasible uncertainty set initialization"
                else:
                    exception_message = "Violation constraint check timed " +\
                     "out after " + f"{self.settings.max_iter_line_search} attempts."
                print(exception_message)
                # raise InfeasibleConstraintException(exception_message)
            train_stats.update_train_stats(
                worst_cost.detach().numpy().copy(),
                in_sample_cost.detach().numpy().copy(),
                cvar_cost.detach().numpy().copy(),
                fin_cost.detach().numpy().copy(),
                avg_cost,
                prob_violation_train,
                constr_cost.detach(),
            )

            if step_num % self.settings.aug_lag_update_interval == 0:
                seed_num += 1
                prev_fin_cost = np.inf
                if (
                    torch.norm(constr_cost.detach())
                    <= self.settings.lambda_update_threshold * curr_cvar
                ):
                    curr_cvar = torch.norm(constr_cost.detach())
                    lam += torch.minimum(
                        mu * constr_cost.detach(),
                        self.settings.lambda_update_max
                        * torch.ones(self.num_g_total, dtype=s.DTYPE),
                    )
                else:
                    mu = self.settings.mu_multiplier * mu

            new_row = train_stats.generate_train_row(
                self._a_tch,
                rho_tch,
                lam,
                mu,
                alpha,
                self.settings.contextual,
                self.settings.linear,
                self.settings.predictor            )
            rows.append(new_row)

            self._update_iters(
                self.settings.save_history,
                a_history,
                b_history,
                rho_history,
                self._a_tch,
                self._b_tch,
                rho_tch,
            )

            if step_num % self.settings.validate_frequency == 0:
                self._validate_flag = True
                self._test_flag = False
                (
                    val_cost,
                    val_cost_constr,
                    x_batch,
                    z_batch,
                    avg_cost,
                    prob_constr_violation,
                    u_batch,cvar_cost,
                    in_sample_cost, worst_cost
                ) = self.monte_carlo(
                    a_tch=a_tch,
                    b_tch=b_tch,
                    rho_tch=rho_tch,
                    alpha=alpha,
                    seed = self.settings.seed + 10000
                )
                record_avg_cost = avg_cost
                if not self._default_simulator:
                    record_avg_cost = avg_cost.repeat(3)

                # Update progress bar with current metrics
                if not hasattr(self, "_pbar"):
                    # Import tqdm here to avoid dependency issues if not available
                    try:
                        from tqdm.auto import tqdm as auto_tqdm

                        # Create progress bar on first evaluation
                        self._pbar = auto_tqdm(
                            total=self.settings.num_iter, desc="Training", leave=True
                        )
                        # Initialize with current step
                        self._pbar.update(step_num)
                    except ImportError:
                        # Fallback to regular print if tqdm not available
                        print(
                            "it %d, loss %1.2e, viol %1.2e"
                            % (step_num, worst_cost.item(), val_cost_constr.mean().item())
                        )
                else:
                    # Update progress bar position
                    self._pbar.update(self.settings.validate_frequency)

                    # Set description with metrics
                    self._pbar.set_postfix(
                        {
                            "loss": f"{worst_cost.item():1.2e}",
                            "viol": f"{val_cost_constr.mean().item():1.2e}",
                        }
                    )

                train_stats.update_validate_stats(worst_cost,in_sample_cost,
                                                  cvar_cost,
                    record_avg_cost, prob_constr_violation, val_cost_constr.detach()
                )
                new_row = train_stats.generate_validation_row(
                    self._calc_coverage,
                    self._a_tch,
                    self._b_tch,
                    alpha,
                    u_batch,
                    rho_tch,
                    self.unc_set,
                    z_batch,
                    self.settings.contextual,
                    x_batch
                )
                rows_validate.append(new_row)

            if step_num % self.settings.test_frequency == 0:
                self._test_flag = True
                self._validate_flag= False
                (
                    val_cost,
                    val_cost_constr,
                    x_batch,
                    z_batch,
                    avg_cost,
                    prob_constr_violation,
                    u_batch,
                    cvar_cost,
                    in_sample_cost, worst_cost
                ) = self.monte_carlo(
                    a_tch=a_tch,
                    b_tch=b_tch,
                    rho_tch=rho_tch,
                    alpha=alpha,
                    seed = self.settings.seed
                )
                record_avg_cost = avg_cost
                if not self._default_simulator:
                    record_avg_cost = avg_cost.repeat(3)

                train_stats.update_test_stats(worst_cost,
                                              in_sample_cost,cvar_cost,
                    record_avg_cost, prob_constr_violation, val_cost_constr.detach()
                )
                new_row = train_stats.generate_test_row(
                    self._calc_coverage,
                    self._a_tch,
                    self._b_tch,
                    alpha,
                    u_batch,
                    rho_tch,
                    self.unc_set,
                    z_batch,
                    self.settings.contextual,
                    x_batch
                )
                rows_test.append(new_row)

        df = pd.DataFrame([r.to_dict() for r in rows]) if rows else pd.DataFrame()
        df_test = (
            pd.DataFrame([r.to_dict() for r in rows_test]) if rows_test else pd.DataFrame()
        )
        df_validate = (
            pd.DataFrame([r.to_dict() for r in rows_validate])
            if rows_validate
            else pd.DataFrame()
        )

        if constr_cost.detach().numpy().sum() / self.num_g_total <= self.settings.kappa:
            fin_val = record_avg_cost[1].item()
        else:
            fin_val = record_avg_cost[1].item() + 10 * abs(constr_cost.detach().numpy().sum())
        a_val = self._a_tch.detach().numpy().copy()
        b_val = self._b_tch.detach().numpy().copy()
        rho_val = rho_tch.detach().numpy().copy()
        param_vals = (a_val, b_val, rho_val, record_avg_cost[1].item())
        ret_context = (self._cur_x, self._cur_u)
        # Close progress bar if it exists
        if hasattr(self, "_pbar"):
            self._pbar.close()
            delattr(self, "_pbar")

        return (
            df,
            df_test,
            df_validate,
            a_history,
            b_history,
            rho_history,
            param_vals,
            fin_val,
            z_batch,
            mu,
            self.settings.predictor,
            ret_context,
        )

    def train(
        self,
        settings: s.TrainerSettings | None = s.TrainerSettings(),
    ):
        r"""
        Trains the uncertainty set parameters to find optimal set
        w.r.t. augmented lagrangian metric

        Args:
        -----------

        Returns:
        A Result object, including a pandas data frame with the following columns:
            Test_val: float
                The out of sample objective value of the Robust Problem
            Probability_violations_test
                Probability of constraint violation over test set
            Avg_prob_test
                Probability of constraint violation averaged over all constraints
            Violations_test:
                Violation of learning constraint over test set
            Rho: float
                The :math:`\rho` value
            Coverage_test: float
                The percentage of testing data covered by the uncertainty set
            var_values: list
                A list of returned variable values from the last solve
        """
        self.settings = settings
        self.train_flag = True
        if self.settings.contextual and not self.settings.train_shape:
            if self.settings.predictor is None:
                raise ValueError("You must give a model if you do not train a model")
        if self.settings.predictor is None:
            self.settings.predictor = LinearPredictor()
        self.settings.linear = isinstance(self.settings.predictor,LinearPredictor)
        self._multistage = self.settings.multistage
        self._init_uncertain_parameter = self.settings.init_uncertain_param
        self._init_context = self.settings.init_context
        if self.settings.data is None:
            self.settings.data = self.unc_set.data
        if self.settings.indices_dict is None:
            self.settings.indices_dict = self.unc_set.indices_dict
        self._split_dataset(self.settings.test_percentage,
                            self.settings.validate_percentage, self.settings.seed)

        if self._multistage:
            self.num_g_total = 1
        else:
            assert self.settings.time_horizon == 1
        if not self.settings.policy:
            self.cvxpylayer = self.create_cvxpylayer()
        else:
            self.cvxpylayer = self.settings.policy
        if self.settings.simulator:
            self.simulator = self.settings.simulator
            self._default_simulator = False
        else:
            self.simulator = DefaultSimulator(self)
            self._default_simulator = True
        if self.settings.num_iter == 0:
            self.settings.num_iter = 1
        self.violation_checker = ViolationChecker(self.cvxpylayer, self.problem_no_unc.constraints)
        self._max_iter_line_search = self.settings.max_iter_line_search
        self.x_endind = self.settings.x_endind

        if self.settings.random_init:
            self.settings.num_random_init = self.settings.num_random_init
        else:
            self.settings.num_random_init = 1
        if self.settings.train_shape:
            self.settings.num_random_init = self.settings.num_random_init
        else:
            self.settings.num_random_init = 1

        # Debugging code - one iteration
        # res = self._train_loop(0)
        # Debugging code - serial
        if not self.settings.parallel:
            res = []
            for init_num in range(self.settings.num_random_init):
                res.append(self._train_loop(init_num))
                gc.collect()
        # Joblib version
        else:
            self.settings.n_jobs = get_n_processes() if self.settings.parallel else 1
            res = Parallel(n_jobs=self.settings.n_jobs, backend="loky")(
                delayed(self._train_loop)(init_num)
                for init_num in range(self.settings.num_random_init)
            )
        (
            df,
            df_test,
            df_validate,
            a_history,
            b_history,
            rho_history,
            param_vals,
            fin_val,
            var_values,
            mu_val,
            predictors,
            ret_context,
        ) = zip(*res)
        index_chosen = np.argmin(np.array(fin_val))
        self.orig_problem_trained = True
        self.unc_set._trained = True
        return_rho = param_vals[index_chosen][2]
        self._rho_mult_parameter[0].value = return_rho
        self._cur_x = ret_context[index_chosen][0]
        self._cur_u = ret_context[index_chosen][1]
        if self.settings.contextual:
            self.unc_set.a.value = param_vals[index_chosen][0][0]
            self.unc_set.b.value = param_vals[index_chosen][1][0]
        else:
            self.unc_set.a.value = param_vals[index_chosen][0]
            self.unc_set.b.value = param_vals[index_chosen][1]

        if self.settings.train_shape and self.settings.train_size:
            self.settings.init_rho = return_rho
            self.settings.trained_shape = True
            self.settings.train_shape = False
            self.settings.init_A = self.unc_set.a.value
            self.settings.init_b = self.unc_set.b.value
            self.settings.random_init = False
            if self.settings.lr_size:
                self.settings.lr = self.settings.lr_size
            else:
                self.settings.lr = self.settings.lr
            if self.settings.num_iter_size:
                self.settings.num_iter = self.settings.num_iter_size
            self.settings.init_mu = mu_val[index_chosen]
            if not self.settings.parallel:
                res = []
                for init_num in range(1):
                    res.append(self._train_loop(init_num))
            else:
                res = Parallel(n_jobs=self.settings.n_jobs)(
                    delayed(self._train_loop)(init_num) for init_num in range(1)
                )
            (
                df_s,
                df_test_s,
                df_validate_s,
                a_history_s,
                b_history_s,
                rho_history_s,
                param_vals_s,
                fin_val_s,
                var_values_s,
                mu_s,
                predictors_s,
                ret_context_s,
            ) = zip(*res)
            return_rho = param_vals_s[0][2]
            self._rho_mult_parameter[0].value = return_rho
            self._cur_x = ret_context[0][0]
            self._cur_u = ret_context[0][1]
            return_df = pd.concat([df[index_chosen], df_s[0]])
            return_df_test = pd.concat([df_test[index_chosen], df_test_s[0]])
            return_df_validate = pd.concat(
                [df_validate[index_chosen], df_validate_s[0]])
            return_a_history = a_history[index_chosen] + a_history_s[0]
            return_b_history = b_history[index_chosen] + b_history_s[0]
            return_rho_history = rho_history[index_chosen] + rho_history_s[0]
            return Result(
                self,
                self.problem_canon,
                return_df,
                return_df_test,
                return_df_validate,
                self.unc_set.a.value,
                self.unc_set.b.value,
                return_rho,
                param_vals[0][3],
                var_values[index_chosen] + var_values_s[0],
                a_history=return_a_history,
                b_history=return_b_history,
                rho_history=return_rho_history,
                predictor=predictors_s[0],
            )
        return Result(
            self,
            self.problem_canon,
            df[index_chosen],
            df_test[index_chosen],
            df_validate[index_chosen],
            self.unc_set.a.value,
            self.unc_set.b.value,
            return_rho,
            param_vals[index_chosen][3],
            var_values[index_chosen],
            a_history=a_history[index_chosen],
            b_history=b_history[index_chosen],
            rho_history=rho_history[index_chosen],
            predictor=predictors[index_chosen],
        )

    def compare_predictors(
            self,
            settings: s.TrainerSettings | None = s.TrainerSettings(),
            predictors_list = [],rho_list = []):
        """This function computes the validation and testing values for a
        list of predictors"""
        settings.num_iter = 0
        settings.max_batch_size = np.inf
        test_dfs = []
        validate_dfs = []
        num_rho = len(rho_list)
        if len(predictors_list)==1:
            if num_rho > 1:
                predictors_list = predictors_list*num_rho
        for ind, predictor in enumerate(predictors_list):
            if predictor is None:
                settings.contextual = False
            settings.predictor = predictor
            settings.initialize_predictor = False
            settings.init_rho = rho_list[ind]
            result = self.train(settings=settings)
            test_dfs.append(result.df_test)
            validate_dfs.append(result.df_validate)
        return pd.concat(validate_dfs), pd.concat(test_dfs)



    def gen_unique_x(self, x_batch):
        """get unique x's from a list of x parameters."""
        x_batch_array = [np.array(ele) for ele in x_batch]
        all_indices = [np.unique(ele, axis=0, return_index=True)[1] for ele in x_batch_array]
        unique_indices = np.unique(np.concatenate(all_indices))
        num_unique_indices = len(unique_indices)
        x_unique = [torch.tensor(ele, dtype=s.DTYPE)[unique_indices] for ele in x_batch_array]
        _unique_array = [ele[unique_indices] for ele in x_batch_array]
        return x_batch_array, num_unique_indices, x_unique, _unique_array

    def gen_new_z(
        self,
        num_unique_indices,
        x_unique_array,
        var_values,
        batch_int,
        x_batch_array,
        contextual=False,
        a_tch=None,
        b_tch=None,
    ):
        """get var_values for all x's, repeated for the repeated x's"""
        # create dictionary from unique y's to var_values
        if not contextual:
            x_to_var_values_dict = {}
            for i in range(num_unique_indices):
                x_to_var_values_dict[tuple(tuple(v[i].flatten()) for v in x_unique_array)] = [
                    v[i] for v in var_values
                ]
            # initialize new var_values
            shapes = [torch.tensor(v.shape) for v in var_values]
            for i in range(len(shapes)):
                shapes[i][0] = batch_int
            new_var_values = [torch.zeros(*shape, dtype=s.DTYPE) for shape in shapes]

            # populate new_var_values using the dictionary
            for i in range(batch_int):
                values_list = x_to_var_values_dict[
                    tuple(tuple(v[i].flatten()) for v in x_batch_array)
                ]
                for j in range(len(var_values)):
                    new_var_values[j][i] = values_list[j]
            return new_var_values, a_tch, b_tch
        else:
            # create dictionary from unique y's to var_values
            x_to_var_values_dict = {}
            for i in range(num_unique_indices):
                x_to_var_values_dict[tuple(tuple(v[i].flatten()) for v in x_unique_array)] = (
                    [v[i] for v in var_values],
                    a_tch[i],
                    b_tch[i],
                )
            # initialize new var_values
            shapes = [torch.tensor(v.shape) for v in var_values]
            for i in range(len(shapes)):
                shapes[i][0] = batch_int
            new_var_values = [torch.zeros(*shape, dtype=s.DTYPE) for shape in shapes]
            ab_shapes = [torch.tensor(a_tch.shape), torch.tensor(b_tch.shape)]
            ab_shapes[0][0] = batch_int
            ab_shapes[1][0] = batch_int
            new_a_tch = torch.zeros(*ab_shapes[0], dtype=s.DTYPE)
            new_b_tch = torch.zeros(*ab_shapes[1], dtype=s.DTYPE)
            # populate new_var_values using the dictionary
            for i in range(batch_int):
                values_list = x_to_var_values_dict[
                    tuple(tuple(v[i].flatten()) for v in x_batch_array)
                ]
                for j in range(len(var_values)):
                    new_var_values[j][i] = values_list[0][j]
                new_a_tch[i] = values_list[1]
                new_b_tch[i] = values_list[2]
        return new_var_values, new_a_tch, new_b_tch

    def grid(
        self,
        rholst=s.RHO_LST_DEFAULT,
        settings = DS
    ):
        r"""
        Perform gridsearch to find optimal :math:`\rho`-ball around data.

        Args:
        rholst : np.array, optional
            The list of :math:`\rho` to iterate over. "Default np.logspace(-3, 1, 20)
        settings:
            TrainerSettings

        Returns:
        A pandas data frame with information on each :math:`\rho` having the following columns:
            Opt_val: float
                The objective value of the Robust Problem
            Lagrangian_val: float
                The value of the lagrangian function applied to the training data
            Eval_val: float
                The value of the lagrangian function applied to the evaluation data
            Rho: float
                The rho value
        """
        self._multistage = False
        self.settings = settings
        self.settings.predictor = settings.predictor
        if self.settings.data is None:
            self.settings.data = self.unc_set.data
        if settings.contextual:
            if settings.predictor is None:
                raise ValueError("Missing NN-Model")

        self.train_flag = False
        df = pd.DataFrame(columns=["Rho"])
        if self.settings.indices_dict is None:
            self.settings.indices_dict = self.unc_set.indices_dict
        self._split_dataset(settings.test_percentage, settings.validate_percentage, settings.seed)
        self.cvxpylayer = self.create_cvxpylayer()

        grid_stats = GridStats()

        lam = 1000 * torch.ones(self.num_g_total, dtype=s.DTYPE)
        # initialize torches
        rho_tch = self._gen_rho_tch(1)
        a_tch_init, b_tch_init, alpha = self._init_torches(
            settings.init_A, settings.init_b, settings.init_alpha, self.u_train_set
        )

        x_batch_array, num_unique_indices, x_unique, x_unique_array = self.gen_unique_x(
            self.x_test_tch
        )

        x_batch_array_t, num_unique_indices_t, x_unique_t, x_unique_array_t = self.gen_unique_x(
            self.x_validate_tch
        )

        for rho in rholst:
            rho_tch = torch.tensor(rho * settings.init_rho,
                                   requires_grad=self.train_flag, dtype=s.DTYPE)
            if settings.contextual:
                a_tch_init, b_tch_init,radius = self.create_predictor_tensors(x_unique)
            z_unique = self.cvxpylayer(
                rho_tch,
                *self.cp_param_tch,
                *x_unique,
                a_tch_init,
                b_tch_init,
                solver_args=settings.solver_args,
            )
            new_z_batch, a_tch_init, b_tch_init = self.gen_new_z(
                num_unique_indices,
                x_unique_array,
                z_unique,
                self.test_size,
                x_batch_array,
                settings.contextual,
                a_tch_init,
                b_tch_init,
            )

            if settings.contextual:
                a_tch_init, b_tch_init,radius = self.create_predictor_tensors(
                    x_unique_t)
            z_unique_t = self.cvxpylayer(
                rho_tch,
                *self.cp_param_tch,
                *x_unique_t,
                a_tch_init,
                b_tch_init,
                solver_args=settings.solver_args,
            )

            new_z_batch_t, _, _ = self.gen_new_z(
                num_unique_indices_t,
                x_unique_array_t,
                z_unique_t,
                self.validate_size,
                x_batch_array_t,
                settings.contextual,
                a_tch_init,
                b_tch_init,
            )

            train_stats = TrainLoopStats(
                step_num=0, train_flag=self.train_flag, num_g_total=self.num_g_total
            )
            with torch.no_grad():
                new_z_batch = self._reduce_variables(new_z_batch)
                new_z_batch_t = self._reduce_variables(new_z_batch_t)
                test_args = self.order_args(z_batch=new_z_batch,
                                             x_batch=self.x_test_tch,
                                             u_batch=self.u_test_tch)
                obj_test = self.evaluation_metric(self.test_size, test_args, settings.quantiles)
                cvar_test = self.evaluation_cvar(self.test_size, test_args)
                in_sample_test = self.train_objective(self.test_size, test_args)

                prob_violation_test = self.prob_constr_violation(self.test_size, test_args)
                _, var_vio = self.lagrangian(self.test_size, test_args, alpha, lam, 1, settings.eta)

                test_args_t = self.order_args(
                    z_batch=new_z_batch_t, x_batch=self.x_validate_tch, u_batch=self.u_validate_tch
                )

                cvar_train = self.evaluation_cvar(self.validate_size, test_args_t)
                obj_train = self.evaluation_metric(self.validate_size,
                                                    test_args_t,
                                                    settings.quantiles)
                in_sample_cost = self.train_objective(self.validate_size, test_args_t)

                prob_violation_train = self.prob_constr_violation(self.validate_size, test_args_t)
                _, var_vio_train = self.lagrangian(
                    self.validate_size, test_args_t, alpha, lam, 1, settings.eta
                )

            train_stats.update_test_stats(cvar_test[1],in_sample_test,
                                          cvar_test[0],
                                          obj_test,
                                          prob_violation_test, var_vio)
            train_stats.update_train_stats(cvar_test[1].item(),
                                           in_sample_cost.item(),
                                    cvar_train[0].item(), None,
                                    obj_train, prob_violation_train,
                                    var_vio_train)
            grid_stats.update(train_stats, cvar_test[0], rho_tch, a_tch_init, new_z_batch)

            new_row = train_stats.generate_test_row(
                self._calc_coverage, a_tch_init,b_tch_init,
                alpha, self.u_test_tch,rho_tch, self.unc_set, new_z_batch,
                settings.contextual, [self.x_test_tch])
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        self.orig_problem._trained = True
        self.unc_set._trained = True

        if settings.contextual:
            self.unc_set.a.value = (grid_stats.minrho * a_tch_init[0]).detach().numpy().copy()
            self.unc_set.b.value = (b_tch_init[0]).detach().numpy().copy()
            b_value = self.unc_set.b.value[0]
        else:
            self.unc_set.a.value = (grid_stats.minrho * a_tch_init).detach().numpy().copy()
            self.unc_set.b.value = (b_tch_init).detach().numpy().copy()
            b_value = self.unc_set.b.value
        return Result(
            self,
            self.problem_canon,
            df,
            None,
            None,
            self.unc_set.a.value,
            b_value,
            grid_stats.minrho.detach().numpy().copy(),
            grid_stats.minval,
            grid_stats.z_batch,
        )


class TrainLoopStats:
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
            Probability of constraint violation over train set
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
            return torch.tensor(0.0, dtype=s.DTYPE)

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

    def update_train_stats(self, worst_cost, in_sample_cost,
                           cvar_cost,temp_lagrangian,
                           obj, prob_violation_train, train_constraint):
        """
        This function updates the statistics after each training iteration
        """
        self.worst_train = worst_cost
        self.in_sample_cost = in_sample_cost
        self.cvar_val = cvar_cost
        self.tot_lagrangian = temp_lagrangian
        self.trainval = obj[1].item()
        self.prob_violation_train = prob_violation_train.detach().numpy()
        self.violation_train = train_constraint.numpy().sum() / self.num_g_total

    def update_test_stats(self, worst_cost, in_sample_test, cvar_cost,
                          obj_test, prob_violation_test, var_vio):
        """
        This function updates the statistics after each training iteration
        """
        self.worst_test = worst_cost.item()
        self.in_sample_test = in_sample_test.item()
        self.test_cvar = cvar_cost.item()
        self.lower_testval = obj_test[0].item()
        self.testval = obj_test[1].item()
        self.upper_testval = obj_test[2].item()
        self.prob_violation_test = prob_violation_test.detach().numpy()
        self.violation_test = var_vio.numpy().sum() / self.num_g_total

    def update_validate_stats(self,worst_cost, in_sample_cost,cvar_cost,
                              obj_vali, prob_violation_vali, var_vio_vali):
        """
        This function updates the statistics after each training iteration
        """
        self.worst_vali = worst_cost.item()
        self.in_sample_vali = in_sample_cost.item()
        self.vali_cvar = cvar_cost.item()
        self.lower_valival = obj_vali[0].item()
        self.valival = obj_vali[1].item()
        self.upper_valival = obj_vali[2].item()
        self.prob_violation_vali = prob_violation_vali.detach().numpy()
        self.violation_vali = var_vio_vali.numpy().sum() / self.num_g_total

    def generate_train_row(
        self, a_tch, rho_tch, lam, mu, alpha, contextual=False, linear=False, predictor=None
    ):
        """
        This function generates a new row with the statistics
        """
        row_dict = {
            "Lagrangian_val": self.tot_lagrangian.item(),
            "Train_val": self.trainval,
            "Train_cvar": self.cvar_val,
            "Worst_val": self.worst_train,
            "Train_insample": self.in_sample_cost,
            "Probability_violations_train": self.prob_violation_train,
            "Violations_train": self.violation_train,
            "Avg_prob_train": np.mean(self.prob_violation_train),
        }
        row_dict["step"] = self.step_num
        row_dict["A_norm"] = np.linalg.norm(a_tch.detach().numpy().copy())
        row_dict["lam_list"] = lam.detach().numpy().copy()
        row_dict["mu"] = mu
        row_dict["alpha"] = alpha.item()
        row_dict["alphagrad"] = alpha.grad
        if contextual:
            row_dict["gradnorm"] = [np.linalg.norm(
                list(predictor.parameters())[param_ind].data.detach().numpy()) \
                    for param_ind in range(len(list(predictor.parameters())))]
            row_dict["grad"] = list(predictor.parameters())[0].data.detach().numpy()
        else:
            row_dict["gradnorm"] = np.linalg.norm(a_tch.grad)
            row_dict["grad"] = a_tch.grad
        row_dict["Rho"] = rho_tch.detach().numpy().copy()
        new_row = pd.Series(row_dict)
        return new_row

    def generate_test_row(
        self,
        calc_coverage,
        a_tch,
        b_tch,
        alpha,
        test_tch,
        rho_tch,
        uncset,
        z_batch=None,
        contextual=False,
        x_test_tch=None,
    ):
        """
        This function generates a new row with the statistics
        """
        coverage_test = calc_coverage(
            test_tch, a_tch, b_tch, uncset._rho * rho_tch, uncset.p, contextual, x_test_tch
        )
        row_dict = {
            "Test_worst": self.worst_test,
            "Test_insample": self.in_sample_test,
            "Test_val": self.testval,
            "Test_cvar": self.test_cvar,
            "Lower_test": self.lower_testval,
            "Upper_test": self.upper_testval,
            "Probability_violations_test": self.prob_violation_test,
            "Violations_test": self.violation_test,
            "Coverage_test": coverage_test,
            "Avg_prob_test": np.mean(self.prob_violation_test),
            "z_vals": z_batch,
            "x_vals": x_test_tch,
            "Rho": rho_tch.detach().numpy().copy(),
        }
        row_dict["step"] = (self.step_num,)
        if not self.train_flag:
            row_dict["Validate_worst"] = self.worst_train
            row_dict["Validate_insample"] = self.in_sample_cost
            row_dict["Validate_cvar"] = self.cvar_val
            row_dict["Validate_val"] = self.trainval
            row_dict["Probability_violations_validate"] = self.prob_violation_train
            row_dict["Violations_validate"] = self.violation_train
            row_dict["Avg_prob_validate"] = np.mean(self.prob_violation_train)
        new_row = pd.Series(row_dict)
        return new_row

    def generate_validation_row(
        self,
        calc_coverage,
        a_tch,
        b_tch,
        alpha,
        vali_tch,
        rho_tch,
        uncset,
        z_batch=None,
        contextual=False,
        x_validate_tch=None,
    ):
        """
        This function generates a new row with the statistics
        """
        coverage_vali = calc_coverage(
            vali_tch, a_tch, b_tch, uncset._rho * rho_tch, uncset.p, contextual, x_validate_tch
        )
        row_dict = {
            "Validate_worst": self.worst_vali,
            "Validate_insample": self.in_sample_vali,
            "Validate_cvar": self.vali_cvar,
            "Validate_val": self.valival,
            "Lower_validate": self.lower_valival,
            "Upper_validate": self.upper_valival,
            "Probability_violations_validate": self.prob_violation_vali,
            "Violations_validate": self.violation_vali,
            "Coverage_validate": coverage_vali,
            "Avg_prob_validate": np.mean(self.prob_violation_vali),
            "z_vals": z_batch,
            "x_vals": x_validate_tch,
            "Rho": rho_tch.detach().numpy().copy(),
        }
        row_dict["step"] = (self.step_num,)
        if not self.train_flag:
            row_dict["Train_worst"] = self.worst_train
            row_dict["Train_cvar"] = self.cvar_val
            row_dict["Train_val"] = self.trainval
            row_dict["Train_insample"] = self.in_sample_cost
            row_dict["Probability_violations_train"] = self.prob_violation_train
            row_dict["Violations_train"] = self.violation_train
            row_dict["Avg_prob_train"] = np.mean(self.prob_violation_train)
        new_row = pd.Series(row_dict)
        return new_row

class GridStats:
    """
    This class contains useful information for grid search
    """

    def __init__(self):
        self.minval = float("inf")
        self.z_batch = 0

    def update(self, train_stats, obj, rho_tch, a_tch, z_batch):
        """
        This function updates the best stats in the grid search.

        Args:
            train_stats:
                The train stats
            obj:
                Calculated test objective
            rho_tch:
                Rho torch
            a_tch
                A torch
            z_batch
                Variable values
        """
        if train_stats.testval <= self.minval:
            self.minval = obj
            self.minrho = rho_tch.clone()
            self.minT = a_tch.clone()
            self.z_batch = z_batch


class Result(ABC):
    """A class to store the results of training"""

    def __init__(
        self,
        prob,
        probnew,
        df,
        df_test,
        df_validate,
        A,
        b,
        rho,
        obj,
        z,
        a_history=None,
        b_history=None,
        rho_history=None,
        predictor=None,
    ):
        self._final_prob = probnew
        self._problem = prob
        self._df = df
        self._df_test = df_test
        self._df_validate = df_validate
        self._A = A
        self._b = b
        self._obj = obj
        self._z = z
        self._rho = rho
        self._a_history = a_history
        self._b_history = b_history
        self._rho_history = rho_history
        self._predictor = predictor

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
    def df_validate(self):
        return self._df_validate

    @property
    def final_problem(self):
        return self._final_prob

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def rho(self):
        return self._rho

    @property
    def obj(self):
        return self._obj

    @property
    def var_values(self):
        return self._x

    @property
    def uncset_iters(self):
        return self._a_history, self._b_history

    @property
    def predictor(self):
        return self._predictor
