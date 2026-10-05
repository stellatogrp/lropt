import contextlib
import copy
import gc
import time
import warnings
from abc import ABC
from dataclasses import dataclass

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
from cvxro.torch_expression_generator import feasibility_constraints
from cvxro.train.predictors.covpred import CovPredictor
from cvxro.train.predictors.linear import LinearPredictor
from cvxro.train.settings import DEFAULT_SETTINGS as DS

# Import settings directly to use throughout the file
from cvxro.train.simulator import DefaultSimulator
from cvxro.train.utils import (
    EVAL_INPUT_CASE,
    eval_constr_indicator,
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
from cvxro.uncertainty_sets.support import (
    project_into_support,
    support_is_empty,
    support_matrices,
)
from cvxro.utils import unique_list
from cvxro.violation_checker.settings import VIOLATION_TOL
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
        self.g_feas = getattr(self.problem_canon, "g_feas", [None] * len(self.problem_canon.g))
        self._last_eval_args = None
        self._last_solved = None
        self.violation_checker = None
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
        radius = None
        if self.settings.contextual:
            a_tch, b_tch, radius = self.create_predictor_tensors(x_0)

        cost = 0.0
        # Shaped [num_g_total], matching what train_constraint returns (its H is one
        # entry per uncertain constraint group). It has to be a tensor rather than a
        # plain float because when both constraint_cvar and delage_coverage are False
        # it is never accumulated into, and downstream code unconditionally calls
        # .detach() on it -- but it also has to be the FULL length, since the
        # accumulation below is in-place: with torch.zeros(1) and num_g_total > 1 that
        # raises "output with shape [1] doesn't match the broadcast shape [k]".
        constraint_cost = torch.zeros(self.num_g_total, dtype=s.DTYPE)
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
            x_solve = x_t
            z_t = self.cvxpylayer(rho_tch, *self.cp_param_tch, *x_t, a_tch, b_tch,
                                  solver_args=self.settings.solver_args)
            training = not (self._test_flag or self._validate_flag)
            train_max_unsolved = float(getattr(self.settings, "train_max_unsolved", 0.0))
            drop_unsolved = training and not self._multistage and train_max_unsolved > 0
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
                # Recording per-sample status forgoes the early exit, and this runs once
                # per line-search attempt while training, so only do it on the
                # evaluation passes -- or when a training batch may drop its unsolvable
                # samples, which needs to know which ones they are.
                per_sample=(not training) or drop_unsolved,
            )
            self._train_n_dropped = 0
            if drop_unsolved and constraints_status is CONSTRAINT_STATUS.INFEASIBLE:
                keep = torch.as_tensor(self.violation_checker.last_sample_feasible)
                n_bad = int((~keep).sum())
                if n_bad < keep.numel() and n_bad / keep.numel() <= train_max_unsolved:
                    # A few contexts have no feasible robust dispatch at this set; the
                    # layer returns a junk solution for them (a power-balance row off by
                    # hundreds of MW on dcopf, 2026-09-18) and a whole batch would
                    # otherwise be undone for it. Train on the solvable samples.
                    z_t, x_t, u_0, a_tch, b_tch, radius = self._drop_unsolved_samples(
                        keep, z_t, x_t, u_0, a_tch, b_tch, radius)
                    self.settings.kwargs_simulator["batch_int"] = int(keep.sum())
                    self._train_n_dropped = n_bad
                    constraints_status = CONSTRAINT_STATUS.FEASIBLE
            z_t = self._reduce_variables(z_t)
            if not self._multistage:
                eval_args = self.order_args(z_t, x_t, u_0)
                self.settings.kwargs_simulator["eval_args"] = eval_args
                self.settings.kwargs_simulator["u_train"] = u_0
                # Kept for _record_feasibility: _train_loop only has the numpy history
                # of x, not the ordered tensors the constraint callables need.
                self._last_eval_args = (eval_args, u_0.shape[0])
                self._last_solved = self.violation_checker.last_sample_feasible

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
                # dist_loss regresses the in-set indicator of u_0 onto the contexts, so
                # its x must be the contexts a_tch/b_tch/radius were fit from -- x_solve,
                # not the state simulate() has since advanced to.
                input_tensors = self.create_input_tensors(x_solve)
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

    def _drop_unsolved_samples(self, keep, z_t, x_t, u_0, a_tch, b_tch, radius):
        """Restrict one training batch to the samples in `keep` (a bool tensor over the
        batch): the layer's outputs, the contexts, the uncertainty realisations and the
        predicted set parameters. Anything without a batch dimension is returned as is."""
        n = keep.numel()

        def _sel(t, unbatched_ndim):
            if isinstance(t, torch.Tensor) and t.dim() > unbatched_ndim and t.shape[0] == n:
                return t[keep]
            return t
        z_t = [_sel(z, len(var.shape)) for z, var in zip(z_t, self.cvxpylayer.variables)]
        x_t = [x[keep] for x in x_t]
        u_0 = _sel(u_0, 1)
        a_tch = _sel(a_tch, len(self.unc_set._a.shape))
        b_tch = _sel(b_tch, len(self.unc_set._b.shape))
        radius = _sel(radius, 0) if isinstance(radius, torch.Tensor) else radius
        return z_t, x_t, u_0, a_tch, b_tch, radius

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

    def create_predictor_tensors(self,x_batch,project=True):
        """Create the tensors of a's and b's using the trained linear model.

        `project=False` returns the predictor's RAW center. It exists for one caller:
        LinearPredictor.pretrain_func regresses this center against the observed u, and a
        clamped center has zero gradient exactly at the points whose prediction is out of
        support -- the ones the regression most needs to move. Everything that BUILDS a
        set takes the projected center; only the fit sees the raw one.
        """
        a_shape = self.unc_set._a.shape
        b_shape = self.unc_set._b.shape
        input_tensors = self.create_input_tensors(x_batch)
        a_tch, b_tch, radius = self.settings.predictor.forward(
            input_tensors,a_shape,b_shape,self.train_flag)
        if project:
            b_tch = self._project_center_to_support(b_tch)
        return a_tch, b_tch, radius

    def _support_spec(self):
        """The set's support as `(C, d)`, cached, or None if it declares none.

        Cached because it is read on every forward pass and never changes for a given
        problem: `add_support_type` assembles `lb`/`ub`/`sum_eq` into `_c`/`_d` when the
        UncertainParameter is constructed, and an explicit `c`/`d` is set there too.
        """
        if not hasattr(self, "_support_cache"):
            self._support_cache = support_matrices(self.unc_set)
        return self._support_cache

    def _project_center_to_support(self, b_tch):
        """Project the center into the uncertainty set's support, whatever its shape.

        This is the choke point for A(x), b(x) on BOTH the training and the evaluation
        paths, so a center projected here is projected everywhere the set is used -- which
        is what makes the guarantee hold for the radius search and the reported metrics
        too, not only for the AL loop.

        Why it matters: if b lies outside the support, the set {A z + b : ||z|| <= rho} can
        fail to meet it AT ALL. U is then empty, the robust constraint quantifies over
        nothing and so bounds nothing, and the solver returns an enormous decision with
        status `optimal`. No violation metric catches that -- the decision really is
        feasible for a constraint that says nothing -- so it surfaces only as a few
        contexts whose cost is orders of magnitude off. With b inside the support, b itself
        lies in both the ellipsoid and the support, so U is nonempty BY CONSTRUCTION.

        Every support kind is handled, not just `lb`/`ub`: a box is projected by clamping
        and any other polyhedron (an explicit `c`/`d`, a `sum_eq`, or a mix) by Dykstra's
        alternating projections. See cvxro.uncertainty_sets.support.

        A support with no points at all -- `lb` above `ub`, or contradictory rows -- is a
        modelling error rather than something to project onto, so it raises here instead of
        failing later as an unbounded solve.

        See TrainerSettings.project_center_to_support to turn this off.
        """
        if not getattr(self.settings, "project_center_to_support", True):
            return b_tch
        spec = self._support_spec()
        if spec is None:
            return b_tch
        rows, rhs = spec
        projected, residual = project_into_support(b_tch, rows, rhs)
        if residual > 1e-6:
            if support_is_empty(rows, rhs):
                raise ValueError(
                    "the uncertainty set's support is empty -- its rows have no common "
                    "point, so no center can lie inside it and every set built from it "
                    "would be empty. Check lb/ub (is some lb above its ub?) and any "
                    "explicit c/d.")
            warnings.warn(
                f"projecting the center into the support left a residual of {residual:.2e}"
                " after the iteration limit; the set may be empty at some contexts. This "
                "is a polyhedral support that Dykstra did not resolve, not a box.",
                stacklevel=2,
            )
        self._warn_center_moved(b_tch, projected)
        return projected

    def _warn_center_moved(self, before, after):
        """Warn ONCE, with the size of the effect, when the projection actually binds.

        A center outside the support is worth knowing about even though projecting it is
        safe: it says the mean model is predicting values the data can never take, which is
        a statement about the fit, not about the support.
        """
        if getattr(self, "_warned_center_projection", False):
            return
        shift = (after - before).abs()
        if before.dim() > 1:
            moved = int(torch.count_nonzero(shift.max(dim=-1).values > 1e-12).item())
            total = before.shape[0]
        else:
            moved = int(bool(shift.max() > 1e-12))
            total = 1
        if moved:
            warnings.warn(
                f"projected {moved}/{total} centers into the uncertainty set's support; "
                "without this those sets could be empty and their robust constraints "
                "vacuous. Set TrainerSettings.project_center_to_support=False to disable.",
                stacklevel=2,
            )
            self._warned_center_projection = True

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
            self.u_train_set = self._init_uncertain_parameter
        elif self._multistage and isinstance(\
            self.settings.predictor,LinearPredictor):
            predictor = self.settings.predictor
            if (predictor.predict or predictor.predict_cov
                    or predictor.knn_cov or predictor.knn_mean):
                if (self._init_uncertain_parameter is None):
                    raise ValueError(
                    "You must provide init_uncertain_param when using a linear "
                    "predictor that reads the training rows (predict_mean, "
                    "predict_cov, knn_cov or knn_mean)"
                )
                self.u_train_tch = torch.tensor(
                self._init_uncertain_parameter, requires_grad=True, dtype=s.DTYPE
            )
                self.x_train_tch = self.settings.simulator.init_state(
                    self._init_uncertain_parameter.shape[0],
                      self.settings.seed,
                                            **self.settings.kwargs_simulator)
                self.u_train_set = self._init_uncertain_parameter

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
        # Start inside the support, for the same reason create_predictor_tensors ends
        # there: a center outside it can put the whole set outside, and an empty set makes
        # the robust constraint vacuous rather than infeasible. This covers the INITIAL
        # center on every path, contextual or not. It does NOT cover a non-contextual
        # center that the AL loop then trains OUT of the support -- see the warning in
        # _train_loop.
        b_tch = self._project_center_to_support(b_tch).detach().requires_grad_(
            self.train_flag)
        a_tch = init_tensor

        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)
        return a_tch, b_tch, alpha

    def _project_step(self):
        """The projected-gradient hook handed to take_step (see project_trained_center)."""
        return self.project_trained_center(getattr(self, "_trained_center", None))

    def project_trained_center(self, b_tch):
        """Projected-gradient hook: keep a NON-CONTEXTUAL trained center in the support.

        The contextual path re-derives b from the predictor on every forward pass, so
        projecting in create_predictor_tensors covers it. A non-contextual run instead
        trains `b` as a bare leaf tensor and hands it to the layer directly, so without
        this the AL loop can walk it out of the support even though it started inside --
        and once the set stops meeting the support, the constraint is vacuous and the
        decision unbounded, silently.

        Applied in place after each optimizer step, which makes the update a projected
        gradient step: descend, then project back onto the feasible set. Returns whether
        anything moved, so the caller can report it.
        """
        if not getattr(self.settings, "project_center_to_support", True):
            return False
        spec = self._support_spec()
        if spec is None or b_tch is None:
            return False
        rows, rhs = spec
        with torch.no_grad():
            projected, _residual = project_into_support(b_tch, rows, rhs)
            moved = bool((projected - b_tch).abs().max() > 1e-12)
            b_tch.copy_(projected)
        return moved

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
                        (a_tch[i].T @ torch.linalg.inv(
                            a_tch[i] @ a_tch[i].T + 0.00001*torch.eye(a_tch[i].shape[0])))
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
                        (a_tch.T @ torch.linalg.inv(
                            (a_tch @ a_tch.T) + 0.00001*torch.eye(a_tch.shape[0])))
                        @ (dset[datind] - b_tch),
                        p,
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

    def evaluation_metric(self, batch_int, eval_args, quantiles, mask=None):
        """
        This function evaluates the evaluation metric over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the evaluation function
            quantiles:
                The upper and lowerx quantiles to be returned.
            mask:
                Optional boolean tensor over the batch, restricting the statistic to the
                selected samples (e.g. the feasible ones).
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
            eta_target= None,
            mask=mask,
        )

    def evaluation_cvar(self, batch_int, eval_args, eta = None, mask=None):
        """
        This function evaluates the evaluation CVaR over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the evaluation function
            eta:
                The eta value for the CVaR.
            mask:
                Optional boolean tensor over the batch, restricting the statistic to the
                selected samples (e.g. the feasible ones).
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
            eta_target = eta,
            mask=mask,
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

    def prob_constr_violation(self, batch_int, eval_args, exclude_epigraph=False):
        """
        This function evaluates the probability of constraint violation
        of all uncertain constraints over the batched set.
        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_args:
                The arguments of the constraints
            exclude_epigraph:
                Whether to leave the epigraph-objective row out, giving the probability of
                genuine infeasibility rather than of certificate violation. Only differs
                when an epigraph row was identified; see max_of_uncertain's
                epigraph_index.
        Returns:
            The average among all evaluated J x N pairs
        """
        g = feasibility_constraints(self.problem_canon) if exclude_epigraph else self.g
        return eval_prob_constr_violation(
            g=g, g_shapes=self.g_shapes, batch_int=batch_int, eval_args=eval_args
        )

    def failure_indicator(self, batch_int, eval_args, solved=None):
        """
        Per-sample indicator of the full failed-decision event: ANY uncertain constraint
        row (the epigraph-objective row included) is violated on the realized data, OR
        the returned robust solution failed the reformulation check. This is the event a
        probabilistic guarantee at level eta is about; prob_constr_violation's per-row
        rates and feasibility_stats' physical-feasibility fraction are both rates of
        strictly smaller or non-nested events and must not be quoted in its place.
        Args:
            batch_int:
                The number of samples in the batch
            eval_args:
                The arguments of the constraints
            solved:
                Optional boolean array from the robust-solution check (e.g.
                _grid_solved_mask). Unsolved samples count as failures, because
                cvxpylayers returns a degenerate decision rather than failing.
        Returns:
            A boolean tensor of length batch_int
        """
        indicator = eval_constr_indicator(
            g=self.g, g_shapes=self.g_shapes, batch_int=batch_int, eval_args=eval_args
        ).sum(axis=0) > 0
        if solved is not None and len(solved) == batch_int:
            solved = torch.as_tensor(np.asarray(solved), dtype=torch.bool)
            indicator = indicator | ~solved
        return indicator

    def prob_failure(self, batch_int, eval_args, solved=None):
        """The probability of the full failed-decision event; see failure_indicator."""
        return self.failure_indicator(batch_int, eval_args, solved=solved).double().mean()

    def _grid_solved_mask(self, settings, split, rho_tch):
        """Per-sample solvability of the robust problem, or None when not tracked.

        Takes a SplitTensors rather than loose tensors so that the decisions, the
        uncertainty-set parameters and the contexts it checks them against are
        guaranteed to describe the same samples.
        """
        if not settings.track_unsolved:
            return None
        # grid() does not go through train(), which is where the checker is normally built.
        if getattr(self, "violation_checker", None) is None:
            self.violation_checker = ViolationChecker(
                self.cvxpylayer, self.problem_no_unc.constraints,
                tol=getattr(settings, "violation_tol", VIOLATION_TOL),
            )
        self.violation_checker.check_constraints(
            z_batch=split.z,
            rho_mult_parameter=self._rho_mult_parameter,
            rho_tch=rho_tch,
            cp_parameters=self._cp_parameters,
            cp_param_tch=self.cp_param_tch,
            x_parameters=self._x_parameters,
            x_batch=split.x_batch,
            shape_parameters=self._shape_parameters,
            shape_torches=[split.a_tch, split.b_tch],
            per_sample=True,
        )
        return self.violation_checker.last_sample_feasible

    def _grid_unique_contexts(self, x_tch, batch_int):
        """Deduplicate one split's contexts, once, for reuse across the rho sweep."""
        x_batch_array, num_unique, x_unique, x_unique_array = self.gen_unique_x(x_tch)
        return _UniqueContexts(
            x_tch=x_tch, batch_int=batch_int, x_batch_array=x_batch_array,
            num_unique=num_unique, x_unique=x_unique, x_unique_array=x_unique_array,
        )

    def _grid_solve_split(self, settings, rho_tch, contexts, a_tch_init, b_tch_init):
        """Solve one split at this rho and return its tensors bound together.

        Both the layer solve and the fan-out back to sample resolution happen here, so
        the A/b that end up in the returned SplitTensors are necessarily the ones this
        split was solved with. That is the property that keeps the four A/b lengths
        alive in a grid() iteration from ever being crossed.
        """
        a_tch, b_tch = a_tch_init, b_tch_init
        if settings.contextual:
            a_tch, b_tch, _radius = self.create_predictor_tensors(contexts.x_unique)
        z_unique = self.cvxpylayer(
            rho_tch,
            *self.cp_param_tch,
            *contexts.x_unique,
            a_tch,
            b_tch,
            solver_args=settings.solver_args,
        )
        z_batch, a_tch, b_tch = self.gen_new_z(
            contexts.num_unique,
            contexts.x_unique_array,
            z_unique,
            contexts.batch_int,
            contexts.x_batch_array,
            settings.contextual,
            a_tch,
            b_tch,
        )
        return SplitTensors(x_batch=contexts.x_tch, z=z_batch, a_tch=a_tch,
                            b_tch=b_tch, contextual=settings.contextual)

    def feasibility_stats(self, batch_int, eval_args, quantiles, eta=None, solved=None):
        """
        This function evaluates the fraction of feasible samples and the evaluation
        statistics restricted to them.
        Args:
            batch_int:
                The number of samples in the batch
            eval_args:
                The arguments of the constraints and the evaluation function
            quantiles:
                The upper and lower quantiles to be returned
            eta:
                The eta value for the CVaR
            solved:
                Optional boolean array saying which samples the robust problem could
                actually be solved for. Unsolved samples count as infeasible, because
                cvxpylayers returns a degenerate decision rather than failing.
        Returns:
            (frac_feasible, (lower, mean, upper), (cvar, quantile), frac_unsolved)
        """
        mask = self.feasibility_indicator(batch_int, eval_args)
        frac_unsolved = float("nan")
        if solved is not None and len(solved) == batch_int:
            solved = torch.as_tensor(np.asarray(solved), dtype=torch.bool)
            frac_unsolved = 1.0 - solved.double().mean().item()
            mask = mask & solved
        frac_feasible = mask.double().mean().item()
        obj = self.evaluation_metric(batch_int, eval_args, quantiles, mask=mask)
        cvar = self.evaluation_cvar(batch_int, eval_args, eta=eta, mask=mask)
        return frac_feasible, obj, cvar, frac_unsolved

    def _record_feasibility(self, train_stats, split):
        """Record the feasibility statistics of the split just evaluated by monte_carlo."""
        if self._last_eval_args is None:
            return
        eval_args, batch_int = self._last_eval_args
        with torch.no_grad():
            train_stats.update_feasibility(
                split,
                *self.feasibility_stats(
                    batch_int, eval_args, self.settings.quantiles, solved=self._last_solved
                ),
            )

    def feasibility_indicator(self, batch_int, eval_args):
        """
        This function evaluates, per sample, whether every uncertain constraint other than
        the epigraph-objective row holds.
        Args:
            batch_int:
                The number of samples in the batch
            eval_args:
                The arguments of the constraints
        Returns:
            A boolean tensor of length batch_int
        """
        indicator = eval_constr_indicator(
            g=feasibility_constraints(self.problem_canon),
            g_shapes=self.g_shapes,
            batch_int=batch_int,
            eval_args=eval_args,
        )
        return indicator.sum(axis=0) == 0

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
            batch_int, eval_args=eval_args, alpha=alpha, eta=eta, kappa=kappa
        )
        H_plus = torch.maximum(H, torch.zeros_like(H))
        if isinstance(mu, torch.Tensor):
            return F + lam @ H_plus + (mu / 2) @ (H_plus ** 2), H.detach()
        else:
            return F + lam @ H_plus + (mu / 2) * (torch.linalg.norm(H_plus) ** 2), H.detach()

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
                     lam: torch.Tensor, mu: "float | torch.Tensor",
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
                h_plus = torch.maximum(constr_cost, torch.zeros_like(constr_cost))
                if self.num_g_total > 1:
                    if isinstance(mu, torch.Tensor):
                        fin_cost = cost + lam @ h_plus + (mu / 2) @ (h_plus ** 2)
                    else:
                        fin_cost = (
                            cost + lam @ h_plus
                            + (mu / 2) * (torch.linalg.norm(h_plus) ** 2)
                        )
                else:
                    if isinstance(mu, torch.Tensor):
                        fin_cost = cost + lam * h_plus + (mu[0] / 2) * (h_plus ** 2)
                    else:
                        fin_cost = cost + lam * h_plus + (mu / 2) * (h_plus ** 2)
            elif self.settings.delage_coverage:
                fin_cost = (1-self.settings.coverage_gamma)*cost + \
                    self.settings.coverage_gamma*constr_cost
            else:
                fin_cost = cost

            if self.settings.line_search:
                search_condition = fin_cost <= self.settings.line_search_threshold*prev_fin_cost
                # if not isinstance(search_condition, bool):
                #     search_condition = search_condition.all().item()
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
                prev_states = take_step(rho_min=getattr(self.settings, "rho_min", 0.001), opt=opt,
                                        rho_tch=rho_tch, scheduler=None,
                                        update_state = False,
                                        prev_states = prev_states,
                                        project=self._project_step)
            elif constraint_status is CONSTRAINT_STATUS.FEASIBLE \
                and violation_counter == current_iter_line_search -1:
                restore_step_size(opt, num_steps=violation_counter,
                                  step_mult = self.settings.line_search_mult)
                opt.zero_grad()
                fin_cost.backward()
                prev_fin_cost = fin_cost.clone().detach()
                break
        if constraint_status is CONSTRAINT_STATUS.INFEASIBLE and step_num != 0:
            # Timed out: every retaken step was infeasible. The step size was reduced
            # once per attempt and would otherwise stay at step_mult**max_iter for the
            # rest of training (dcopf 2026-09-17: 0.8**50 ~ 1e-5, after which no later
            # step could move the predictor). The parameters are already at the last
            # (tiny) retaken step; only the optimizer's lr needs restoring.
            restore_step_size(opt, num_steps=current_iter_line_search - 1,
                              step_mult = self.settings.line_search_mult)
        # KNOWN, DELIBERATELY UNFIXED (2026-09-21, lropt_power/multistage HANDOFF §5r):
        # `prev_fin_cost` is updated above on acceptance but never returned, so the
        # caller compares every step against its initial np.inf and the descent check
        # `fin_cost <= line_search_threshold * prev_fin_cost` has never rejected a step
        # (unchanged since 7acbcdc, 2026-01-13). Returning it is NOT sufficient: the
        # retry loop never re-takes a step (take_step happens once, in the caller), so a
        # non-improving step has no backtracking branch, spins to exhaustion, and then
        # restore_step_size(violation_counter) multiplies lr by (1/step_mult)**(max_iter-1)
        # for shrinks that never happened -- measured as Violations_train 9.6e12 in
        # test_portfolio_intro and a NaN profile in test_profile_predictor when the check
        # was made live. A real fix re-takes the step with the reduced lr inside this
        # loop; that is a design decision for the authors. Every reported result to date
        # was trained with this check inert: Adam + infeasibility undo + interval updates.
        # settings.line_search_mode = "backtrack" selects _backtracking_line_search, which
        # does re-take the step from a parameter + optimizer-state snapshot; this legacy
        # path is kept bit-identical as the default.
        return constraint_status, fin_cost, avg_cost, \
            prob_violation_train, constr_cost, cvar_cost, in_sample_cost, worst_cost

    def _evaluate_fin_cost(self, a_tch, b_tch, rho_tch, alpha, lam, mu, seed_num):
        """One training rollout at the current parameters, and the objective the line
        search judges (the augmented Lagrangian, or the coverage/plain cost). The same
        formula as _line_search's loop body, factored out for _backtracking_line_search."""
        self._validate_flag = False
        self._test_flag = False
        cost, constr_cost, _, _, constraint_status, \
            avg_cost, prob_violation_train, _, cvar_cost, \
            in_sample_cost, worst_cost = self.loss_and_constraints(
                a_tch=a_tch,
                b_tch=b_tch,
                seed=self.settings.seed + seed_num + 1,
                rho_tch=rho_tch,
                alpha=alpha,
                batch_size=self.settings.batch_size,
            )
        if not self._default_simulator:
            avg_cost = avg_cost.repeat(3)

        if self.settings.constraint_cvar:
            h_plus = torch.maximum(constr_cost, torch.zeros_like(constr_cost))
            if self.num_g_total > 1:
                if isinstance(mu, torch.Tensor):
                    fin_cost = cost + lam @ h_plus + (mu / 2) @ (h_plus ** 2)
                else:
                    fin_cost = (
                        cost + lam @ h_plus
                        + (mu / 2) * (torch.linalg.norm(h_plus) ** 2)
                    )
            else:
                if isinstance(mu, torch.Tensor):
                    fin_cost = cost + lam * h_plus + (mu[0] / 2) * (h_plus ** 2)
                else:
                    fin_cost = cost + lam * h_plus + (mu / 2) * (h_plus ** 2)
        elif self.settings.delage_coverage:
            fin_cost = (1-self.settings.coverage_gamma)*cost + \
                self.settings.coverage_gamma*constr_cost
        else:
            fin_cost = cost
        return constraint_status, fin_cost, avg_cost, \
            prob_violation_train, constr_cost, cvar_cost, in_sample_cost, worst_cost

    def _backtracking_line_search(self, step_num, a_tch, b_tch, rho_tch, alpha, lam, mu,
                                  opt, scheduler, seed_num, prev_fin_cost,
                                  resample_seed_num=None):
        """settings.line_search_mode = "backtrack": a real backtracking line search.

        Unlike _line_search, this method OWNS the optimizer step. The gradient in
        p.grad (from the previous accepted step's backward) fixes the direction; the
        parameters and the optimizer state (Adam's moments and step count) are
        snapshotted, and try k re-takes the step from that snapshot at
        lr0 * line_search_mult**k. A try is accepted when it is FEASIBLE and
        fin_cost <= line_search_threshold * prev_fin_cost (the descent test is skipped
        when settings.line_search is False, or prev_fin_cost is inf). The lr is reset
        to lr0 exactly afterwards, never multiplied back.

        Exhaustion (max_iter_line_search tries, none accepted):
          * last try FEASIBLE (it never descended): accept it -- the smallest step --
            as the settings docstring always promised ("proceed with the last step
            size").
          * last try INFEASIBLE: revert to the snapshot and re-evaluate there, so the
            returned statistics and the gradient belong to the parameters actually
            kept; returns INFEASIBLE so the caller logs the time-out. With a fixed
            training batch the next step repeats the same direction from the same
            state, so this recurs until the batch (seed_num) or the multipliers change.

        settings.line_search_on_exhaust = "revert" (default "keep_last") treats the
        FEASIBLE-but-never-descended case like the infeasible one: the snapshot is
        restored and no step is taken (and the scheduler does not tick). "keep_last"
        drifts uphill by one lr0 * mult**(tries-1) step per rejection (lropt_power c5:
        18 such steps in 19).

        resample_seed_num (settings.line_search_resample, passed by the caller): on
        either rejection the kept point is re-evaluated on THAT batch instead of the
        current one, and its cost becomes prev_fin_cost, so the next step has a new
        direction and a same-batch descent reference. Without it a rejection on a fixed
        batch recurs every step. The caller advances its batch counter on rejection.

        Returns the same eight values as _line_search plus prev_fin_cost (the accepted
        objective, or the old one if nothing was accepted), the number of tries, and
        an outcome string ("accepted", "no_descent", "infeasible", "first_step").
        """
        mult = self.settings.line_search_mult
        tries = self._max_iter_line_search

        def _descends(fin_cost):
            if not self.settings.line_search or not np.isfinite(float(prev_fin_cost)):
                return True
            return bool(fin_cost <= self.settings.line_search_threshold * prev_fin_cost)

        if step_num == 0:
            # No step yet: evaluate the initial point once, as _line_search does.
            out = self._evaluate_fin_cost(a_tch, b_tch, rho_tch, alpha, lam, mu, seed_num)
            constraint_status, fin_cost = out[0], out[1]
            if constraint_status is CONSTRAINT_STATUS.FEASIBLE:
                opt.zero_grad()
                fin_cost.backward()
                prev_fin_cost = fin_cost.clone().detach()
            return (*out, prev_fin_cost, 1, "first_step")

        params = [p for group in opt.param_groups for p in group["params"]]
        theta0 = [p.detach().clone() for p in params]
        # deepcopy at capture AND at every restore: Optimizer.state_dict() returns
        # references, and load_state_dict() adopts the given tensors without copying
        # (torch 2.13), so the first retried opt.step() would otherwise mutate the
        # snapshot in place -- the same aliasing class as undo_step's 2026-09-18 bug.
        state0 = copy.deepcopy(opt.state_dict())
        lr0 = [group["lr"] for group in opt.param_groups]

        def _restore(k):
            with torch.no_grad():
                for p, t in zip(params, theta0):
                    p.copy_(t)
            opt.load_state_dict(copy.deepcopy(state0))
            for group, lr in zip(opt.param_groups, lr0):
                group["lr"] = lr * mult ** k

        outcome = None
        for k in range(tries):
            _restore(k)
            take_step(rho_min=getattr(self.settings, "rho_min", 0.001), opt=opt, rho_tch=rho_tch,
                      scheduler=None,
                      project=self._project_step)
            out = self._evaluate_fin_cost(a_tch, b_tch, rho_tch, alpha, lam, mu, seed_num)
            constraint_status, fin_cost = out[0], out[1]
            if constraint_status is CONSTRAINT_STATUS.FEASIBLE and _descends(fin_cost):
                outcome = "accepted"
                break
        n_tries = k + 1

        revert = getattr(self.settings, "line_search_on_exhaust", "keep_last") == "revert"
        eval_seed = seed_num if resample_seed_num is None else resample_seed_num
        if outcome is None and constraint_status is CONSTRAINT_STATUS.FEASIBLE:
            outcome = "no_descent"
            if revert:
                _restore(0)
            if revert or resample_seed_num is not None:
                out = self._evaluate_fin_cost(a_tch, b_tch, rho_tch, alpha, lam, mu,
                                              eval_seed)
                fin_cost = out[1]
        elif outcome is None:
            outcome = "infeasible"
            _restore(0)
            out = self._evaluate_fin_cost(a_tch, b_tch, rho_tch, alpha, lam, mu, eval_seed)
            # Report the time-out even if the snapshot itself re-evaluates FEASIBLE.
            out = (CONSTRAINT_STATUS.INFEASIBLE, *out[1:])
            fin_cost = out[1]

        for group, lr in zip(opt.param_groups, lr0):
            group["lr"] = lr
        stepped = outcome == "accepted" or (outcome == "no_descent" and not revert)
        if stepped and scheduler:
            # One scheduler tick per step actually taken, after the lr is back at lr0.
            scheduler.step()
        if outcome != "infeasible" or resample_seed_num is not None:
            prev_fin_cost = fin_cost.clone().detach()
        opt.zero_grad()
        fin_cost.backward()
        return (*out, prev_fin_cost, n_tries, outcome)

    def _eval_grad_context(self):
        """no_grad for the train loop's evaluation rollouts, unless a branch needs a
        graph during the rollout itself (delage_coverage's dist_loss). See the call
        sites for the measurement that motivates it."""
        if getattr(self.settings, "delage_coverage", False):
            return contextlib.nullcontext()
        return torch.no_grad()

    def _train_loop(self, init_num):
        _init_start_time = time.time()
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
        # The center the projected-gradient hook keeps inside the support. Only the
        # NON-CONTEXTUAL path trains this tensor: with a predictor, b is re-derived from it
        # on every forward pass and projected in create_predictor_tensors instead, so
        # projecting this leaf as well would be projecting a tensor nothing reads.
        self._trained_center = None if self.settings.contextual else b_tch

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
        # Predictors that expose param_groups() (e.g. LinearPredictor's
        # cov_lr_mult) get per-submodule learning rates instead of every
        # parameter sharing self.settings.lr.
        #
        # NOT in the size-only phase, though: when trained_shape is True
        # (the second phase of the train_shape+train_size flow), the shape
        # predictor is meant to be FROZEN and only rho/alpha trained --
        # _set_train_variables has already returned [rho_tch, alpha] for
        # exactly that. Overwriting it with the predictor's parameter groups
        # here silently dropped rho_tch from the optimizer entirely, so the
        # "train rho" phase never moved rho at all (result.rho stayed at
        # init_rho), while instead continuing to train the shape predictor
        # at lr_size -- typically much larger than the phase-1 lr, which
        # wrecked the covariance fit (observed: condition number 2.7 -> 670).
        if (self.settings.contextual
                and hasattr(self.settings.predictor, "param_groups")
                and not self.settings.trained_shape):
            # param_groups() REPLACES the variable list above, which contained rho_tch, so
            # on this path rho takes no gradient and stays at init_rho for the whole run.
            # settings.train_rho puts it back, as its own group at the base lr. Default
            # False, so every arm trained before this existed is bit-identical.
            extra_params = [alpha]
            if getattr(self.settings, "train_rho", False):
                extra_params.append(rho_tch)
            variables = self.settings.predictor.param_groups(self.settings.lr,
                                                             extra_params=extra_params)
        if self.settings.optimizer == "SGD":
            opt = s.OPTIMIZERS[self.settings.optimizer](
                variables, lr=self.settings.lr, momentum=self.settings.momentum
            )
        else:
            opt = s.OPTIMIZERS[self.settings.optimizer](variables,
                                                         lr=self.settings.lr,
                                                         weight_decay = self.settings.weight_decay)

        if self.settings.scheduler:
            scheduler_ = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.settings.lr_step_size, gamma=self.settings.lr_gamma
            )
        else:
            scheduler_ = None
        # y's and cvxpylayer begin
        lam = self.settings.init_lam * torch.ones(self.num_g_total, dtype=s.DTYPE)
        mu = self.settings.init_mu
        strategy = self.settings.dual_update_strategy

        if strategy == "pid":
            xi = torch.zeros(self.num_g_total, dtype=s.DTYPE)
            pid_step = 0

        seed_num = 0
        curr_cvar = np.inf
        prev_fin_cost = np.inf
        if self._default_simulator:
            self.settings.kwargs_simulator = {
                "trainer": self,
            }
        prev_states = []
        backtrack = getattr(self.settings, "line_search_mode", "legacy") == "backtrack"
        resample = backtrack and getattr(self.settings, "line_search_resample", False)
        # Batches drawn after a rejection (line_search_resample): offset by a stride far
        # beyond any AL update count so they never coincide with seed_num's own batches.
        batch_bump = 0
        _BUMP_STRIDE = 10007
        # batch_per_step: the training batch is seeded by the step, not by seed_num (which
        # moves only at AL updates). Offset so these seeds never coincide with seed_num's.
        batch_per_step = bool(getattr(self.settings, "batch_per_step", False))
        _STEP_BASE = 20011
        _init_elapsed_time = time.time() - _init_start_time
        for step_num in range(self.settings.num_iter):
            if step_num > 0 and not backtrack:
                prev_states = take_step(rho_min=getattr(self.settings, "rho_min", 0.001), opt=opt,
                                        rho_tch=rho_tch, scheduler=scheduler_,
                                        project=self._project_step)
            train_stats = TrainLoopStats(
                step_num=step_num, train_flag=self.train_flag, num_g_total=self.num_g_total
            )

            torch.manual_seed(self.settings.seed + step_num)
            # Per-step predictor hook, duck-typed exactly like param_groups() below:
            # a predictor that defines on_step(step_num) is told which step is about to
            # be differentiated, so it can change WHICH of its parameters take a
            # gradient during this step (e.g. alternating between a frozen mean head and
            # a frozen covariance head). Predictors without the method are unaffected.
            # Called before the backward, so the flags it sets govern this step's
            # gradients; opt.step() at the top of the next iteration then moves only the
            # parameters that received one.
            if hasattr(self.settings.predictor, "on_step"):
                self.settings.predictor.on_step(step_num)
            _train_start_time = time.time()
            batch_seed = _STEP_BASE + step_num if batch_per_step else seed_num
            if batch_per_step:
                prev_fin_cost = np.inf   # new batch, so the last step's cost is not comparable
            if backtrack:
                constraint_status, fin_cost, avg_cost, \
                    prob_violation_train, constr_cost, cvar_cost, in_sample_cost, worst_cost, \
                    prev_fin_cost, ls_tries, ls_outcome = \
                    self._backtracking_line_search(
                        step_num=step_num, a_tch=a_tch, b_tch=b_tch, rho_tch=rho_tch,
                        alpha=alpha, lam=lam, mu=mu, opt=opt, scheduler=scheduler_,
                        seed_num=batch_seed + _BUMP_STRIDE * batch_bump,
                        prev_fin_cost=prev_fin_cost,
                        resample_seed_num=(batch_seed + _BUMP_STRIDE * (batch_bump + 1)
                                           if resample else None))
                if resample and ls_outcome in ("no_descent", "infeasible"):
                    batch_bump += 1
            else:
                constraint_status, fin_cost, avg_cost, \
                    prob_violation_train, constr_cost, cvar_cost, in_sample_cost,worst_cost = \
                    self._line_search(step_num=step_num, a_tch=a_tch, b_tch=b_tch,
                                      rho_tch=rho_tch, alpha=alpha, lam=lam, mu=mu, opt=opt,
                                      prev_states=prev_states,
                                      seed_num = batch_seed,
                                      prev_fin_cost = prev_fin_cost)
            _train_elapsed_time = time.time() - _train_start_time
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
                # The batch and the multipliers change here, so the old objective is not
                # comparable: backtrack always resets (a cross-batch descent test is noise).
                if self.settings.reset_prev_cost_on_al_update or backtrack:
                    prev_fin_cost = np.inf

                if strategy == "classic":
                    # Original behavior (unchanged)
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

                elif strategy == "pid":
                    # νPI dual variable update (arXiv:2406.04558)
                    e_t = constr_cost.detach()
                    nu = self.settings.pid_nu
                    Ki = self.settings.pid_Ki
                    Kp = self.settings.pid_Kp

                    if pid_step == 0:
                        update = Ki * e_t
                    else:
                        bc = 1.0 - nu ** pid_step
                        xi_bc = xi / bc
                        update = Ki * e_t + Kp * (1 - nu) * (e_t - xi_bc)

                    xi = nu * xi + (1 - nu) * e_t
                    pid_step += 1

                    lam += torch.minimum(
                        update,
                        self.settings.lambda_update_max
                        * torch.ones(self.num_g_total, dtype=s.DTYPE),
                    )
                    lam = torch.maximum(
                        lam, torch.zeros(self.num_g_total, dtype=s.DTYPE)
                    )

                    if (
                        torch.norm(e_t)
                        <= self.settings.lambda_update_threshold * curr_cvar
                    ):
                        curr_cvar = torch.norm(e_t)
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
            # Initialization is charged to the FIRST step's Train_time, so that the
            # `df["Train_time"].sum()` every experiment already computes is the method's
            # full training cost -- previously it omitted the predictor fit entirely,
            # which for a num_iter=0 baseline is the whole method. Init_time reports the
            # same quantity separately for auditing; do not add it to Train_time again.
            _init_charge = _init_elapsed_time if step_num == 0 else 0.0
            new_row["Train_time"] = _train_elapsed_time + _init_charge
            new_row["Init_time"] = _init_charge
            if backtrack:
                new_row["Line_search_tries"] = ls_tries
                new_row["Line_search_outcome"] = ls_outcome
                new_row["Line_search_batch"] = batch_bump
            rows.append(new_row)
            _train_row, _validation_row = new_row, None

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
                _validate_start_time = time.time()
                # Evaluation only: the outputs are recorded via .detach()/.item(),
                # yet the rollout was building a full autograd graph at
                # test_batch_size every iteration -- measured at 0.10 GB per
                # test-sample-stage, the same as a TRAINING sample, i.e. 23 GB instead
                # of 9 at test_batch 48 (lropt_power/multistage, 2026-09-20).
                # _evaluate_rho_multistage already runs this call under no_grad.
                # EXCEPT under delage_coverage: dist_loss differentiates INSIDE the
                # rollout (`loss_conf.backward(create_graph=True)`, ~l.1215), on
                # evaluation passes too, and under no_grad that loss has no grad_fn
                # (test_delage_coverage caught it). That branch keeps its graph.
                with self._eval_grad_context():
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
                _validate_elapsed_time = time.time() - _validate_start_time
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
                self._record_feasibility(train_stats, "validate")
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
                new_row["Validate_time"] = _validate_elapsed_time
                rows_validate.append(new_row)
                _validation_row = new_row

            if step_num % self.settings.test_frequency == 0:
                self._test_flag = True
                self._validate_flag= False
                _test_start_time = time.time()
                # Evaluation only: the outputs are recorded via .detach()/.item(),
                # yet the rollout was building a full autograd graph at
                # test_batch_size every iteration -- measured at 0.10 GB per
                # test-sample-stage, the same as a TRAINING sample, i.e. 23 GB instead
                # of 9 at test_batch 48 (lropt_power/multistage, 2026-09-20).
                # _evaluate_rho_multistage already runs this call under no_grad.
                # EXCEPT under delage_coverage: dist_loss differentiates INSIDE the
                # rollout (`loss_conf.backward(create_graph=True)`, ~l.1215), on
                # evaluation passes too, and under no_grad that loss has no grad_fn
                # (test_delage_coverage caught it). That branch keeps its graph.
                with self._eval_grad_context():
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
                _test_elapsed_time = time.time() - _test_start_time
                record_avg_cost = avg_cost
                if not self._default_simulator:
                    record_avg_cost = avg_cost.repeat(3)

                train_stats.update_test_stats(worst_cost,
                                              in_sample_cost,cvar_cost,
                    record_avg_cost, prob_constr_violation, val_cost_constr.detach()
                )
                self._record_feasibility(train_stats, "test")
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
                new_row["Test_time"] = _test_elapsed_time
                rows_test.append(new_row)

            # Per-step END hook, the read-only twin of on_step above and duck-typed the
            # same way: a predictor that defines on_step_end(step_num, train_row,
            # validation_row) is handed this step's training statistics (the row just
            # appended to `rows`: Lagrangian_val, Train_val, Train_cvar,
            # Violations_train, lam_list, mu, ...) and, on the steps where the validation
            # rollout ran (step_num % validate_frequency == 0), the validation row
            # (Validate_val, Validate_cvar, Violations_validate, ...); None otherwise.
            # This is how a predictor can ADAPT its per-step behaviour (which head takes
            # gradients, via on_step) to the training trajectory -- e.g. by measuring
            # which head lowers the held-out objective -- without reaching into the
            # loop's locals. Nothing is passed back, and predictors without the method
            # are unaffected, so every existing run is bit-identical.
            if hasattr(self.settings.predictor, "on_step_end"):
                self.settings.predictor.on_step_end(step_num, _train_row, _validation_row)

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
            x_batch
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

        # if not self._multistage:
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
        self.violation_checker = ViolationChecker(
            self.cvxpylayer, self.problem_no_unc.constraints,
            tol=getattr(self.settings, "violation_tol", VIOLATION_TOL))
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
            x_batch
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
                x_batch_s
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
            result = Result(
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
                x=x_batch[index_chosen] + x_batch_s[0]
            )
        else:
            result = Result(
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
                x=x_batch[index_chosen]
        )

        # Post-training rho calibration
        if self.settings.tune_rho:
            calibrated_rho = self._tune_rho(result)
            result._rho = calibrated_rho
            self._rho_mult_parameter[0].value = calibrated_rho

        return result

    def compare_predictors(
            self,
            settings: s.TrainerSettings | None = s.TrainerSettings(),
            predictors_list = [],rho_list = []):
        """This function computes the validation and testing values for a
        list of predictors.

        The fields below are ARGUMENTS of this call, not durable configuration, so the
        caller's settings object is left exactly as it was found. It used to keep the
        last radius and predictor written onto it, which is silent and easy to miss:
        `grid()` and `bisection()` read `rholst` as a MULTIPLIER on `init_rho`, so any
        code that swept radii here and then ran a grid on the same settings searched a
        rescaled range without any error. See experiments/newsvendor/news.py, where the
        mean-variance baseline's grid inherited the sweep's last radius. Several call
        sites already worked around this by re-assigning `settings.init_rho` by hand
        before their own search; that is no longer necessary.
        """
        saved = {name: getattr(settings, name) for name in
                 ("num_iter", "max_batch_size", "init_rho", "predictor",
                  "initialize_predictor", "contextual")}
        settings.num_iter = 0
        settings.max_batch_size = np.inf
        test_dfs = []
        validate_dfs = []
        num_rho = len(rho_list)
        if len(predictors_list)==1:
            if num_rho > 1:
                predictors_list = predictors_list*num_rho
        try:
            for ind, predictor in enumerate(predictors_list):
                if predictor is None:
                    settings.contextual = False
                settings.predictor = predictor
                settings.initialize_predictor = False
                settings.init_rho = rho_list[ind]
                result = self.train(settings=settings)
                test_dfs.append(result.df_test)
                validate_dfs.append(result.df_validate)
        finally:
            for name, value in saved.items():
                setattr(settings, name, value)
        return pd.concat(validate_dfs), pd.concat(test_dfs)

    def _tune_rho(self, result):
        """Post-training rho calibration via grid search on validation data.

        Finds the smallest rho where validation violation probability <= target_eta.
        Uses compare_predictors to evaluate each candidate rho.
        """
        trained_rho = result.rho
        target_eta = self.settings.target_eta
        n_grid = self.settings.tune_rho_n_grid
        lo, hi = self.settings.tune_rho_range

        # Save settings (compare_predictors -> train() overwrites self.settings)
        saved_settings = self.settings

        # Build log-spaced grid of absolute rho values
        rho_grid = np.logspace(np.log10(lo), np.log10(hi), n_grid) * trained_rho

        best_rho = trained_rho
        best_violation = np.inf
        feasible_found = False

        for rho_val in rho_grid:
            eval_settings = copy.copy(saved_settings)
            eval_settings.tune_rho = False  # prevent recursive calibration
            try:
                df_valid, _ = self.compare_predictors(
                    settings=eval_settings,
                    predictors_list=[result.predictor],
                    rho_list=[rho_val],
                )
                viol = df_valid["Avg_prob_validate"].iloc[0]
            except Exception:
                continue

            if viol <= target_eta:
                if not feasible_found or rho_val < best_rho:
                    best_rho = rho_val
                    best_violation = viol
                    feasible_found = True
            elif not feasible_found:
                if viol < best_violation:
                    best_violation = viol
                    best_rho = rho_val

        # Restore settings
        self.settings = saved_settings

        return best_rho

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

    def _rho_search_setup(self, settings):
        """Prepare the state every radius of a rho search reuses.

        Splitting the dataset, building the CvxpyLayer and deduplicating the contexts
        cost the same whether one radius is evaluated or a hundred, so grid() and
        bisection() both do them exactly once, here.

        Honours settings.multistage. This used to hard-set self._multistage = False, so
        grid() and bisection() scored the SINGLE-STAGE problem for a multistage model.
        What that looked like depended on the problem: some raise in setup ("need at
        least one array to concatenate"), others run to completion and report numbers
        for a different model than train() evaluates, with no error at all. Callers
        worked around it by bisecting over compare_predictors themselves. A multistage
        radius is now evaluated by the same monte_carlo rollout train() uses, so the
        workaround is no longer needed.
        """
        self._multistage = bool(settings.multistage)
        self.settings = settings
        self.settings.predictor = settings.predictor
        if self.settings.data is None:
            self.settings.data = self.unc_set.data
        if settings.contextual and settings.predictor is None:
            raise ValueError("Missing NN-Model")

        self.train_flag = False
        if self.settings.indices_dict is None:
            self.settings.indices_dict = self.unc_set.indices_dict
        self._split_dataset(settings.test_percentage, settings.validate_percentage, settings.seed)

        if self._multistage:
            # The same runtime state train() builds for a rollout. Kept here rather
            # than shared with train() so the single-stage search stays byte-identical.
            self._init_uncertain_parameter = settings.init_uncertain_param
            self._init_context = settings.init_context
            self.num_g_total = 1
            self.cvxpylayer = settings.policy if settings.policy \
                else self.create_cvxpylayer()
            if settings.simulator:
                self.simulator = settings.simulator
                self._default_simulator = False
            else:
                self.simulator = DefaultSimulator(self)
                self._default_simulator = True
            self.violation_checker = ViolationChecker(
                self.cvxpylayer, self.problem_no_unc.constraints,
                tol=getattr(settings, "violation_tol", VIOLATION_TOL))
            self._max_iter_line_search = settings.max_iter_line_search
            self.x_endind = settings.x_endind
        else:
            self.cvxpylayer = self.create_cvxpylayer()

        lam = 1000 * torch.ones(self.num_g_total, dtype=s.DTYPE)
        a_tch_init, b_tch_init, alpha = self._init_torches(
            settings.init_A, settings.init_b, settings.init_alpha, self.u_train_set
        )
        return _RhoSearchContext(
            lam=lam,
            alpha=alpha,
            a_tch_init=a_tch_init,
            b_tch_init=b_tch_init,
            # Deduplicated once, outside the sweep: the contexts do not depend on rho.
            # A multistage rollout draws its own contexts from simulator.init_state at
            # every radius, so there is nothing to deduplicate up front.
            test=None if self._multistage
            else self._grid_unique_contexts(self.x_test_tch, self.test_size),
            validate=None if self._multistage
            else self._grid_unique_contexts(self.x_validate_tch, self.validate_size),
        )

    def _evaluate_rho(self, rho, settings, ctx, grid_stats):
        """Evaluate one radius: solve both splits, score them, build the result row.

        `rho` is a multiplier on settings.init_rho, not an absolute radius. Returns the
        row alongside the test split, which is what the uncertainty set is written from.
        """
        if self._multistage:
            return self._evaluate_rho_multistage(rho, settings, ctx, grid_stats)
        rho_tch = torch.tensor(rho * settings.init_rho,
                               requires_grad=self.train_flag, dtype=s.DTYPE)
        # Each split accumulates only the segments that are its own work, so
        # Test_time and Validate_time are disjoint and their sum is the radius's real
        # cost. They used to be two overlapping stopwatches started before each solve
        # and stopped much later, which billed the validate solve and the test metrics
        # to both splits, and the test feasibility check to the validate split.
        test_time = _Stopwatch()
        validate_time = _Stopwatch()

        with test_time:
            test_split = self._grid_solve_split(
                settings, rho_tch, ctx.test, ctx.a_tch_init, ctx.b_tch_init)
        with validate_time:
            validate_split = self._grid_solve_split(
                settings, rho_tch, ctx.validate, ctx.a_tch_init, ctx.b_tch_init)

        train_stats = TrainLoopStats(
            step_num=0, train_flag=self.train_flag, num_g_total=self.num_g_total
        )
        with torch.no_grad():
            with test_time:
                new_z_batch = self._reduce_variables(test_split.z)
                test_args = self.order_args(z_batch=new_z_batch,
                                             x_batch=test_split.x_batch,
                                             u_batch=self.u_test_tch)
                obj_test = self.evaluation_metric(
                    self.test_size, test_args, settings.quantiles)
                cvar_test = self.evaluation_cvar(self.test_size, test_args)
                in_sample_test = self.train_objective(self.test_size, test_args)

                # One solved mask per split, shared by the violation probability and the
                # feasibility statistics, so Avg_prob_* and Pct_infeasible_* judge the
                # same solutions. Avg_prob_* is the FULL failed-decision event
                # (certificate violated OR robust-solution check failed): it used to be
                # the certificate event alone, which let a radius pass selection while
                # its failed solves pushed the true violation rate past the target.
                solved_test = self._grid_solved_mask(settings, test_split, rho_tch)
                prob_violation_test = self.prob_failure(
                    self.test_size, test_args, solved=solved_test)
                _, var_vio = self.lagrangian(
                    self.test_size, test_args, ctx.alpha, ctx.lam, 1, settings.eta)

            with validate_time:
                new_z_batch_t = self._reduce_variables(validate_split.z)
                test_args_t = self.order_args(
                    z_batch=new_z_batch_t, x_batch=validate_split.x_batch,
                    u_batch=self.u_validate_tch
                )

                cvar_train = self.evaluation_cvar(self.validate_size, test_args_t)
                obj_train = self.evaluation_metric(self.validate_size,
                                                    test_args_t,
                                                    settings.quantiles)
                in_sample_cost = self.train_objective(self.validate_size, test_args_t)

                solved_validate = self._grid_solved_mask(settings, validate_split, rho_tch)
                prob_violation_train = self.prob_failure(
                    self.validate_size, test_args_t, solved=solved_validate)
                _, var_vio_train = self.lagrangian(
                    self.validate_size, test_args_t, ctx.alpha, ctx.lam, 1, settings.eta
                )

            with test_time:
                train_stats.update_feasibility(
                    "test",
                    *self.feasibility_stats(
                        self.test_size, test_args, settings.quantiles,
                        solved=solved_test,
                    ),
                )
            with validate_time:
                train_stats.update_feasibility(
                    "validate",
                    *self.feasibility_stats(
                        self.validate_size, test_args_t, settings.quantiles,
                        solved=solved_validate,
                    ),
                )

        train_stats.update_test_stats(cvar_test[1],in_sample_test,
                                      cvar_test[0],
                                      obj_test,
                                      prob_violation_test, var_vio)
        # In a rho search, update_train_stats carries the VALIDATE-split statistics --
        # generate_test_row surfaces them as the Validate_* columns when
        # train_flag is False. Every argument here must therefore come from the
        # validate split: cvar_train/obj_train/in_sample_cost/prob_violation_train
        # are all computed from test_args_t (x_validate_tch/u_validate_tch) above.
        # The worst-case slot used to be passed cvar_test[1], which silently put
        # the TEST eta-quantile in the Validate_worst column and made any radius
        # selection on that column read test data.
        train_stats.update_train_stats(cvar_train[1].item(),
                                       in_sample_cost.item(),
                                cvar_train[0].item(), None,
                                obj_train, prob_violation_train,
                                var_vio_train)
        grid_stats.update(train_stats, cvar_test[0], rho_tch, test_split.a_tch,
                          new_z_batch)

        # Coverage in generate_test_row is measured on u_test_tch/x_test_tch, so it
        # must use the test split's shape tensors.
        new_row = train_stats.generate_test_row(
            self._calc_coverage, test_split.a_tch, test_split.b_tch,
            ctx.alpha, self.u_test_tch,rho_tch, self.unc_set, new_z_batch,
            settings.contextual, [test_split.x_batch])
        new_row["Test_time"] = test_time.elapsed
        new_row["Validate_time"] = validate_time.elapsed
        return new_row, test_split

    def _evaluate_rho_multistage(self, rho, settings, ctx, grid_stats):
        """Evaluate one radius by ROLLOUT, the multistage counterpart of _evaluate_rho.

        A multistage radius cannot be scored by solving the layer once per context:
        the decision at period t depends on the state the previous periods left, so the
        only honest evaluation is a rollout. That is exactly what monte_carlo does, and
        what train() calls on its validate and test passes, so this reuses it rather
        than reimplementing the loop -- a radius probed here and the same radius seen
        during training are then measured by identical code.

        Row schema matches the single-stage path exactly, so grid(), bisection() and
        every caller that reads Avg_prob_validate / Validate_worst_feasible work
        unchanged. Coverage is 0 by construction (_calc_coverage returns 0 in
        multistage: a rollout has no single u sample to cover), and the feasibility
        columns come back NaN for the same reason they do in train() -- the per-sample
        constraint arguments are only assembled on the single-stage branch.
        """
        rho_tch = torch.tensor(rho * settings.init_rho,
                               requires_grad=False, dtype=s.DTYPE)
        train_stats = TrainLoopStats(
            step_num=0, train_flag=self.train_flag, num_g_total=self.num_g_total
        )
        test_time = _Stopwatch()
        validate_time = _Stopwatch()

        def rollout(seed):
            # monte_carlo -> loss_and_constraints refits a_tch/b_tch from each
            # rollout's own initial contexts when contextual, so the tensors passed
            # here are only the non-contextual fallback.
            return self.monte_carlo(a_tch=ctx.a_tch_init, b_tch=ctx.b_tch_init,
                                    rho_tch=rho_tch, alpha=ctx.alpha, seed=seed)

        with torch.no_grad():
            # Same seeds train() uses, so a radius scored here and the same radius
            # scored mid-training see the same draws.
            with validate_time:
                self._validate_flag, self._test_flag = True, False
                (_, vali_constr, x_hist_v, z_v, avg_cost_v, prob_vio_v,
                 u_v, cvar_v, in_sample_v, worst_v) = rollout(settings.seed + 10000)
            with test_time:
                self._test_flag, self._validate_flag = True, False
                (_, test_constr, x_hist_t, z_t, avg_cost_t, prob_vio_t,
                 u_t, cvar_t, in_sample_t, worst_t) = rollout(settings.seed)
        self._test_flag = self._validate_flag = False

        # A custom simulator reports one scalar average cost; the row wants the
        # (lower, mean, upper) triple the default simulator produces.
        obj_test = avg_cost_t if self._default_simulator else avg_cost_t.repeat(3)
        obj_vali = avg_cost_v if self._default_simulator else avg_cost_v.repeat(3)

        train_stats.update_test_stats(worst_t, in_sample_t, cvar_t,
                                      obj_test, prob_vio_t, test_constr.detach())
        # update_train_stats, not update_validate_stats: generate_test_row surfaces the
        # TRAIN slots as the Validate_* columns when train_flag is False, which is the
        # convention the single-stage rho search already follows.
        train_stats.update_train_stats(worst_v, in_sample_v, cvar_v, None,
                                       obj_vali, prob_vio_v, vali_constr.detach())

        split = self._multistage_split(x_hist_t, ctx)
        grid_stats.update(train_stats, cvar_t, rho_tch, split.a_tch, z_t)
        new_row = train_stats.generate_test_row(
            self._calc_coverage, split.a_tch, split.b_tch, ctx.alpha, u_t,
            rho_tch, self.unc_set, z_t, settings.contextual, split.x_batch)
        new_row["Test_time"] = test_time.elapsed
        new_row["Validate_time"] = validate_time.elapsed
        return new_row, split

    def _multistage_split(self, x_hist, ctx):
        """The shape tensors _finalize_rho_search writes onto the uncertainty set.

        Rebuilt from the rollout's INITIAL contexts -- x_hist[0] -- because that is the
        state the period-0 set was fit from, and period 0 is the only decision a
        receding-horizon policy actually commits.
        """
        x_0 = [torch.tensor(np.asarray(v), dtype=s.DTYPE) for v in x_hist[0]]
        if self.settings.contextual:
            a_tch, b_tch, _ = self.create_predictor_tensors(x_0)
        else:
            a_tch, b_tch = ctx.a_tch_init, ctx.b_tch_init
        return _MultistageSplit(a_tch=a_tch, b_tch=b_tch, x_batch=x_0)

    def _finalize_rho_search(self, settings, test_split, grid_stats, df):
        """Write the selected radius onto the uncertainty set and build the Result."""
        self.orig_problem._trained = True
        self.unc_set._trained = True

        # A contextual set has one ellipsoid per sample, so Result/unc_set can only
        # carry a representative. Take it from the test split, like grid_stats.z_batch
        # and the Test_* columns.
        if settings.contextual:
            self.unc_set.a.value = (
                grid_stats.minrho * test_split.a_tch[0]).detach().numpy().copy()
            self.unc_set.b.value = (test_split.b_tch[0]).detach().numpy().copy()
            b_value = self.unc_set.b.value[0]
        else:
            self.unc_set.a.value = (
                grid_stats.minrho * test_split.a_tch).detach().numpy().copy()
            self.unc_set.b.value = (test_split.b_tch).detach().numpy().copy()
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

    def grid(
        self,
        rholst=s.RHO_LST_DEFAULT,
        settings = DS,
        violation_col="Avg_prob_validate",
        objective_col="Validate_worst_feasible",
        max_unsolved=0.0,
        target_eta=None,
    ):
        r"""
        Perform gridsearch to find optimal :math:`\rho`-ball around data.

        Args:
        rholst : np.array, optional
            The list of :math:`\rho` to iterate over. "Default np.logspace(-3, 1, 20)
        settings:
            TrainerSettings
        violation_col, objective_col : str
            The validation columns to constrain and to minimize when picking the radius
            written onto the uncertainty set -- the same defaults, and the same selection
            rule (:func:`_select_probe`), as :meth:`bisection`. Radii where the robust
            problem was unsolvable, or which miss ``settings.target_eta``, are excluded;
            previously the pick came from :class:`GridStats`, which minimized the TEST
            objective with no regard to solvability or to the target.
        max_unsolved : float
            Fraction of instances allowed to be unsolvable before a radius is rejected;
            see :func:`_classify_probe`. Default 0 preserves the historical rule.
        target_eta : float, optional
            Violation target used to ACCEPT a radius, overriding
            ``settings.target_eta`` for the search only. Pass a value below the
            reported target to leave headroom where the validation split's violation
            is optimistic about the test split's -- e.g. a contextual predictor whose
            mean head is fit on the same data the radius is calibrated on. Kept
            separate from ``settings.target_eta`` because that field is also the
            quantile the CVaR columns are reported at, so lowering it would silently
            redefine the reported metric as well as the selection.

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
        ctx = self._rho_search_setup(settings)
        df = pd.DataFrame(columns=["Rho"])

        # Frac_unsolved_* is not just a reporting column: monte_carlo ANDs the solvability
        # mask into the feasibility mask, so with it off the samples where the robust
        # problem had no feasible decision are counted as feasible, and their degenerate
        # cvxpylayers decisions are averaged into Test_worst_feasible / Test_cvar_feasible.
        # bisection() forces it on for the same reason; so does grid() now, so the two
        # searches score a radius identically rather than only sampling it differently.
        saved_track_unsolved = settings.track_unsolved
        settings.track_unsolved = True
        probes = []
        try:
            for rho in rholst:
                # A fresh GridStats per radius, so each radius's own tensors stay
                # recoverable for _select_probe to choose between -- GridStats.update
                # selects on the TEST objective, which is neither the split nor the
                # criterion this search is meant to report.
                stats = GridStats()
                new_row, split = self._evaluate_rho(rho, settings, ctx, stats)
                probes.append(_RhoProbe(rho, new_row, split, stats,
                                        _classify_probe(new_row, settings,
                                                        violation_col, max_unsolved,
                                                        target_eta)))
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        finally:
            settings.track_unsolved = saved_track_unsolved

        if not probes:
            raise ValueError("rholst was empty; nothing to search over")
        try:
            chosen = _select_probe(probes, violation_col, objective_col,
                                   float(np.min(rholst)), float(np.max(rholst)))
        except ValueError:
            # Every radius unsolvable. bisection() raises here, but grid() is the path
            # every existing experiment runs and its Result.rho is not what they report
            # (they re-select downstream from `df`), so failing the whole run over an
            # attribute nobody reads would be gratuitous. Warn and keep the last radius.
            warnings.warn(
                "the robust problem was unsolvable at every radius in rholst; "
                "Result.rho/A/b are not meaningful. The per-radius `df` is still "
                "returned -- select from it on Frac_unsolved_* and target_eta.",
                stacklevel=2,
            )
            chosen = probes[-1]
        return self._finalize_rho_search(settings, chosen.split, chosen.stats, df)

    def bisection(
        self,
        rho_lo=0.01,
        rho_hi=3.0,
        settings=DS,
        tol=0.01,
        max_iters=25,
        violation_col="Avg_prob_validate",
        objective_col="Validate_worst_feasible",
        max_unsolved=0.0,
        target_eta=None,
    ):
        r"""
        Bisect over :math:`\rho` for the best validation performance, instead of
        scanning a list of radii as :meth:`grid` does.

        Returns the radius with the best validation worst-case among those whose
        validation violation meets ``settings.target_eta``. Both quantities are
        monotone in :math:`\rho` -- violation decreasing, worst-case increasing -- so
        the search is a root-find on the violation curve, and roughly
        :math:`\log_2` as many radii need evaluating as a grid of the same resolution.

        Args:
        rho_lo, rho_hi : float
            Bracket, as MULTIPLIERS on ``settings.init_rho`` -- the same convention as
            :meth:`grid`'s ``rholst`` and ``settings.tune_rho_range``.
        tol : float
            Stop once the bracket is this narrow in log-:math:`\rho`. The search
            closes on the boundary from above, so the radius returned is the
            smallest one *known* to meet the target, overshooting the true
            boundary by at most ``tol``; the worst-case rises with
            :math:`\rho`, so that overshoot is conservative. The default costs
            about 12 probes on a 300-fold bracket.
        max_iters : int
            Hard cap on bisection steps, in addition to the two endpoint probes.
        violation_col, objective_col : str
            The validation columns to constrain and to minimize.
        max_unsolved : float
            Fraction of instances allowed to be unsolvable before a radius is rejected;
            see :func:`_classify_probe`. Default 0 preserves the historical rule.
        target_eta : float, optional
            Violation target used to ACCEPT a radius, overriding
            ``settings.target_eta`` for the search only. Pass a value below the
            reported target to leave headroom where the validation split's violation
            is optimistic about the test split's -- e.g. a contextual predictor whose
            mean head is fit on the same data the radius is calibrated on. Kept
            separate from ``settings.target_eta`` because that field is also the
            quantile the CVaR columns are reported at, so lowering it would silently
            redefine the reported metric as well as the selection.

        Returns:
        A :class:`Result` whose ``df`` has one row per radius actually probed, in probe
        order, and whose ``rho``/``A``/``b`` come from the selected radius.
        """
        if not 0 < rho_lo < rho_hi:
            raise ValueError(f"need 0 < rho_lo < rho_hi, got {rho_lo} and {rho_hi}")

        # Frac_unsolved_validate is what keeps the search out of the degenerate region
        # (see _classify_probe), so it has to be populated whatever the caller set.
        saved_track_unsolved = settings.track_unsolved
        settings.track_unsolved = True
        try:
            probes = self._bisect_rho(
                rho_lo, rho_hi, settings, tol, max_iters, violation_col, max_unsolved,
                target_eta)
        finally:
            settings.track_unsolved = saved_track_unsolved

        chosen = _select_probe(probes, violation_col, objective_col, rho_lo, rho_hi)
        df = pd.DataFrame(columns=["Rho"])
        for probe in probes:
            df = pd.concat([df, probe.row.to_frame().T], ignore_index=True)
        return self._finalize_rho_search(settings, chosen.split, chosen.stats, df)

    def _bisect_rho(self, rho_lo, rho_hi, settings, tol, max_iters, violation_col,
                    max_unsolved=0.0, target_eta=None):
        """Probe radii until the feasible band's lower edge is bracketed to `tol`."""
        ctx = self._rho_search_setup(settings)
        probes = []

        def probe(rho):
            # A fresh GridStats per radius, so each probe's own tensors and objective
            # stay recoverable -- GridStats.update selects on the TEST objective, which
            # is not the criterion this search is built on.
            stats = GridStats()
            row, split = self._evaluate_rho(rho, settings, ctx, stats)
            probes.append(_RhoProbe(rho, row, split, stats,
                                    _classify_probe(row, settings, violation_col,
                                                    max_unsolved, target_eta)))
            return probes[-1]

        # Both early exits below lean on monotonicity: if the widest radius still
        # violates, nothing narrower can meet the target, and if the narrowest already
        # meets it, nothing narrower was in the bracket to find.
        if probe(rho_hi).status is _VIOLATING:
            return probes
        if probe(rho_lo).status is _OK:
            return probes

        lo, hi = np.log(rho_lo), np.log(rho_hi)
        for _ in range(max_iters):
            if hi - lo <= tol:
                break
            mid = 0.5 * (lo + hi)
            if probe(float(np.exp(mid))).status is not _OK:
                lo = mid
            else:
                hi = mid
        return probes


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
        self._feasibility = {}

    def update_feasibility(self, split, frac_feasible, obj, cvar, frac_unsolved=0.0):
        """
        This function records the feasibility statistics of one data split.
        Args:
            split:
                One of "test", "validate", "train"
            frac_feasible:
                Fraction of samples satisfying every non-epigraph uncertain constraint
            obj:
                evaluation_metric over the feasible samples, as (lower, mean, upper)
            cvar:
                evaluation_cvar over the feasible samples, as (cvar, quantile)
            frac_unsolved:
                Fraction of samples whose robust problem had no feasible decision
        """
        self._feasibility[split] = (frac_feasible, obj, cvar, frac_unsolved)

    def feasibility_row(self, split):
        """
        This function returns the feasibility columns of one data split, filled with NaN
        when they were not computed.
        """
        nan = float("nan")
        frac, obj, cvar, unsolved = self._feasibility.get(split, (nan, None, None, nan))
        prefix = "Test" if split == "test" else split.capitalize()
        return {
            f"{prefix}_worst_feasible": nan if cvar is None else float(cvar[1]),
            f"{prefix}_val_feasible": nan if obj is None else float(obj[1]),
            f"{prefix}_cvar_feasible": nan if cvar is None else float(cvar[0]),
            f"Frac_feasible_{split}": frac,
            f"Pct_infeasible_{split}": 1 - frac,
            f"Frac_unsolved_{split}": unsolved,
        }

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
        row_dict["mu"] = mu.detach().numpy().copy() if isinstance(mu, torch.Tensor) else mu
        row_dict["alpha"] = alpha.item()
        row_dict["alphagrad"] = alpha.grad
        if contextual:
            row_dict["gradnorm"] = [np.linalg.norm(
                list(predictor.parameters())[param_ind].data.detach().numpy()) \
                    for param_ind in range(len(list(predictor.parameters())))]
            row_dict["grad"] = list(predictor.parameters())[0].data.detach().numpy()
        else:
            # a_tch.grad is None whenever nothing backpropagated into the shape: a
            # NON-CONTEXTUAL fit, or num_iter=0. gradnorm is a diagnostic, so report
            # nan rather than raising -- np.linalg.norm(None) dies with an opaque
            # "unsupported operand type(s) for *: 'NoneType' and 'NoneType'", which is
            # how the MV, non-robust and Scenario baselines failed under a rollout.
            row_dict["gradnorm"] = (float("nan") if a_tch.grad is None
                                    else np.linalg.norm(a_tch.grad))
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
        row_dict.update(self.feasibility_row("test"))
        if not self.train_flag:
            row_dict["Validate_worst"] = self.worst_train
            row_dict["Validate_insample"] = self.in_sample_cost
            row_dict["Validate_cvar"] = self.cvar_val
            row_dict["Validate_val"] = self.trainval
            row_dict["Probability_violations_validate"] = self.prob_violation_train
            row_dict["Violations_validate"] = self.violation_train
            row_dict["Avg_prob_validate"] = np.mean(self.prob_violation_train)
            row_dict.update(self.feasibility_row("validate"))
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
        row_dict.update(self.feasibility_row("validate"))
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

@dataclass
class _UniqueContexts:
    """One data split's contexts, deduplicated once for reuse across the rho sweep.

    grid() solves the CVXPYLayer at unique-context resolution -- once per distinct
    context rather than once per sample -- and then fans the answers back out. That
    is what puts tensors of four different lengths (unique/full x test/validate) in
    flight at the same time.
    """

    x_tch: list
    batch_int: int
    x_batch_array: list
    num_unique: int
    x_unique: list
    x_unique_array: list


@dataclass
class SplitTensors:
    """One split's decisions and uncertainty-set tensors, bound to the x they describe.

    Every field here is indexed by the same sample. They are bound together at the
    moment they are produced so that no consumer can be handed z from one split and
    A/b from another: within a single grid() iteration A/b exist at four different
    lengths, all with identical rank and dtype, and as loose variables they are
    indistinguishable. Passing this object instead of four tensors makes the
    mismatch unrepresentable rather than merely detectable.
    """

    x_batch: list
    z: list
    a_tch: torch.Tensor
    b_tch: torch.Tensor
    contextual: bool

    def __post_init__(self):
        if not self.contextual:
            # A and b are shared across samples here, so there is no batch to agree on.
            return
        batch_size = self.x_batch[0].shape[0]
        for name, tch in (("a_tch", self.a_tch), ("b_tch", self.b_tch)):
            if tch.shape[0] != batch_size:
                raise ValueError(
                    f"SplitTensors: {name} has batch dimension {tch.shape[0]}, but "
                    f"x_batch has {batch_size}. A contextual split's A and b must "
                    f"carry one entry per sample of that same split."
                )


class _ProbeStatus:
    """Where a radius sits relative to the feasible band a bisection is looking for."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


_OK = _ProbeStatus("ok")
_VIOLATING = _ProbeStatus("violating")
_UNSOLVED = _ProbeStatus("unsolved")


@dataclass
class _RhoProbe:
    """One radius evaluated during a bisection, and how it was classified."""

    rho: float
    row: "pd.Series"
    split: "SplitTensors"
    stats: "GridStats"
    status: _ProbeStatus


def _classify_probe(row, settings, violation_col, max_unsolved=0.0, target_eta=None):
    """Classify a probed radius as violating, unsolved, or acceptable.

    Solvability is checked FIRST and overrides the violation reading. Where the robust
    problem has no feasible decision, cvxpylayers returns a degenerate near-zero one
    that satisfies every constraint, so the violation probability reads ~0 and an
    unsolvable radius would otherwise score better than any real one.

    `max_unsolved` is the fraction of instances allowed to be unsolvable before a radius
    is rejected. It defaults to 0, i.e. any unsolvable instance disqualifies the radius,
    which is the rule this function has always applied.

    Raise it for instances whose unsolvability has a nonzero FLOOR rather than a cliff.
    Two ways that happens, both measured on `experiments/dcopf`: the context distribution
    has a tail beyond whatever quantile the instance was sized against (there, ~0.5% of
    contexts draw more load than the capacity margin covers, so a radius can be perfectly
    reasonable and still lose one sample in eighty); and a CONTEXTUAL set can be
    unsolvable at its own center, since as rho goes to 0 the set collapses to the
    predicted mean and a poor prediction is an infeasible point -- measured there,
    `Frac_unsolved` was at least 0.0125 at EVERY radius for the k-NN-centered baseline,
    including rho = 0.001. Against a floor like that, `max_unsolved = 0` rejects every
    radius including the good ones, and the search falls back to its least-violating
    probe or reports nothing acceptable at all.
    """
    unsolved = float(row.get("Frac_unsolved_validate", 0.0))
    if not np.isnan(unsolved) and unsolved > max_unsolved:
        return _UNSOLVED
    eta = settings.target_eta if target_eta is None else float(target_eta)
    if float(np.mean(row[violation_col])) > eta:
        return _VIOLATING
    return _OK


def _select_probe(probes, violation_col, objective_col, rho_lo, rho_hi):
    """The best acceptable radius, or the least-violating one if none was acceptable."""
    def objective(probe):
        value = float(probe.row[objective_col])
        return np.inf if np.isnan(value) else value

    acceptable = [probe for probe in probes if probe.status is _OK]
    if acceptable:
        # An objective column that is NaN at every acceptable radius carries no
        # ordering, and mapping NaN to inf would tie them all -- min() would then
        # return whichever happened to be probed FIRST, which for a bisection is an
        # endpoint of the bracket. That is how a multistage search came back with
        # rho=40 (violation 0.0) after having probed rho=1.34 at the 0.10 target:
        # silently the worst acceptable radius rather than the best.
        #
        # Fall back on monotonicity instead, which is the assumption the search is
        # built on anyway: violation falls and worst-case rises with rho, so among
        # radii that meet the target the SMALLEST is the best one. That is also
        # exactly what bisection()'s docstring promises it returns.
        if all(np.isnan(float(probe.row[objective_col])) for probe in acceptable):
            warnings.warn(
                f"{objective_col} is NaN at every acceptable radius, so it cannot "
                f"order them; selecting the smallest radius meeting target_eta "
                f"instead (valid because violation is monotone in rho). This is "
                f"expected in multistage, where the feasibility columns are not "
                f"populated.",
                stacklevel=3,
            )
            return min(acceptable, key=lambda probe: probe.rho)
        return min(acceptable, key=objective)

    solvable = [probe for probe in probes if probe.status is not _UNSOLVED]
    if not solvable:
        raise ValueError(
            f"the robust problem was unsolvable at every radius probed in "
            f"[{rho_lo}, {rho_hi}]; widen the bracket downwards"
        )
    fallback = min(solvable, key=lambda p: float(np.mean(p.row[violation_col])))
    warnings.warn(
        f"no radius in [{rho_lo}, {rho_hi}] met target_eta; falling back to the "
        f"least-violating one (rho={fallback.rho:.4g}, "
        f"{violation_col}={float(np.mean(fallback.row[violation_col])):.4g})",
        stacklevel=3,
    )
    return fallback


class _Stopwatch:
    """Accumulates the time spent inside its `with` blocks, which need not be adjacent.

    A rho evaluation interleaves the two splits' work, so each split's cost is a sum of
    disjoint segments rather than one contiguous span.
    """

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._entered = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed += time.time() - self._entered
        return False


@dataclass
class _MultistageSplit:
    """What a multistage radius hands back, standing in for _GridSplit.

    Only the three fields _finalize_rho_search and generate_test_row read: a rollout
    has no single per-context solve to carry.
    """

    a_tch: "torch.Tensor"
    b_tch: "torch.Tensor"
    x_batch: list


@dataclass
class _RhoSearchContext:
    """The state a rho search reuses across radii.

    Everything here is independent of rho, so grid() and bisection() build it once and
    hand it to _evaluate_rho for each radius they probe.
    """

    lam: "torch.Tensor"
    alpha: "torch.Tensor"
    a_tch_init: "torch.Tensor"
    b_tch_init: "torch.Tensor"
    test: "_UniqueContexts | None"
    validate: "_UniqueContexts | None"


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
        x = None
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
        self._x = x
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
        return self._z

    @property
    def context_values(self):
        return self._x

    @property
    def uncset_iters(self):
        return self._a_history, self._b_history

    @property
    def predictor(self):
        return self._predictor
