import abc
from abc import ABC
from enum import Enum

import numpy as np
import pandas as pd
import scipy as sc
import torch
from cvxpy import Parameter as OrigParameter
from cvxpy.expressions.variable import Variable
from cvxpylayers.torch import CvxpyLayer
from joblib import Parallel, delayed

import lropt.train.settings as settings
from lropt import RobustProblem
from lropt.train.parameter import ContextParameter, ShapeParameter, SizeParameter
from lropt.train.utils import (
    get_n_processes,
    halve_step_size,
    restore_step_size,
    take_step,
    undo_step,
)
from lropt.uncertain_parameter import UncertainParameter
from lropt.utils import unique_list
from lropt.violation_checker.settings import DEFAULT_MAX_ITER_LINE_SEARCH
from lropt.violation_checker.utils import CONSTRAINT_STATUS, InfeasibleConstraintException
from lropt.violation_checker.violation_checker import ViolationChecker

# add a simulator class. abstract class. user defines.
# simulate (dynamics)
# stage cost
# constraint (per stage or not) (constraint cvar, some notion of violation)


# Trainer
# inject the data into the trainer
# loss and constraints (calls monte carlo, stage cost, constraint)
# monte-carlo - evaluate without gradients



class Simulator(ABC):
    """Simulator class for the multi-stage problem. All parameters should be tensors."""

    @abc.abstractmethod
    def simulate(self,x,u,**kwargs):
        """Simulate next set of parameters using current parameters x
        and variables u, with added uncertainty
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stage_cost(self,x,u, **kwargs):
        """ Create the current stage cost using the current state x
        and decision u
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stage_cost_eval(self,x,u, **kwargs):
        """ Create the current stage evaluation cost using the current state x
        and decision u. This may differ from the stage cost, which is used
        for training.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def constraint_cost(self,x,u,alpha,**kwargs):
        """ Create the current constraint penalty cost
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def init_state(self,batch_size, seed, **kwargs):
        """ initialize the parameter value
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prob_constr_violation(self,x,u,**kwargs):
        """ calculate current probability of constraint violation
        """
        return torch.tensor(0,dtype = settings.DTYPE)



class Default_Simulator(ABC):
    def __init__(self,trainer):
        self.trainer = trainer

    def simulate(self,x,u,**kwargs):
        """Simulate next set of parameters using current parameters x
        and variables u, with added uncertainty
        """
        return x

    def stage_cost(self,x,u,**kwargs):
        """ Create the current stage cost using the current state x
        and decision u
        """
        return kwargs['trainer'].train_objective(kwargs['batch_int'], kwargs['eval_args'])


    def stage_cost_eval(self,x,u,**kwargs):
        """ Create the current stage evaluation cost using the current state x
        and decision u
        """
        return torch.tensor(kwargs['trainer'].evaluation_metric(
            kwargs['batch_int'], kwargs['eval_args'],
            kwargs['quantiles']),dtype=settings.DTYPE)


    def constraint_cost(self,x,u,alpha, **kwargs):
        """ Create the current constraint penalty cost
        """
        return kwargs['trainer'].train_constraint(kwargs['batch_int'],
                                                  kwargs['eval_args'],
                                                    alpha,
                                                    kwargs['slack'],
                                                    kwargs['eta'],
                                                    kwargs['kappa'])

    def init_state(self,batch_size, seed,**kwargs):
        """ initialize the parameter value
        """
        if kwargs['trainer']._eval_flag:
            return kwargs['trainer']._gen_batch(kwargs['trainer'].test_size,
                                                kwargs['trainer'].x_test_tch,
                                                kwargs['trainer'].u_test_set,
                                                1, kwargs["max_batch_size"])

        else:
            return kwargs['trainer']._gen_batch(kwargs['trainer'].train_size,
                                                kwargs['trainer'].x_train_tch,
                                                kwargs['trainer'].u_train_set,
                                                kwargs['batch_percentage'],
                                                kwargs["max_batch_size"])

    def prob_constr_violation(self,x,u,**kwargs):
        """ calculate current probability of constraint violation
        """
        return kwargs['trainer'].prob_constr_violation(kwargs['batch_int'],
                                                kwargs['eval_args'])

class Trainer():
    _EVAL_INPUT_CASE = Enum("_EVAL_INPUT_CASE", "MEAN EVALMEAN MAX")

    """Create a class to handle training"""
    def __init__(self, problem: RobustProblem):
        if not (self.count_unq_uncertain_param(problem) == 1):
            raise ValueError("Must have a single uncertain parameter " + \
                             "for training")
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

    def monte_carlo(self, rho_tch, alpha, solver_args, time_horizon,
                    a_tch, b_tch, batch_size = 1,
                    seed=None,contextual = False,
                    kwargs_simulator = None, model = None):
        """ This function calls the loss and constraint function, and returns
        an error if an infeasibility is encountered. This is infeasibility
        dependent on the testing data-set.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # remove for loop, set batch size to trials
        cost, constraint_cost, x_hist, z_hist, \
            constraint_status, eval_cost, prob_vio, \
                u_hist = self.loss_and_constraints(
            time_horizon=time_horizon, a_tch=a_tch, b_tch=b_tch,
            batch_size = batch_size,
            seed=seed,rho_tch = rho_tch, alpha = alpha,
            solver_args = solver_args,
            contextual = contextual, kwargs_simulator=kwargs_simulator,
            model = model)
        if constraint_status is CONSTRAINT_STATUS.INFEASIBLE:
            raise InfeasibleConstraintException(
                "Found an infeasible constraint during a call to monte_carlo."
                + "Possibly an infeasible uncertainty set initialization."
                + "Or infeasibility encountered in the testing set")
        return cost, constraint_cost, x_hist, z_hist, eval_cost, prob_vio, u_hist

    def loss_and_constraints(self, rho_tch, alpha, solver_args,
                             time_horizon, a_tch, b_tch,
                             batch_size = 1, seed=None,
                             contextual = False, kwargs_simulator = None,
                             model = None):
        """
        This function propagates the system state, calculates the costs,
        and checks feasibility
        Args:
            rho_tch
                size of the uncertainty set. A torch tensor.
            alpha
                cvar variable for the constraint cost.
            solver_args
                parameters for the solver: eg, tolerance, solver name
            time_horizon
                total time horizon for the multistage problem
            a_tch, b_tch
                the initialized size and shape parameters
            batch_size
                batch size for each time step
            seed
                seed to generate uncertain data
            contextual
                whether or not the learned set is contextual
            kwargs_simulator
                extra arguments for the simulator class functions
            model
                the model for the contextual uncertainty set
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
            eval_cost
                evaluation function cost, averaged across time steps
            prob_vio
                probability of constraint violation, averaged across time steps
            u_0
                uncertainty data for the single-stage case
        """
        if seed is not None:
            torch.manual_seed(seed)

        if self._multistage:
            u_0 = 0
            x_0 = self.simulator.init_state(batch_size, seed,**kwargs_simulator)
        else:
            batch_int, x_0, u_0 = self.simulator.init_state(batch_size, seed, **kwargs_simulator)
            kwargs_simulator['batch_int'] = batch_int

        if not isinstance(x_0,list):
            x_0 = [x_0]
        x_0 = [torch.tensor(x, dtype=settings.DTYPE) for x in x_0]
        if contextual:
            a_tch, b_tch = self.create_tensors_linear(x_0, model)

        cost = 0.0
        constraint_cost = 0.0
        if self._default_simulator:
            eval_cost = torch.tensor([0,0,0],dtype=settings.DTYPE)
        else:
            eval_cost = 0.0
        prob_vio = 0.0
        x_t = x_0
        x_hist = [[xval.detach().numpy().copy() for xval in x_t.copy()]]
        z_hist = []
        for t in range(time_horizon):
            z_t = self.cvxpylayer(rho_tch, *self.cp_param_tch,
                *x_t,a_tch,b_tch,solver_args=solver_args)
            constraints_status = self.violation_checker.check_constraints(z_batch=z_t,
                                        rho_mult_parameter=self._rho_mult_parameter,
                                        rho_tch=rho_tch, cp_parameters=self._cp_parameters,
                                        cp_param_tch=self.cp_param_tch,
                                        x_parameters=self._x_parameters, x_batch=x_t,
                                        shape_parameters=self._shape_parameters,
                                        shape_torches=[a_tch, b_tch])
            z_t = self._reduce_variables(z_t)
            if not self._multistage:
                eval_args = self.order_args(z_t,x_t,u_0)
                kwargs_simulator['eval_args'] = eval_args
            x_t = self.simulator.simulate(x_t, z_t,**kwargs_simulator)
            cost += self.simulator.stage_cost(x_t, z_t,**kwargs_simulator)
            eval_cost += self.simulator.stage_cost_eval(x_t, z_t,**kwargs_simulator)
            constraint_cost += self.simulator.constraint_cost(x_t, z_t, alpha,**kwargs_simulator)
            prob_vio += self.simulator.prob_constr_violation(x_t, z_t, **kwargs_simulator)
            x_hist.append([xval.detach().numpy().copy() for xval in x_t])
            z_hist.append(z_t)
            if contextual:
                a_tch, b_tch = self.create_tensors_linear(x_t,model)
            self._a_tch = a_tch
            self._b_tch = b_tch
        return cost, constraint_cost, x_hist, z_hist, constraints_status, eval_cost, prob_vio, u_0

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

    def count_unq_uncertain_param(self,problem) -> int:
        unc_params = self.uncertain_parameters(problem)
        return len(unique_list(unc_params))

    def uncertain_parameters(self,problem):
        """Find uncertain (u) parameters"""
        return [v for v in problem.parameters() if isinstance(v, UncertainParameter)]

    def x_parameters(self,problem):
        """Find context (x) parameters"""
        return [v for v in problem.parameters() if isinstance(v, ContextParameter)]

    def cp_params(self,problem):
        """Find cvxpy (noncontextual) parameters"""
        return [v for v in problem.parameters() if isinstance(v, OrigParameter)
                 and not (isinstance(v,ContextParameter) or
                           isinstance(v,UncertainParameter))]

    def rho_mult_param(self,problem):
        """Find the rho multiplier parameter"""
        return [v for v in problem.parameters() if isinstance(v, SizeParameter)]

    def gen_rho_mult_tch(self,rhoparams=[]):
        """Generate the torch of the rho multiplier value, placed in a list"""
        return [torch.tensor(rho.value,
                        dtype=settings.DTYPE,
                        requires_grad=self.train_flag) for rho in rhoparams][0]

    def shape_parameters(self, problem):
        """Get the reshaping parameters a and b"""
        return [v for v in problem.parameters() if isinstance(v, ShapeParameter)]

    def create_cvxpylayer(self,parameters = None, variables=None) -> CvxpyLayer:
        """Create cvxpylayers.
        Default parameter order: rho multiplier, cvxpy parameters, context parameters, a, b.
        Default variable order: the variables of problem_canon """
        if parameters is None:
            new_parameters = self._rho_mult_parameter + self._cp_parameters +\
                  self._x_parameters + self._shape_parameters
        else:
            assert isinstance(parameters, list)
            self._x_tchs_init = False
            self._x_parameters = parameters
            new_parameters = self._rho_mult_parameter + parameters + self._shape_parameters
        if variables is None:
            #TODO: Clean up if works properly
            # variables = self.problem_no_unc.variables()
            variables = self.problem_canon.variables()
        cvxpylayer = CvxpyLayer(self.problem_no_unc, parameters=new_parameters,
                                variables=self.problem_no_unc.variables())
        self._reduced_variables = variables
        return cvxpylayer

    def x_parameter_shapes(self, x_params):
        """Get the size of all y parameters"""
        return [v.size for v in x_params]

    def initialize_dimensions_linear(self):
        """Find the dimensions of the linear model"""
        x_endind = self.x_endind
        x_shapes = self.x_parameter_shapes(self._x_parameters)
        a_shape = self.unc_set._a.shape
        b_shape = self.unc_set._b.shape
        if x_endind:
            in_shape = x_endind
        else:
            in_shape = sum(x_shapes)
        out_shape = int(a_shape[0]*a_shape[1]+ b_shape[0])
        return in_shape, out_shape, a_shape[0]*a_shape[1]

    def create_tensors_linear(self,x_batch, linear):
        """Create the tensors of a's and b's using the trained linear model"""
        x_endind = self.x_endind
        a_shape = self.unc_set._a.shape
        b_shape = self.unc_set._b.shape
        x_batch = [torch.flatten(x, start_dim = 1) for x in x_batch]
        if x_endind:
            input_tensors = torch.hstack(x_batch)[:,:x_endind]
        else:
            input_tensors = torch.hstack(x_batch)
        theta = linear(input_tensors)
        raw_a = theta[:,:a_shape[0]*a_shape[1]]
        raw_b = theta[:,a_shape[0]*a_shape[1]:]
        a_tch = raw_a.view(theta.shape[0],a_shape[0],a_shape[1])
        b_tch = raw_b.view(theta.shape[0],b_shape[0])
        if not self.train_flag:
            a_tch = torch.tensor(a_tch, requires_grad=False)
            b_tch = torch.tensor(b_tch, requires_grad=False)
        return a_tch, b_tch

    def init_linear_model(self, a_tch, b_tch, random_init = False,
                    init_num=1,seed = 0, init_weight = None, init_bias = None):
        """Initializes the linear model weights and bias"""
        in_shape, out_shape, a_totsize = \
                                self.initialize_dimensions_linear()
        torch.manual_seed(seed+init_num)
        lin_model = torch.nn.Linear(in_features = in_shape,
                                out_features = out_shape).double()
        lin_model.bias.data[a_totsize:] = b_tch
        if not random_init:
            with torch.no_grad():
                torch_b = b_tch
                torch_a = a_tch.flatten()
                torch_concat = torch.hstack([torch_a, torch_b])
            lin_model.weight.data.fill_(0.000)
            lin_model.bias.data = torch_concat
            if init_weight is not None:
                lin_model.weight.data = torch.tensor(init_weight,
                                    dtype=torch.double,requires_grad=True)
            if init_bias is not None:
                lin_model.bias.data = torch.tensor(
                        init_bias, dtype=torch.double,
                                    requires_grad=True)
        return lin_model

    def init_model(self,linear):
        return torch.nn.Sequential(linear)

    def create_cp_param_tch(self,num):
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
            param_tch = torch.tensor(param.value,
                    dtype=settings.DTYPE,
                    requires_grad=self.train_flag)
            if num == 0:
                cp_param_tchs.append(param_tch)
                continue
            param_tch_dim = param_tch.dim()
            if param_tch_dim == 0:
                shape = (num)
            elif param_tch_dim == 1:
                shape = (num,1)
            elif param_tch_dim == 2:
                shape = (num, 1,1)
            else:
                raise ValueError("Maximum dimension of parameters is 2")
            cp_param_tchs.append(param_tch.repeat(shape))
        return cp_param_tchs

    def _split_dataset(self, test_percentage=settings.BATCH_PERCENTAGE_DEFAULT, seed=0):
        """
        This function splits the uncertainty set into train and test sets
            and also creates torch tensors

        Args:
        test_percentage
            Fraction of data to place into the test set
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
        num_test = max(1, int(self.unc_set.data.shape[0]*test_percentage))
        num_train = int(self.unc_set.data.shape[0] - num_test)
        test_indices = np.random.choice(self.unc_set.data.shape[0],
                                      num_test, replace=False)
        train_indices = [i for i in range(self.unc_set.data.shape[0]) if i not in test_indices]

        unc_train_set = np.array([self.unc_set.data[i] for i in train_indices])
        unc_test_set = np.array([self.unc_set.data[i] for i in test_indices])
        unc_train_tch = torch.tensor(
            self.unc_set.data[train_indices], requires_grad=self.train_flag, dtype=settings.DTYPE)
        unc_test_tch = torch.tensor(
            self.unc_set.data[test_indices], requires_grad=self.train_flag, dtype=settings.DTYPE)

        cp_param_tchs = []
        x_train_tchs = []
        x_test_tchs = []

        if self._x_tchs_init:
            cp_param_tchs = self.create_cp_param_tch(0)
            for i in range(len(self._x_parameters)):
                x_train_tchs.append(torch.tensor(
                    self._x_parameters[i].data[train_indices], requires_grad=self.train_flag,
                    dtype=settings.DTYPE))
                x_test_tchs.append(torch.tensor(
                    self._x_parameters[i].data[test_indices], requires_grad=self.train_flag,
                    dtype=settings.DTYPE))



        self.u_train_tch = unc_train_tch
        self.u_test_tch = unc_test_tch
        self.u_train_set = unc_train_set
        self.u_test_set = unc_test_set
        self.x_train_tch = x_train_tchs
        self.x_test_tch = x_test_tchs
        self.train_size = num_train
        self.test_size = num_test
        self.cp_param_tch = cp_param_tchs

        return unc_train_set, unc_test_set, unc_train_tch, \
                unc_test_tch, x_train_tchs, x_test_tchs, cp_param_tchs

    def _gen_batch(self, num_xs,x_data, u_data, batch_percentage, max_size=10000, min_size=1):
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

        batch_int = max(min(int(num_xs*batch_percentage),max_size),min_size)
        random_int = np.random.choice(
            num_xs, batch_int, replace=False)
        x_tchs = []
        for i in range(len(x_data)):
            x_tchs.append(x_data[i].data[random_int])

        u_tch = torch.tensor(u_data[random_int], requires_grad=self.train_flag,
                                dtype=settings.DTYPE)

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
        rho_tch = torch.tensor(
            scalar, requires_grad=self.train_flag, dtype=settings.DTYPE)

        return rho_tch

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

        cov_len_cond = (len(np.shape(np.cov(train_set.T))) >= 1)
        if (init_A is None):
            if cov_len_cond:
                return sc.linalg.sqrtm(np.cov(train_set.T))
            return np.array([[np.cov(train_set.T)]])

        mat_shape = train_set.shape[1] if cov_len_cond else 1
        matrix = np.array(init_A) if (
            init_A is not None) else np.eye(mat_shape)
        return matrix

    def _init_torches(self, init_A, init_b, init_alpha, train_set):
        """
        This function Initializes and returns a_tch, b_tch, and alpha as tensors.
        It also initializes alpha and the slack variables as 0
        """
        # train_set = train_set.detach().numpy()
        self._init = self._gen_init(train_set, init_A)
        init_tensor = torch.tensor(self._init, requires_grad=self.train_flag, dtype=settings.DTYPE)
        b_tch = None

        if init_b is not None:
            b_tch_data = np.array(init_b)
        else:
            b_tch_data = np.mean(train_set, axis=0)
        b_tch = torch.tensor(b_tch_data, requires_grad=self.train_flag, dtype=settings.DTYPE)
        a_tch = init_tensor

        alpha = torch.tensor(init_alpha, requires_grad=self.train_flag)
        slack = torch.zeros(self.num_g_total, requires_grad=self.train_flag, dtype=settings.DTYPE)
        return a_tch, b_tch, alpha, slack

    def _update_iters(self, save_history, a_history, b_history,
                      rho_history,a_tch, b_tch, rho_tch):
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

    def _set_train_variables(self, fixb, alpha, slack, a_tch, b_tch,
                              rho_tch, train_size,contextual, model):
        """
        This function sets the variables to be trained in the outer level problem.

        Args:

        fixb
            Whether to hold b constant or not when training
        alpha
            Torch tensor of alpha for CVaR
        slack
            Torch tensor of the slack for CVaR
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
            variables = [rho_tch, alpha, slack]
        elif fixb:
            variables = [rho_tch, a_tch, alpha, slack]
        elif contextual:
            variables = [rho_tch, alpha,slack]
            variables.extend(list(model.parameters()))
        else:
            variables = [rho_tch, a_tch, b_tch, alpha, slack]

        return variables

    def _calc_coverage(self, dset, a_tch, b_tch, rho=1,p=2,
                       contextual=False,y_set = None, linear = None):
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
            a_tch, b_tch = self.create_tensors_linear([torch.tensor(y) for y in y_set[-1]], linear)
            for i in range(dset.shape[0]):
                coverage += torch.where(
                    torch.norm((a_tch[i].T@torch.linalg.inv(a_tch[i]@a_tch[i].T)) @ (dset[i]-
                            b_tch[i]),p)
                    <= rho,
                    1,
                    0,
                )
        else:
            for datind in range(dset.shape[0]):
                coverage += torch.where(
                    torch.norm((a_tch.T@torch.linalg.inv(a_tch@a_tch.T)) @ (dset[datind]-
                            b_tch),p)
                    <= rho,
                    1,
                    0,
                )
        return (coverage/dset.shape[0]).detach().numpy().item()

    def order_args(self, z_batch, x_batch, u_batch):
        """
        This function orders z_batch (decisions), x_batch (context), and
        u_batch (uncertainty) according to the order in vars_params.
        """
        problem = self.problem_canon
        args = []
        # self.vars_params is a dictionary, hence unsorted. Need to iterate over it in order
        ind_dict = {
            Variable: 0,
            ContextParameter: 0,
            UncertainParameter: 0,
        }
        args_dict = {
            Variable: z_batch,
            ContextParameter: x_batch,
            UncertainParameter: u_batch,
        }

        for i in range(len(problem.vars_params)):
            curr_type = type(problem.vars_params[i])
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

        return args

    #BATCHED
    def _eval_input(self, batch_int,eval_func, eval_args, init_val,
                    eval_input_case, quantiles, serial_flag=False, **kwargs):
        """
        This function takes decision variables, y's, and u's,
            evaluates them and averages them on a given function.

        Args:
            batch_int:
                The number of samples in the batch, to take the mean over
            eval_func:
                The function used for evaluation.
            eval_args:
                The arguments for eval_func
            init_val:
                The placeholder for the returned values
            eval_input_case:
                The type of evaluation performed. Can be MEAN, EVALMEAN, or MAX
            quantiles:
                The quantiles for mean values. can be None.
            serial_flag:
                Whether or not to evalute the function in serial
            kwargs:
                Additional arguments for the eval_func

        Returns:
            The average among all evaluated J x N pairs
        """
        def _serial_eval(batch_int, eval_args, init_val=None, **kwargs):
            """
            This is a helper function that calls eval_func in a serial way.
            """
            def _sample_args(eval_args, sample_ind):
                """
                This is a helper function that samples arguments to be passed to eval_func.
                """
                res = []
                for eval_arg in eval_args:
                    curr_arg = eval_arg[sample_ind]
                    res.append(curr_arg)
                return res
            curr_result = {}
            for j in range(batch_int):
                curr_eval_args = _sample_args(eval_args, j)
                if init_val:
                    init_val[:,j] = eval_func(*curr_eval_args, **kwargs)
                else:
                    curr_result[j] = eval_func(*curr_eval_args, **kwargs)
            return curr_result

        if eval_input_case != Trainer._EVAL_INPUT_CASE.MAX:
            if serial_flag:
                curr_result = _serial_eval(batch_int, eval_args, **kwargs)
            else:
                curr_result = eval_func(*eval_args, **kwargs)
        if eval_input_case == Trainer._EVAL_INPUT_CASE.MEAN:
            if serial_flag:
                init_val = torch.vstack([curr_result[v] for v in curr_result])
            else:
                init_val = curr_result
            init_val = torch.mean(init_val,axis=0)
        elif eval_input_case == Trainer._EVAL_INPUT_CASE.EVALMEAN:
            if serial_flag:
                init_val = torch.vstack([curr_result[v] for v in curr_result])
            else:
                init_val = curr_result
            bot_q, top_q = quantiles
            init_val_lower = torch.quantile(init_val, bot_q, axis=0)
            init_val_mean = torch.mean(init_val,axis=0)
            init_val_upper = torch.quantile(init_val, top_q,axis=0)
            return (init_val_lower, init_val_mean, init_val_upper)
        elif eval_input_case == Trainer._EVAL_INPUT_CASE.MAX:
            # We want to see if there's a violation: either 1 from previous iterations,
            # or new positive value from now
            if serial_flag:
                _ = _serial_eval(batch_int, eval_args, init_val, **kwargs)
            else:
                init_val = eval_func(*eval_args, **kwargs)
                if len(init_val.shape) > 1:
                    init_val = init_val.T
            init_val = (init_val > settings.TOLERANCE_DEFAULT).float()
        return init_val

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
        return self._eval_input(batch_int,eval_func=self.f, eval_args=eval_args, init_val=0,
                                eval_input_case=Trainer._EVAL_INPUT_CASE.MEAN, quantiles=None)

    def train_constraint(self, batch_int,eval_args, alpha, slack, eta, kappa):
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
            slack:
                The slack of the CVaR constraint to become equality
            kappa:
                The target CVaR threshold
        Returns:
            The average among all evaluated J x N pairs
        """
        H = torch.zeros(self.num_g_total, dtype=settings.DTYPE)
        for k, h_k in enumerate(self.h):
            init_val = self._eval_input(batch_int,h_k, eval_args, 0,
                                        Trainer._EVAL_INPUT_CASE.MEAN, quantiles=None,
                                        alpha=alpha, eta=eta)
            h_k_expectation = init_val + alpha - kappa + \
                slack[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])]
            H[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])] \
                = h_k_expectation
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
        if (self.eval is None):
            return 0

        return self._eval_input(batch_int,eval_func=self.eval, eval_args=eval_args,init_val=0,
                                eval_input_case=Trainer._EVAL_INPUT_CASE.EVALMEAN,
                                quantiles=quantiles, serial_flag=True)

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
        G = torch.zeros((self.num_g_total, batch_int), dtype=settings.DTYPE)
        for k, g_k in enumerate(self.g):
            G[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])] = \
            self._eval_input(batch_int, eval_func=g_k, eval_args=eval_args, init_val=\
                        G[sum(self.g_shapes[:k]):sum(self.g_shapes[:(k+1)])],
                        eval_input_case=Trainer._EVAL_INPUT_CASE.MAX, quantiles=None)
        return G.mean(axis=1)

    def lagrangian(self, batch_int,eval_args, alpha, slack, lam, mu,
                   eta=settings.ETA_LAGRANGIAN_DEFAULT, kappa=settings.KAPPA_LAGRANGIAN_DEFAULT):
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
        F = self.train_objective(batch_int,eval_args=eval_args)
        H = self.train_constraint(batch_int,eval_args=eval_args,
                                  alpha=alpha, slack=slack, eta=eta, kappa=kappa)
        return F + lam @ H + (mu/2)*(torch.linalg.norm(H)**2), H.detach()

    def _reduce_variables(self, z_batch: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        This helper function reduces z_batch whose len is len(self.problem_no_unc.variables())
        to len(_reduced_variables) (default: len(self.problem_canon.variables())).
        It returns the reduced list of tensors, where only the tensors that correspond to variables
        of self.problem_canon.variables() are preserved.
        """
        res = [None]*len(self._reduced_variables)
        for problem_no_unc_ind, problem_no_unc_var in enumerate(self.problem_no_unc.variables()):
            for problem_canon_ind, problem_canon_var in enumerate(self._reduced_variables):
                if problem_no_unc_var.id==problem_canon_var.id:
                    res[problem_canon_ind] = z_batch[problem_no_unc_ind]
                    break
        return res

    def _train_loop(self, init_num, **kwargs):
        if kwargs['random_init'] and kwargs['train_shape']:
            if init_num >= 1:
                np.random.seed(kwargs['seed']+init_num)
                shape = self.unc_set._a.shape
                kwargs['init_A'] = np.random.rand(shape[0],shape[1])
                    #  + 0.01*np.eye(kwargs['u_size'])
                kwargs['init_b'] = np.mean(self.u_train_set, axis=0)
        a_history = []
        b_history = []
        rho_history = []
        df = pd.DataFrame(columns=["step"])
        df_test = pd.DataFrame(columns=["step"])

        rho_tch = self._gen_rho_tch(kwargs['init_rho'])
        a_tch, b_tch, alpha, slack = self._init_torches(kwargs['init_A'],
                                        kwargs['init_b'],
                                        kwargs['init_alpha'], self.u_train_set)

        self._update_iters(kwargs['save_history'], a_history,
                           b_history, rho_history,a_tch, b_tch, rho_tch)

        if kwargs["contextual"]:
            if kwargs["linear"] is None:
                kwargs['linear'] = self.init_linear_model(a_tch, b_tch,
                                    kwargs['random_init'],init_num,
                                    kwargs['seed'],kwargs['init_weight'],
                                    kwargs['init_bias'])
            kwargs['model'] = torch.nn.Sequential(kwargs['linear'])
        else:
            kwargs['model'] = None
            kwargs['linear'] = None

        variables = self._set_train_variables(kwargs['fixb'], alpha,
                                              slack, a_tch, b_tch,rho_tch,kwargs["trained_shape"],
                                            kwargs["contextual"], kwargs['linear'])
        if kwargs['optimizer'] == "SGD":
            opt = settings.OPTIMIZERS[kwargs['optimizer']](
                variables, lr=kwargs['lr'], momentum=kwargs['momentum'])
        else:
            opt = settings.OPTIMIZERS[kwargs['optimizer']](
                variables, lr=kwargs['lr'])

        if kwargs['scheduler']:
            scheduler_ = torch.optim.lr_scheduler.StepLR(opt, step_size=kwargs['lr_step_size'],
                                                        gamma=kwargs['lr_gamma'])
        else:
            scheduler_ = None
        # if kwargs['scheduler']:
        #     scheduler_ = torch.optim.lr_scheduler.StepLR(
        #         opt, step_size=kwargs['lr_step_size'], gamma=kwargs['lr_gamma'])
        # y's and cvxpylayer begin
        lam = kwargs['init_lam'] * torch.ones(self.num_g_total, dtype=settings.DTYPE)
        mu = kwargs['init_mu']
        curr_cvar = np.inf
        if self._default_simulator:
            kwargs['kwargs_simulator'] = {'trainer':self, 'slack':slack,
                            'eta':kwargs['eta'],'kappa':kwargs['kappa'],
                            'quantiles': kwargs['quantiles'],
                              'max_batch_size':kwargs['max_batch_size'],
                                'batch_percentage': kwargs['batch_percentage']}
        for step_num in range(kwargs['num_iter']):
            if step_num>0:
                take_step(opt=opt, slack=slack, rho_tch=rho_tch, scheduler=scheduler_)
            train_stats = TrainLoopStats(
                step_num=step_num, train_flag=self.train_flag, num_g_total=self.num_g_total)

            torch.manual_seed(kwargs['seed'] + step_num)
            #In the first epoch we try only once
            current_iter_line_search = 1 if step_num==0 else self._max_iter_line_search+1
            for violation_counter in range(current_iter_line_search):
                self._eval_flag = False
                cost, constr_cost, _, _, constraint_status,eval_cost,\
                    prob_violation_train, _ = self.loss_and_constraints(
                    time_horizon=kwargs['time_horizon'],
                      a_tch=a_tch, b_tch=b_tch, batch_size=kwargs['batch_size'],
                    seed=kwargs['seed']+step_num+1,rho_tch=rho_tch,
                    alpha = alpha,
                    solver_args=kwargs['solver_args'],
                    contextual = kwargs['contextual'],
                    kwargs_simulator = kwargs["kwargs_simulator"],
                      model = kwargs['model'])

                if self.num_g_total > 1:
                    fin_cost = cost + lam @ constr_cost +\
                            (mu/2)*(torch.linalg.norm(constr_cost)**2)
                else:
                    fin_cost = cost+ lam*constr_cost +\
                    (mu/2)*(constr_cost**2)
                fin_cost.backward()
                if constraint_status is CONSTRAINT_STATUS.FEASIBLE:
                    restore_step_size(opt, num_steps=violation_counter)
                    break
                elif constraint_status is CONSTRAINT_STATUS.INFEASIBLE:
                    undo_step(opt=opt)
                    halve_step_size(opt=opt)

            if constraint_status is CONSTRAINT_STATUS.INFEASIBLE:
                if step_num==0:
                    exception_message = "Infeasible uncertainty set initialization"
                else:
                    exception_message = "Violation constraint check timed "
                    + "out after " + f"{DEFAULT_MAX_ITER_LINE_SEARCH} attempts."
                raise InfeasibleConstraintException(exception_message)
            if not self._default_simulator:
                eval_cost = eval_cost.repeat(3)
            train_stats.update_train_stats(
                fin_cost.detach().numpy().copy(),
                eval_cost,prob_violation_train, constr_cost.detach())

            if step_num % kwargs['aug_lag_update_interval'] == 0:
                if torch.norm(constr_cost.detach()) <= \
                    kwargs['lambda_update_threshold']*curr_cvar:
                    curr_cvar= torch.norm(constr_cost.detach())
                    lam += torch.minimum(mu*constr_cost.detach(), kwargs['lambda_update_max']*\
                                            torch.ones(self.num_g_total,dtype=settings.DTYPE))
                else:
                    mu = kwargs['mu_multiplier']*mu

            new_row = train_stats.generate_train_row(self._a_tch, rho_tch, lam,
                                        mu, alpha, slack,
                                        kwargs["contextual"], kwargs['linear'])
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

            self._update_iters(kwargs['save_history'], a_history,
                               b_history, rho_history, self._a_tch, self._b_tch, rho_tch)

            if step_num % kwargs['test_frequency'] == 0:
                self._eval_flag = True
                val_cost, val_cost_constr, x_batch, z_batch, eval_cost,\
                      prob_constr_violation, u_batch = self.monte_carlo(
                        time_horizon=kwargs['time_horizon'],
                        a_tch=a_tch, b_tch=b_tch,
                        batch_size = kwargs['test_batch_size'],
                        seed=kwargs['seed'], rho_tch = rho_tch, alpha = alpha,
                        solver_args = kwargs['solver_args'],
                        contextual = kwargs['contextual'],
                        kwargs_simulator = kwargs['kwargs_simulator'],
                          model = kwargs['model'] )
                record_eval_cost = eval_cost
                if not self._default_simulator:
                    record_eval_cost = eval_cost.repeat(3)
                print("iteration %d, valid %.4e, vio %.4e" % (step_num,
                                                record_eval_cost[1].item(),
                                                val_cost_constr.mean().item()) )

                train_stats.update_test_stats(
                    record_eval_cost, prob_constr_violation, constr_cost.detach())
                new_row = train_stats.generate_test_row(
                    self._calc_coverage, self._a_tch, self._b_tch, alpha,
                    u_batch, rho_tch, self.unc_set, z_batch,
                    kwargs['contextual'], kwargs['linear'], x_batch)
                df_test = pd.concat([df_test,
                                     new_row.to_frame().T], ignore_index=True)

        if constr_cost.detach().numpy().sum()/self.num_g_total \
            <= kwargs["kappa"]:
            fin_val = record_eval_cost[1].item()
        else:
            fin_val = record_eval_cost[1].item() + 10*abs(constr_cost.detach().numpy().sum())
        a_val = self._a_tch.detach().numpy().copy()
        b_val = self._b_tch.detach().numpy().copy()
        rho_val = rho_tch.detach().numpy().copy() if kwargs['trained_shape'] else 1
        param_vals = (a_val, b_val, rho_val, record_eval_cost[1].item())
        # tqdm.write("Testing objective: {}".format(obj_test[1].item()))
        # tqdm.write("Probability of constraint violation: {}".format(
        #            prob_violation_test))
        return df, df_test, a_history, b_history, rho_history, \
            param_vals, fin_val, z_batch, mu, kwargs["linear"]

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
        init_rho=settings.INIT_RHO_DEFAULT,
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
        max_batch_size = settings.MAX_BATCH_SIZE,
        contextual = settings.CONTEXTUAL_DEFAULT,
        linear = settings.CONTEXTUAL_LINEAR_DEFAULT,
        init_weight = settings.CONTEXTUAL_WEIGHT_DEFAULT,
        init_bias = settings.CONTEXTUAL_BIAS_DEFAULT,
        x_endind = settings.X_ENDIND_DEFAULT,
        max_iter_line_search = DEFAULT_MAX_ITER_LINE_SEARCH,
        policy = settings.POLICY_DEFAULT,
        time_horizon = settings.TIME_HORIZON_DEFAULT,
        batch_size = settings.BATCH_SIZE_DEFAULT,
        test_batch_size = settings.TEST_BATCH_SIZE_DEFAULT,
        simulator = settings.SIMULATOR_DEFAULT,
        kwargs_simulator = settings.KWARGS_SIM_DEFAULT,
        multistage = settings.MULTISTAGE_DEFAULT
    ):
        r"""
        Trains the uncertainty set parameters to find optimal set
        w.r.t. augmented lagrangian metric

        Parameters TODO (Amit): Irina - update all the variables
        -----------
        train_size : bool, optional
           If True, train only rho
        train_shape: bool, optional
            If True, train both the shape A, b, and size rhos
        fixb : bool, optional
            If True, do not train b
        num_iter : int, optional
            The total number of gradient steps performed.
        num_iter_size : int, optional
            The total number of gradient steps performed for training
            only rho
        lr : float, optional
            The learning rate of gradient descent.
        lr_size : float, optional
            The learning rate of gradient descent for training only rho
        momentum: float between 0 and 1, optional
            The momentum for gradient descent.
        optimizer: str or letters, optional
            The optimizer to use tor the descent algorithm.
        init_rho : float, optional
            The rho (radius) to initialize :math:`A` and :math:`b`, if passed.
        init_A : numpy array, optional
            Initialization for the reshaping matrix, if passed.
            If not passed, :math:`A` will be initialized as the
            inverse square root of the
            covariance of the data.
        init_b : numpy array, optional
            Initialization for the relocation vector, if passed.
            If not passed, b will be initialized as :math:`\bar{d}`.
        save_history: bool, optional
            Whether or not to save the A and b over the training iterations
        init_alpha : float, optional
            The initial alpha value for the CVaR constraint in the outer
            level problem.
        eta: float, optional
            The eta value for the CVaR constraint
        init_lam : float, optional
            The initial lambda value for the outer level lagrangian function.
        init_mu : float, optional
            The initial mu value for the outer level lagrangian function.
        mu_multiplier : float, optional
            The initial mu multiplier for the outer level lagrangian function.
        kappa : float, optional
            The target threshold for the outer level CVaR constraint.
        random_int : bool, optional
            Whether or not to initialize the set with random values
        num_random_int : int, optional
            The number of random initializations performed if random_int is True
        test_frequency : int, optional
            The number of iterations before testing results are recorded
        test_percentage : float, optional
            The percentage of data to use in the testing set.
        seed : int, optional
            The seed to control the random state of the train-test data split.
        batch_percentage : float, optional
            The percentage of data to use in each training step.
        solver_args:
            The optional arguments passed to the solver
        parallel : bool, optional
            Whether or not to parallelize the training loops
        position: bool, optional
            The position of the tqdm statements for the training loops
        scheduler: bool, optional
            Whether or not the learning rate is decreased over steps
        lr_step_size: int, optional
            The number of iterations before the learning rate is decreased,
            if scheduler is enabled
        lr_gamma: float, optional
            The multiplier of the lr if the scheduler is enabled
        quantiles: tuple, optional
            The lower and upper quantiles of the test values desired
        aug_lag_update_interval: int, optional
            The number of iterations before the augmented lagrangian parameters
            (lambda, mu) are updated
        lambda_update_threshold: float, optional
            The threshold of CVaR improvement, between 0 and 1, where an update
            to lambda is accepted. Otherwise, mu is updated.
        lambda_update_max: float, optional
            The maximum allowed lambda value
        max_batch_size: int, optional
            The maximum data batch size allowed for each iteration
        contextual: bool, optional
            Whether or not the learned set is contextual
        linear: NN model, optional
            The linear NN model to use
        init_weight: np.array, optional
            The initial weight of the NN model
        init_bias: np.array, optional
            The initial bias of the NN model
        n_jobs:
            The number of parallel processes

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
        self.train_flag = True
        if contextual and not train_shape:
            if linear is None:
                raise ValueError("You must give a model if you do not train a model")
        _,_,_,_,_,_,_ = self._split_dataset(test_percentage, seed)
        self._multistage = multistage
        if self._multistage:
            self.num_g_total = 1
        else:
            assert (time_horizon == 1)
        self.cvxpylayer = self.create_cvxpylayer() if not policy else policy
        if simulator:
            self.simulator = simulator
            self._default_simulator = False
        else:
            self.simulator = Default_Simulator(self)
            self._default_simulator = True
        self.violation_checker = ViolationChecker(self.cvxpylayer, self.problem_no_unc.constraints)
        self._max_iter_line_search = max_iter_line_search
        self.x_endind = x_endind

        num_random_init = num_random_init if random_init else 1
        num_random_init = num_random_init if train_shape else 1
        kwargs = {"train_size": train_size,
                   "trained_shape": not train_shape, "train_shape": train_shape,
                  "init_A": init_A, "init_b": init_b,
                  "init_rho": init_rho,
                  "random_init": random_init,
                  "seed": seed,
                  "init_alpha": init_alpha, "save_history": save_history,
                  "fixb": fixb, "optimizer": optimizer,
                  "lr": lr, "momentum": momentum,
                  "scheduler": scheduler, "init_lam":
                  init_lam, "init_mu": init_mu,
                  "num_iter": num_iter,
                  "batch_percentage": batch_percentage,
                  "solver_args": solver_args,
                  "kappa": kappa, "test_frequency": test_frequency,
                  "mu_multiplier": mu_multiplier,
                  "quantiles": quantiles, "lr_step_size": lr_step_size,
                  "lr_gamma": lr_gamma, "eta": eta,
                  "position": position, "test_percentage": test_percentage,
                  "aug_lag_update_interval": aug_lag_update_interval,
                  "lambda_update_threshold":lambda_update_threshold,
                  "lambda_update_max":lambda_update_max,
                  "max_batch_size":max_batch_size,"contextual":contextual,
                  "linear": linear,'init_weight':init_weight,
                  'init_bias':init_bias, 'batch_size': batch_size,
                  'test_batch_size':test_batch_size,
                  'time_horizon': time_horizon,
                  'kwargs_simulator': kwargs_simulator}

        # Debugging code - one iteration
        # res = self._train_loop(0, **kwargs)
        # Debugging code - serial
        if not parallel:
            res = []
            for init_num in range(num_random_init):
                res.append(self._train_loop(init_num, **kwargs))
        # n_jobs = get_n_processes() if parallel else 1
        # pool_obj = Pool(processes=n_jobs)
        # loop_fn = partial(self._train_loop, **kwargs)
        # res = pool_obj.map(loop_fn, range(num_random_init))
        # Joblib version
        else:
            n_jobs = get_n_processes() if parallel else 1
            res = Parallel(n_jobs=n_jobs)(delayed(self._train_loop)(
                init_num, **kwargs) for init_num in range(num_random_init))
        df, df_test, a_history, b_history, rho_history, param_vals, \
            fin_val, var_values, mu_val, linear_models = zip(*res)
        index_chosen = np.argmin(np.array(fin_val))
        self.orig_problem_trained = True
        self.unc_set._trained = True
        return_rho = param_vals[index_chosen][2]
        if contextual:
            self.unc_set.a.value = param_vals[index_chosen][0][0]
            self.unc_set.b.value = param_vals[index_chosen][1][0]
        else:
            self.unc_set.a.value = param_vals[index_chosen][0]
            self.unc_set.b.value = param_vals[index_chosen][1]

        if train_shape and train_size:
            kwargs["init_rho"] = return_rho
            kwargs["trained_shape"] = True
            kwargs["train_shape"] = False
            kwargs["init_A"] = self.unc_set.a.value
            kwargs["init_b"] = self.unc_set.b.value
            # kwargs["init_rho"] = 1
            kwargs["random_init"] = False
            kwargs["lr"] = lr_size if lr_size else lr
            kwargs["num_iter"] = num_iter_size if num_iter_size else num_iter
            kwargs["init_mu"] = mu_val[index_chosen]
            if not parallel:
                res = []
                for init_num in range(1):
                    res.append(self._train_loop(init_num, **kwargs))
            else:
                res = Parallel(n_jobs=n_jobs)(delayed(self._train_loop)(
                init_num, **kwargs) for init_num in range(1))
            df_s, df_test_s, a_history_s, b_history_s,rho_history_s,\
            param_vals_s, fin_val_s, var_values_s, mu_s, \
                linear_models_s = zip(*res)
            return_rho = param_vals_s[0][2]
            return_df = pd.concat([df[index_chosen],df_s[0]])
            return_df_test = pd.concat([df_test[index_chosen], df_test_s[0]])
            return_a_history = a_history[index_chosen] + a_history_s[0]
            return_b_history = b_history[index_chosen] + b_history_s[0]
            return_rho_history = rho_history[index_chosen] + rho_history_s[0]
            return Result(self, self.problem_canon, return_df,
                      return_df_test, self.unc_set.a.value,
                      self.unc_set.b.value,
                      return_rho, param_vals[0][3],
                      var_values[index_chosen] + var_values_s[0],
                      a_history=return_a_history,
                      b_history=return_b_history,
                      rho_history = return_rho_history,
                      linear = linear_models_s[0])
        return Result(self, self.problem_canon, df[index_chosen],
                      df_test[index_chosen], self.unc_set.a.value,
                      self.unc_set.b.value,
                      return_rho, param_vals[index_chosen][3],
                      var_values[index_chosen],
                      a_history=a_history[index_chosen],
                      b_history=b_history[index_chosen],
                      rho_history = rho_history[index_chosen],
                      linear = linear_models[index_chosen])

    def gen_unique_x(self,y_batch):
        """ get unique y's from a list of y parameters. """
        y_batch_array = [np.array(ele) for ele in y_batch]
        all_indices = [np.unique(ele,axis=0, return_index=True)[1] for ele in y_batch_array]
        unique_indices = np.unique(np.concatenate(all_indices))
        num_unique_indices = len(unique_indices)
        y_unique = [torch.tensor(ele, dtype=settings.DTYPE)[unique_indices] \
                    for ele in y_batch_array]
        y_unique_array = [ele[unique_indices] for ele in y_batch_array]
        return y_batch_array, num_unique_indices, y_unique, y_unique_array

    def gen_new_z(self, num_unique_indices,
                y_unique_array, var_values, batch_int,
            y_batch_array, contextual=False, a_tch=None, b_tch=None):
        """get var_values for all y's, repeated for the repeated y's """
        # create dictionary from unique y's to var_values
        if not contextual:
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
            return new_var_values, a_tch, b_tch
        else:
            # create dictionary from unique y's to var_values
            y_to_var_values_dict = {}
            for i in range(num_unique_indices):
                y_to_var_values_dict[tuple(tuple(v[i].flatten())\
                        for v in y_unique_array)] = ([v[i] for v in var_values], a_tch[i], b_tch[i])
            # initialize new var_values
            shapes = [torch.tensor(v.shape) for v in var_values]
            for i in range(len(shapes)):
                shapes[i][0] = batch_int
            new_var_values = [torch.zeros(*shape, dtype=settings.DTYPE) for shape in shapes]
            ab_shapes = [torch.tensor(a_tch.shape),torch.tensor(b_tch.shape)]
            ab_shapes[0][0] = batch_int
            ab_shapes[1][0] = batch_int
            new_a_tch = torch.zeros(*ab_shapes[0], dtype=settings.DTYPE)
            new_b_tch = torch.zeros(*ab_shapes[1], dtype=settings.DTYPE)
            # populate new_var_values using the dictionary
            for i in range(batch_int):
                values_list = y_to_var_values_dict[tuple(tuple(v[i].flatten())\
                        for v in y_batch_array)]
                for j in range(len(var_values)):
                    new_var_values[j][i] = values_list[0][j]
                new_a_tch[i] = values_list[1]
                new_b_tch[i] = values_list[2]
        return new_var_values, new_a_tch, new_b_tch

    def grid(
        self,
        rholst=settings.RHO_LST_DEFAULT,
        seed=settings.SEED_DEFAULT,
        init_A=settings.INIT_A_DEFAULT,
        init_b=settings.INIT_B_DEFAULT,
        init_rho = settings.INIT_RHO_DEFAULT_GRID,
        init_alpha=settings.INIT_ALPHA_DEFAULT,
        test_percentage=settings.TEST_PERCENTAGE_DEFAULT,
        solver_args=settings.LAYER_SOLVER,
        quantiles=settings.QUANTILES,
        newdata = settings.NEWDATA_DEFAULT,
        eta = settings.ETA_LAGRANGIAN_DEFAULT,
        contextual = settings.CONTEXTUAL_DEFAULT,
        linear = settings.CONTEXTUAL_LINEAR_DEFAULT
    ):
        r"""
        Perform gridsearch to find optimal :math:`\rho`-ball around data.

        Args:
        rholst : np.array, optional
            The list of :math:`\rho` to iterate over. "Default np.logspace(-3, 1, 20)
        seed: int, optional
            The seed to control the train test split. Default 1.
        init_A: np.array
            The shape A of the set
        init_b: np.array
            The shape b of the set
        init_alpha: float, optional
            The alpha value of the CVaR constraint
        test_percentage: float, optional
            The percengate of the data used in the testing set
        solver_args: dict, optional
            Optional arguments to pass to the solver
        quantiles: tuple, optional
            The quantiles to calculate for the testing results
        newdata: tuple, optional
            New data for the uncertain parameter and context parameters. should
            be given as a tuple with two entries, a np.array for u, and a list
            of np.arrays for x.
        eta:
            The eta value for the CVaR constraint
        contextual:
            Whether or not a contextual set is considered
        linear:
            The linear NN model if contextual is true

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

        if contextual:
            if linear is None:
                raise ValueError("Missing NN-Model")

        self.train_flag = False
        df = pd.DataFrame(
            columns=["Rho"])
        _,_,_,_,_,_,_ = self._split_dataset(test_percentage, seed)
        if newdata is not None:
            newtest_set, x_set = newdata
            self.test_size = newtest_set.shape[0]
            self.u_test_tch = torch.tensor(newtest_set, requires_grad=
                                         self.train_flag, dtype=settings.DTYPE)
            self.u_test_set = newdata
            if not isinstance(x_set, list):
                self.x_test_tch = [torch.tensor(x_set,
                        requires_grad=self.train_flag, dtype=settings.DTYPE)]
            else:
                self.x_test_tch = [torch.tensor(x, requires_grad=self.train_flag,
                                        dtype=settings.DTYPE) for x in x_set]

        self.cvxpylayer = self.create_cvxpylayer()

        grid_stats = GridStats()

        lam = 1000 * torch.ones(self.num_g_total, dtype=settings.DTYPE)
        # initialize torches
        rho_tch = self._gen_rho_tch(1)
        a_tch_init, b_tch_init, alpha, slack = self._init_torches(
            init_A, init_b,init_alpha, self.u_train_set)

        x_batch_array, num_unique_indices, x_unique, \
            x_unique_array = self.gen_unique_x(self.x_test_tch)

        x_batch_array_t, num_unique_indices_t, x_unique_t, \
            x_unique_array_t = self.gen_unique_x(self.x_train_tch)

        for rho in rholst:
            rho_tch = torch.tensor(
                rho*init_rho, requires_grad=self.train_flag, dtype=settings.DTYPE)
            if contextual:
                a_tch_init, b_tch_init = self.create_tensors_linear(
                                    x_unique, linear)
            z_unique = self.cvxpylayer(rho_tch, *self.cp_param_tch,
                                    *x_unique, a_tch_init,b_tch_init,
                                    solver_args=solver_args)
            new_z_batch, a_tch_init, b_tch_init = self.gen_new_z(
                num_unique_indices,x_unique_array, z_unique,
                self.test_size, x_batch_array, contextual,a_tch_init, b_tch_init )

            if contextual:
                a_tch_init, b_tch_init = self.create_tensors_linear(
                                    x_unique_t, linear)
            z_unique_t = self.cvxpylayer(rho_tch, *self.cp_param_tch,
                                    *x_unique_t, a_tch_init,b_tch_init,
                                    solver_args=solver_args)

            new_z_batch_t,_,_ = self.gen_new_z(num_unique_indices_t,
                     x_unique_array_t, z_unique_t, self.train_size,
                       x_batch_array_t,contextual,a_tch_init, b_tch_init)

            train_stats = TrainLoopStats(
                step_num=np.NAN, train_flag=self.train_flag, num_g_total=self.num_g_total)
            with torch.no_grad():
                test_args = self.order_args(z_batch=new_z_batch,
                                             x_batch=self.x_test_tch,
                                             u_batch=self.u_test_tch)
                obj_test = self.evaluation_metric(self.test_size, test_args, quantiles)
                prob_violation_test = self.prob_constr_violation(self.test_size, test_args)
                _, var_vio = self.lagrangian(self.test_size, \
                                                      test_args, alpha, \
                                                        slack, lam, 1, eta
                )

                test_args_t = self.order_args(z_batch=new_z_batch_t,
                                               x_batch=self.x_train_tch,
                                                u_batch=self.u_train_tch)
                obj_train = self.evaluation_metric(self.train_size,
                                                test_args_t, quantiles)
                prob_violation_train = self.prob_constr_violation(self.train_size,test_args_t)
                _, var_vio_train = self.lagrangian(self.train_size,
                                        test_args_t, alpha,slack, lam, 1, eta
                )

            train_stats.update_test_stats(obj_test, prob_violation_test, var_vio)
            train_stats.update_train_stats(None, obj_train, prob_violation_train,var_vio_train)
            grid_stats.update(train_stats, obj_test, rho_tch, a_tch_init, z_unique)

            new_row = train_stats.generate_test_row(
                self._calc_coverage, a_tch_init,b_tch_init,
                alpha, self.u_test_tch,rho_tch, self.unc_set, z_unique,
                contextual,linear, self.x_test_tch)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

        self.orig_problem._trained = True
        self.unc_set._trained = True

        if contextual:
            self.unc_set.a.value = (
                grid_stats.minrho * a_tch_init[0]).detach().numpy().copy()
            self.unc_set.b.value = (
                b_tch_init[0]).detach().numpy().copy()
            b_value = self.unc_set.b.value[0]
        else:
            self.unc_set.a.value = (
                grid_stats.minrho * a_tch_init).detach().numpy().copy()
            self.unc_set.b.value = (
                b_tch_init).detach().numpy().copy()
            b_value = self.unc_set.b.value
        return Result(
            self,
            self.problem_canon,
            df,
            None,
            self.unc_set.a.value,
            b_value,
            grid_stats.minrho.detach().numpy().copy(),
            grid_stats.minval,
            grid_stats.z_batch,
        )


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
        self.trainval = obj[1].item()
        self.prob_violation_train = prob_violation_train.detach().numpy()
        self.violation_train = train_constraint.numpy().sum()/self.num_g_total

    def update_test_stats(self, obj_test, prob_violation_test, var_vio):
        """
        This function updates the statistics after each training iteration
        """
        self.lower_testval = obj_test[0].item()
        self.testval = obj_test[1].item()
        self.upper_testval = obj_test[2].item()
        self.prob_violation_test = prob_violation_test.detach().numpy()
        self.violation_test = var_vio.numpy().sum()/self.num_g_total

    def generate_train_row(self, a_tch, rho_tch, lam, mu, alpha,
                            slack,contextual=False, linear = None):
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
        if contextual:
            row_dict["dfnorm"] = np.linalg.norm(linear.weight.grad) \
                        + np.linalg.norm(linear.bias.grad)
            row_dict["gradnorm"] = torch.hstack([linear.weight.grad,
                                                 linear.bias.grad.view(
                                            linear.bias.grad.shape[0],1)])
        else:
            row_dict["dfnorm"] = np.linalg.norm(a_tch.grad)
            row_dict["gradnorm"] = a_tch.grad
        row_dict["Rho"] = rho_tch.detach().numpy().copy()
        new_row = pd.Series(row_dict)
        return new_row

    def generate_test_row(self, calc_coverage, a_tch, b_tch,
                        alpha, test_tch, rho_tch, uncset, z_batch= None,
                        contextual = False,linear = None,x_test_tch = None):
        """
        This function generates a new row with the statistics
        """
        coverage_test = calc_coverage(
            test_tch, a_tch, b_tch, uncset._rho*rho_tch,uncset.p,contextual,x_test_tch, linear)
        row_dict = {
            "Test_val":         self.testval,
            "Lower_test": self.lower_testval,
            "Upper_test": self.upper_testval,
            "Probability_violations_test":       self.prob_violation_test,
            "Violations_test":   self.violation_test,
            "Coverage_test":    coverage_test,
            "Avg_prob_test": np.mean(self.prob_violation_test),
            "z_vals": z_batch,
            "x_vals": x_test_tch,
            "Rho": rho_tch.detach().numpy().copy()
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
            self.minval = obj[1]
            self.minrho = rho_tch.clone()
            self.minT = a_tch.clone()
            self.z_batch = z_batch


class Result(ABC):
    """ A class to store the results of training """

    def __init__(self, prob, probnew, df, df_test, A, b, rho, \
                 obj, z, a_history=None,
                 b_history=None, rho_history = None, linear=None):
        self._final_prob = probnew
        self._problem = prob
        self._df = df
        self._df_test = df_test
        self._A = A
        self._b = b
        self._obj = obj
        self._z = z
        self._rho = rho
        self._a_history = a_history
        self._b_history = b_history
        self._rho_history = rho_history
        self._linear = linear

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
    def linear(self):
        return self._linear
