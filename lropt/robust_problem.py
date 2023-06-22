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

from lropt.parameter import Parameter
from lropt.remove_uncertain.remove_uncertain import RemoveUncertainParameters
from lropt.settings import DTYPE, EPS_LST_DEFAULT, LAYER_SOLVER, OPTIMIZERS
from lropt.shape_parameter import ShapeParameter
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.distribute_uncertain_params import Distribute_Uncertain_Params
from lropt.uncertain_canon.uncertain_chain import UncertainChain
from lropt.uncertainty_sets.mro import MRO


class RobustProblem(Problem):
    """Create a Robust Optimization Problem with uncertain variables"""

    def __init__(self, objective, constraints, objective_torch=None, constraints_torch=None):
        self._trained = False
        self._values = None
        self._numvars = 0
        super(RobustProblem, self).__init__(objective, constraints)
        self._trained = False
        self._values = None
        self.new_prob = None

        self.num_scenarios = self.verify_y_parameters()
        self.f, self.g = objective_torch, constraints_torch
        self.l, self.h = self.fg_to_lh(objective_torch, constraints_torch)
        self.num_g = len(self.g)

    @property
    def trained(self):
        return self._trained

    @property
    def param_values(self):
        return self._values

    def uncertain_parameters(self):
        """Find uncertain parameters
        """
        return [v for v in self.parameters() if isinstance(v, UncertainParameter)]

    def y_parameters(self):
        """Find y parameters
        """
        return [v for v in self.parameters() if isinstance(v, Parameter)]

    def shape_parameters(self, problem):
        return [v for v in problem.parameters() if isinstance(v, ShapeParameter)]

    def verify_y_parameters(self):
        y_parameters = self.y_parameters()
        num_scenarios = 1
        if len(y_parameters) > 0:
            num_scenarios = y_parameters[0].data.shape[0]
        for param in y_parameters:
            assert (param.data.shape[0] == num_scenarios)
        return num_scenarios

    def fg_to_lh(self, f_tch, g_tch):
        '''Returns l and h for single x,y,u triplet (i.e. one instance of each)'''
        sig_f = signature(f_tch)
        arg_num = len(sig_f.parameters)
        true_arg_num = (len(self.variables())
                        + len(self.y_parameters()) + len(self.uncertain_parameters()))
        assert arg_num == true_arg_num, "incorrect number of user's pytorch function arguments"
        h_funcs = []
        for g in g_tch:
            assert len(signature(g).parameters) == true_arg_num, \
                  "incorrect number of user's pytorch function arguments"
            def hg(vars, y_params, u_params, alpha, eta, kappa):
                return (torch.maximum(g(*vars,*y_params,*u_params)- alpha,
                                      torch.tensor(0.,requires_grad = True)) / eta + alpha - kappa)
            h_funcs.append(hg)

        l_func = f_tch
        return l_func, h_funcs

    def F(self, vars, y_params_mat, u_params_mat):
        '''
        Arguments
        _________
        vars: list
            list of torch instances of the problem variables, presented in the order
            they are defined in the cvxpy problem
        y_params_mat: 2-d list
            2-d list of y_parameters with shape = (num instances, num y parameters)
                [[y_1^(scene 1),...,y_n^(scene 1)],
                [y_1^(scene 2),...,y_n^(scene 2)],
                ...,
                [y_1^(scene J),...,y_n^(scene J)]]
            in order they are presented in the problem.
            However, each y_k^j is a tensor with shape of the kth y parameter introduced
        u_params_mat: 2-d list
            2-d list of u parameters with shape = (num instances, num u parameters)
                [[u_1^(1),...,u_m^(1)],
                [u_1^(2),...,u_m^(2)],
                ...,
                [u_1^(N),...,u_m^(N)]]
            in order they are presented in the problem.
            However, each u_k^n is a tensor with shape of the kth u parameter introduced
        alpha: torch.tensor
            alpha from problem
        eta: float
        kappa: float
        '''
        J = len(y_params_mat)
        N = len(u_params_mat)

        sum = 0
        for i in range(N):
            for j in range(J):
                sum += self.l(*vars, *y_params_mat[j], *u_params_mat[i])
        expectation = sum / (J * N)
        return expectation

    def H(self, vars, y_params_mat, u_params_mat, alpha, eta, kappa):
        '''
        Arguments
        _________
        vars: list
            list of torch instances of the problem variables, presented in the order
            they are defined in the cvxpy problem
        y_params_mat: 2-d list
            2-d list of y_parameters with shape = (num instances, num y parameters)
                [[y_1^(scene 1),...,y_n^(scene 1)],
                [y_1^(scene 2),...,y_n^(scene 2)],
                ...,
                [y_1^(scene J),...,y_n^(scene J)]]
            in order they are presented in the problem.
            However, each y_k^j is a tensor with shape of the kth y parameter introduced
        u_params_mat: 2-d list
            2-d list of u parameters with shape = (num instances, num u parameters)
                [[u_1^(1),...,u_m^(1)],
                [u_1^(2),...,u_m^(2)],
                ...,
                [u_1^(N),...,u_m^(N)]]
            in order they are presented in the problem.
            However, each u_k^n is a tensor with shape of the kth u parameter introduced
        alpha: torch.tensor
            alpha from problem
        eta: float
        kappa: float
        '''
        J = len(y_params_mat)
        N = len(u_params_mat)

        num_g = len(self.h)
        H = torch.zeros(num_g, dtype=float)
        for k, h_k in enumerate(self.h):
            sum = 0
            for i in range(N):
                for j in range(J):
                    sum += h_k(vars, y_params_mat[j], u_params_mat[i], alpha, eta, kappa)
            h_k_expectation = sum / (J * N)
            H[k] = h_k_expectation
        return H

    def prob_constr_violation(self, vars, y_params_mat, u_params_mat):
        num_g = len(self.g)
        J = len(y_params_mat)
        N = len(u_params_mat)
        G = torch.zeros((num_g, J, N), dtype=float)

        for k, g_k in enumerate(self.g):
            for i in range(N):
                for j in range(J):
                    G[k, j, i] = g_k(*vars, *y_params_mat[j], *u_params_mat[i])

        G_max = torch.max(G, dim=0)[0]
        G_max_violate = (G_max > 0).float()
        return torch.mean(G_max_violate)

    # helper function for intermediate version
    def _udata_to_lst(self, data):
        num_instances = data.shape[0]
        u_params_mat = []
        for i in range(num_instances):
            u_params_mat.append([data[i, :]])
        return u_params_mat

    def aug_lag(self, vars, y_params_mat, u_params_mat, alpha, mu, lam, eta=0.05, kappa=-0.015):
        '''Defines L augmented lagrangian function, which computes loss for
        Arguments
        _________
        vars: list
            list of torch instances of the problem variables, presented in the order
            they are defined in the cvxpy problem
        y_params_mat: 2-d list
            2-d list of y_parameters with shape = (num instances, num y parameters)
                [[y_1^(scene 1),...,y_n^(scene 1)],
                [y_1^(scene 2),...,y_n^(scene 2)],
                ...,
                [y_1^(scene J),...,y_n^(scene J)]]
            in order they are presented in the problem.
            However, each y_k^j is a tensor with shape of the kth y parameter introduced
        u_params_mat: 2-d list
            2-d list of u parameters with shape = (num instances, num u parameters)
                [[u_1^(1),...,u_m^(1)],
                [u_1^(2),...,u_m^(2)],
                ...,
                [u_1^(N),...,u_m^(N)]]
            in order they are presented in the problem.
            However, each u_k^n is a tensor with shape of the kth u parameter introduced
        alpha: torch.tensor
            alpha from problem
        eta: float
        kappa: float
        '''
        F = self.F(vars, y_params_mat, u_params_mat)
        H = self.H(vars, y_params_mat, u_params_mat, alpha, eta, kappa)
        prob_constr_violation = self.prob_constr_violation(vars, y_params_mat, u_params_mat)
        return F + lam @ H + (mu / 2) * (torch.norm(H)**2), F, H, prob_constr_violation

    # create function for only remove_uncertain reduction
    def _construct_chain(
        self, solver: Optional[str] = None, gp: bool = False,
        enforce_dpp: bool = True, ignore_dpp: bool = False,
        solver_opts: Optional[dict] = None,
        canon_backend: str | None = None,
    ) -> SolvingChain:
        """
        Construct the chains required to reformulate and solve the problem.
        In particular, this function
        # finds the candidate solvers
        # constructs the solving chain that performs the
           numeric reductions and solves the problem.
        Arguments
        ---------
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
        Returns
        -------
        A solving chain
        """
        # if enforce_dpp is False:
        #      warnings.warn("should enforce problem is dpp")

        candidate_solvers = self._find_candidate_solvers(solver=solver, gp=gp)
        self._sort_candidate_solvers(candidate_solvers)
        solving_chain = construct_solving_chain(self, candidate_solvers, gp=gp,
                                                enforce_dpp=enforce_dpp,
                                                ignore_dpp=ignore_dpp,
                                                canon_backend=canon_backend,
                                                solver_opts=solver_opts
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

    def train(
        self, eps=False, fixb=False, step=45, lr=0.0001, scheduler=True, momentum=0.8,
        optimizer="SGD", init_eps=None,
        init_A=None, init_b=None, save_iters=False, seed=1, init_lam=0., init_mu=1,
        mu_multiplier=1.01, init_alpha=-0.01,
        target_cvar=-0.015, test_percentage=0.2, scenarios=None, num_scenarios=None,
        max_inner_iter=10, solver: Optional[str] = None
    ):
        r"""
        Trains the uncertainty set parameters to find optimal set w.r.t. loss metric

        Parameters
        -----------
        eps : bool, optional
           If True, train only epsilon, where :math:`A = \epsilon I, \
           b = \epsilon \bar{d}`, where :math:`\bar{d}` is the centroid of the
           training data. Default False.
        step : int, optional
            The total number of gradient steps performed. Default 45.
        lr : float, optional
            The learning rate of gradient descent. Default 0.01.
        momentum: float between 0 and 1, optional
            The momentum for gradient descent. Default 0.8.
        optimizer: str or letters, optional
            The optimizer to use tor the descent algorithm. Default "SGD".
        init_eps : float, optional
            The epsilon to initialize :math:`A` and :math:`b`, if passed. If not passed,
            :math:`A` will be initialized as the inverse square root of the
            covariance of the data, and b will be initialized as :math:`\bar{d}`.
        seed : int, optional
            The seed to control the random state of the train-test data split. Default 1.

        Returns
        -------
        A pandas data frame with information on each :math:r`\epsilon` having the following columns:
            Opt_val: float
                The objective value of the Robust Problem
            Loss_val: float
                The value of the loss function applied to the training data
            Eval_val: float
                The value of the loss function applied to the evaluation data
            Eps: float
                The :math:`\epsilon` value
        """

        T_iter = []
        b_iter = []
        if self.uncertain_parameters():

            unc_set = self.uncertain_parameters()[0].uncertainty_set

            if unc_set.data is None:
                raise ValueError("Cannot train without uncertainty set data")


            unc_reductions = []
            if type(self.objective) == Maximize:
                unc_reductions += [FlipObjective()]
            unc_reductions += [RemoveUncertainParameters()]

            newchain = UncertainChain(self, reductions=unc_reductions)

            prob, inverse_data = newchain.apply(self)
            if unc_set.paramT is not None:

                if type(unc_set) == MRO:
                    mro_set = True
                else:
                    mro_set = False

                df = pd.DataFrame(columns=["step", "Opt_val", "Eval_val", "Loss_val",
                                           "Violations", "A_norm"])

                # setup train and test data
                train, test = train_test_split(unc_set.data, test_size=int(
                    unc_set.data.shape[0]*test_percentage), random_state=seed)
                val_dset = torch.tensor(train, requires_grad=True, dtype=DTYPE)
                eval_set = torch.tensor(test, requires_grad=True, dtype=DTYPE)

                cvxpylayer = CvxpyLayer(prob, parameters=self.y_parameters()
                                        + self.shape_parameters(prob), variables=self.variables())
                if not eps:
                    # initialize parameters to train
                    if len(np.shape(np.cov(train.T))) >= 1:
                        if init_eps and init_A is None:
                            init = (1/init_eps)*np.eye(train.shape[1])
                        elif init_A is not None:
                            init = np.array(init_A)
                            if init_eps:
                                init = (1/init_eps)*init
                        else:
                            init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
                        paramb_tch = torch.tensor(-init@np.mean(train, axis=0), requires_grad=True,
                                                  dtype=DTYPE)
                    else:
                        if init_eps and init_A is None:
                            init = (1/init_eps)*np.eye(1)
                        elif init_A is not None:
                            init = np.array(init_A)
                            if init_eps:
                                init = (1/init_eps)*init
                        else:
                            init = np.array([[np.cov(train.T)]])
                        paramb_tch = torch.tensor(-init@np.mean(train, axis=0), requires_grad=True,
                                                  dtype=DTYPE)
                    if init_b is not None:
                        paramb_tch = torch.tensor(np.array(init_b), requires_grad=True,
                                                  dtype=DTYPE)

                    alpha = torch.tensor(init_alpha, requires_grad=True)
                    paramT_tch = torch.tensor(init, requires_grad=True, dtype=DTYPE)

                    if save_iters:
                        T_iter.append(paramT_tch.detach().numpy().copy())
                        if not mro_set:
                            b_iter.append(paramb_tch.detach().numpy().copy())

                    if fixb or mro_set:
                        if mro_set and unc_set._uniqueA:
                            if init_A is None:
                                paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                            elif init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m):
                                paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                        paramT = paramT_tch.detach().numpy()
                        paramT_tch = torch.tensor(paramT, requires_grad=True, dtype=DTYPE)
                        variables = [paramT_tch, alpha]
                    else:
                        variables = [paramT_tch, paramb_tch, alpha]

                    opt = OPTIMIZERS[optimizer](variables, lr=lr, momentum=momentum)
                    if scheduler:
                        scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

                    # y's and cvxpylayer begin
                    y_parameters = self.y_parameters()
                    num_scenarios = self.num_scenarios
                    prob.parameters()


                    newlst = {}
                    for scene in range(num_scenarios):
                        newlst[scene] = []
                        for i in range(len(y_parameters)):
                            newlst[scene].append(torch.tensor(
                                np.array(y_parameters[i].data[scene,:])
                                .astype(float), requires_grad=True, dtype=DTYPE))
                        newlst[scene].append(paramT_tch)
                        if not mro_set:
                            newlst[scene].append(paramb_tch)
                    # train
                    lam = init_lam * torch.ones((num_scenarios, self.num_g), dtype=float)
                    curlam = init_lam * torch.ones(self.num_g, dtype=float)
                    mu = init_mu

                    for steps in range(step):

                        totloss = torch.tensor(0., dtype=float)
                        totevalloss = torch.tensor(0., dtype=float)
                        optval = torch.tensor(0., dtype=float)
                        testval = torch.tensor(0., dtype=float)
                        test_vio = torch.tensor(0., dtype=float)
                        train_vio = torch.tensor(0., dtype=float)
                        violation_val = torch.tensor(0., dtype=float)
                        violation_train = torch.tensor(0., dtype=float)
                        totdloss = torch.tensor(0., dtype=float) #v^(t-1)
                        totuloss = torch.tensor(0., dtype=float) #v_1^t Line 8 algorithm 2
                        totvloss = torch.tensor(0., dtype=float)
                        random_int = np.random.randint(0, val_dset.shape[0],
                                                       int(val_dset.shape[0]/10))
                        # FOR LOOP 1 line 6 in algorithm 2

                        for scene in range(num_scenarios):
                            if not mro_set:
                                newlst[scene][-1] = paramb_tch
                                newlst[scene][-2] = paramT_tch
                            else:
                                newlst[scene][-1] = paramT_tch
                            var_values = cvxpylayer(*newlst[scene],
                                                    solver_args=LAYER_SOLVER)

                            temploss, obj, cvar_update, violations = self.aug_lag(var_values,
                                [newlst[scene][:-2]], self._udata_to_lst(val_dset),
                                    alpha, mu, curlam)

                            # temploss, obj, violations, cvar_update =
                            #      unc_set.loss(# v_0 line 3 algorithm 2
                            #     *var_values, *newlst[scene][:-2], alpha, val_dset, mu, curlam,
                            #     target=target_cvar)
                            uloss, _, _, _ = self.aug_lag(var_values, [newlst[scene][:-2]],
                                            self._udata_to_lst(val_dset[random_int]),
                                                alpha, mu, curlam)
                            # uloss, _, _, _ = unc_set.loss(
                            #     *var_values, *newlst[scene][:-2], alpha, val_dset[random_int], \
                            #  mu, # v_1^T (can use random y's as well)
                            #     curlam, target=target_cvar)
                            totdloss = totdloss + temploss/num_scenarios
                            totuloss = totuloss + uloss/num_scenarios  #v_1^t
                        backuptotdloss = totdloss.detach().clone()
                        totdloss.backward(retain_graph=True) # gradient for line 6 (v^(t-1))
                        totdloss = backuptotdloss
                        opt.step()
                        opt.zero_grad()
                        if scheduler:
                            scheduler_.step(totdloss)
                        tot_inner_step = np.random.randint(0, max_inner_iter)
                        # FOR LOOP 2

                        for inner_step in range(tot_inner_step-1):
                            totvloss = torch.tensor(0., requires_grad=True) #v_2^(t) Line 9
                            newuloss = torch.tensor(0., requires_grad=True) #v_1^(t+1) Line 8
                            for scene in range(num_scenarios):
                                if not mro_set:
                                    newlst[scene][-1] = paramb_tch
                                    newlst[scene][-2] = paramT_tch
                                else:
                                    newlst[scene][-1] = paramT_tch
                                var_values = cvxpylayer(*newlst[scene],
                                                        solver_args=LAYER_SOLVER)
                                vloss, _, _, _ = self.aug_lag(var_values, [newlst[scene][:-2]],
                                                self._udata_to_lst(val_dset[random_int]),
                                                    alpha, mu, curlam)
                                # vloss, _, _, _ = unc_set.loss(
                                #     *var_values, *newlst[scene][:-2], alpha, val_dset[random_int],
                                #     mu, curlam,
                                #     target=target_cvar)
                                random_int = np.random.randint(0, val_dset.shape[0],
                                                               int(val_dset.shape[0]/10))
                                uloss, _, _, _ = self.aug_lag(var_values, [newlst[scene][:-2]],
                                                self._udata_to_lst(val_dset[random_int]),
                                                    alpha, mu, curlam)
                                # uloss, _, _, _ = unc_set.loss(
                                #     *var_values, *newlst[scene][:-2], alpha, val_dset[random_int],
                                #     mu, curlam, target=target_cvar)
                                totvloss = totvloss + vloss/num_scenarios
                                newuloss = newuloss + uloss/num_scenarios
                            totdloss = totvloss + (1-0.01)*(totdloss - totuloss) # Line 10 in alg
                            backuptotdloss = totdloss.detach().clone()
                            totdloss.backward(retain_graph=True)
                            totdloss = backuptotdloss
                            totuloss = newuloss.detach().clone()
                            if inner_step < tot_inner_step-2:
                                opt.step()
                                opt.zero_grad()
                                if scheduler:
                                    scheduler_.step(totdloss)
                        # update lagrange multipliers

                        # BEFORE UPDTATING PANDAS DATAFRAME
                        # FOR LOOP 3 --> Algorithm 1 update lambda/mu
                        for scene in range(num_scenarios):
                            if not mro_set:
                                newlst[scene][-1] = paramb_tch
                                newlst[scene][-2] = paramT_tch
                            else:
                                newlst[scene][-1] = paramT_tch
                            var_values = cvxpylayer(*newlst[scene],
                                                    solver_args=LAYER_SOLVER)
                            temploss, obj, cvar_update, violations = self.aug_lag(var_values,
                                        [newlst[scene][:-2]], self._udata_to_lst(val_dset),
                                            alpha, mu, curlam)
                            # temploss, obj, violations, cvar_update = unc_set.loss(
                            #     *var_values, *newlst[scene][:-2], alpha, val_dset,  mu, curlam,
                            #     target=target_cvar)
                            evalloss, obj2, var_vio, violations2 = self.aug_lag(var_values,
                                     [newlst[scene][:-2]], self._udata_to_lst(eval_set),
                                        alpha, mu, curlam)
                            # evalloss, obj2, violations2, var_vio = unc_set.loss(
                            #     *var_values, *newlst[scene][:-2], alpha, eval_set, mu, curlam,
                            #     target=target_cvar)
                            lam[scene, :] = cvar_update
                            totloss = totloss + temploss/num_scenarios
                            totevalloss = totevalloss + evalloss/num_scenarios
                            optval += obj.item()
                            testval += obj2.item()
                            test_vio += violations2.item()
                            train_vio += violations.item()
                            violation_val += var_vio.item()
                            violation_train += cvar_update.item()

                        curlam = torch.maximum(curlam + mu*(torch.mean(lam, axis=0)), \
                                               torch.zeros(self.num_g, dtype=float))
                        mu = mu*mu_multiplier
                        # BEFORE UPDTATING PANDAS DATAFRAME
                        newrow = pd.Series(
                            {"step": steps,
                             "Loss_val": totloss.item(),
                             "Eval_val": totevalloss.item(),
                             "Opt_val": optval/num_scenarios,
                             "Test_val": testval/num_scenarios,
                             "Violations": test_vio/num_scenarios,
                             "Violations_train": train_vio/num_scenarios,
                             "Violation_val": violation_val/num_scenarios,
                             "Violation_train": violation_train/num_scenarios,
                             "A_norm": np.linalg.norm(paramT_tch.detach().numpy().copy()),
                             "mu": mu,
                             "lam": curlam.detach().numpy().copy(),
                             "alpha": alpha.item(),
                             "alphagrad": alpha.grad,
                             "dfnorm": np.linalg.norm(paramT_tch.grad),
                             "gradnorm": paramT_tch.grad}
                        )
                        df = pd.concat([df, newrow.to_frame().T], ignore_index=True)

                        if save_iters:
                            T_iter.append(paramT_tch.detach().numpy().copy())
                            if not mro_set:
                                b_iter.append(paramb_tch.detach().numpy().copy())

                    self._trained = True
                    unc_set._trained = True
                    unc_set.paramT.value = paramT_tch.detach().numpy().copy()
                    if not mro_set:
                        unc_set.paramb.value = paramb_tch.detach().numpy().copy()

                # HAVE NOT YET CHANGED
                else:
                    if init_eps:
                        eps_tch = torch.tensor(1/np.array(init_eps), requires_grad=True,
                                               dtype=DTYPE)

                        if mro_set:
                            if unc_set._uniqueA and eps_tch.shape == torch.Size([]):
                                eps_tch = eps_tch.repeat(unc_set._K)
                                eps_tch = eps_tch.detach().numpy()
                                eps_tch = torch.tensor(eps_tch, requires_grad=True,
                                                       dtype=DTYPE)
                    else:
                        eps_tch = torch.tensor(1., requires_grad=True, dtype=DTYPE)
                        if mro_set and unc_set._uniqueA:
                            eps_tch = eps_tch.repeat(unc_set._K)
                            eps_tch = eps_tch.detach().numpy()
                            eps_tch = torch.tensor(eps_tch, requires_grad=True, dtype=DTYPE)

                    if init_A is not None:
                        init = torch.tensor(init_A, requires_grad=True, dtype=DTYPE)
                    else:
                        init = torch.tensor(np.eye(train.shape[1]), requires_grad=True,
                                            dtype=DTYPE)

                    if init_b is not None:
                        init_bval = torch.tensor(init_b, requires_grad=True, dtype=DTYPE)
                    else:
                        init_bval = -init@torch.tensor(np.mean(train, axis=0), dtype=DTYPE)

                    if not mro_set:
                        paramb_tch = eps_tch*init_bval
                        paramT_tch = eps_tch*init
                    elif unc_set._uniqueA:
                        if init_A is None or \
                            (init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m)):
                            paramT_tch = eps_tch[0]*init
                            for k_ind in range(1, unc_set._K):
                                paramT_tch = torch.vstack((paramT_tch, eps_tch[k_ind]*init))
                            case = 0
                        else:
                            paramT_tch = eps_tch[0]*torch.tensor(init_A[0:unc_set._m, 0:unc_set._m],
                                                                 dtype=DTYPE)
                            for k_ind in range(1, unc_set._K):
                                paramT_tch = torch.vstack(
                                    (paramT_tch, eps_tch[k_ind] *
                                        torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)*unc_set._m,
                                                     0:unc_set._m],
                                                     dtype=DTYPE)))
                            case = 1
                    else:
                        paramT_tch = eps_tch*init
                        case = 2
                    alpha = torch.tensor(init_alpha, requires_grad=True)
                    variables = [eps_tch, alpha]
                    opt = OPTIMIZERS[optimizer](variables, lr=lr, momentum=momentum)
                    if scheduler:
                        scheduler_ = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

                    y_parameters = self.y_parameters()
                    num_scenarios = self.num_scenarios

                    newlst = {}
                    for scene in range(num_scenarios):
                        newlst[scene] = []
                        for i in range(len(y_parameters)):
                            newlst[scene].append(torch.tensor(np.array(y_parameters[i].data[scene,:]).astype(
                                float), requires_grad=True, dtype=DTYPE))
                        newlst[scene].append(paramT_tch)
                        if not mro_set:
                            newlst[scene].append(paramb_tch)

                    # train
                    lam = init_lam * torch.ones((num_scenarios, self.num_g), dtype=float)
                    curlam = init_lam * torch.ones(self.num_g, dtype=float)
                    mu = init_mu

                    for steps in range(step):
                        totloss = torch.tensor(0., dtype=float)
                        totevalloss = torch.tensor(0., dtype=float)
                        optval = torch.tensor(0., dtype=float)
                        testval = torch.tensor(0., dtype=float)
                        test_vio = torch.tensor(0., dtype=float)
                        train_vio = torch.tensor(0., dtype=float)
                        violation_val = torch.tensor(0., dtype=float)
                        violation_train = torch.tensor(0., dtype=float)
                        random_int = np.random.randint(0, val_dset.shape[0],
                                                       int(val_dset.shape[0]/10))
                        for scene in range(num_scenarios):
                            if not mro_set:
                                newlst[scene][-1] = eps_tch*init_bval
                                newlst[scene][-2] = eps_tch*init
                            else:
                                if case == 0:
                                    paramT_tch = eps_tch[0]*init
                                    for k_ind in range(1, unc_set._K):
                                        paramT_tch = torch.vstack((paramT_tch, eps_tch[k_ind]*init))
                                elif case == 1:
                                    paramT_tch = eps_tch[0] * \
                                        torch.tensor(init_A[0:unc_set._m, 0:unc_set._m],
                                                     dtype=DTYPE)
                                    for k_ind in range(1, unc_set._K):
                                        paramT_tch = torch.vstack(
                                            (paramT_tch, eps_tch[k_ind] *
                                                torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)
                                                                    * unc_set._m, 0:unc_set._m],
                                                             dtype=DTYPE)))
                                else:
                                    paramT_tch = eps_tch*init
                                newlst[scene][-1] = paramT_tch
                            var_values = cvxpylayer(*newlst[scene],
                                                    solver_args=LAYER_SOLVER)
                            # temploss, obj, violations, cvar_update = unc_set.loss(
                            #     *var_values,  *newlst[scene][:-2], alpha, val_dset, mu, curlam,
                            #     target=target_cvar)

                            temploss, obj, cvar_update, violations = self.aug_lag(var_values,
                                            [newlst[scene][:-2]], self._udata_to_lst(val_dset),
                                                alpha, mu, curlam)

                            # temploss, obj, violations, cvar_update =
                            # unc_set.loss(# v_0 line 3 algorithm 2
                            #     *var_values, *newlst[scene][:-2], alpha, val_dset, mu, curlam,
                            #     target=target_cvar)

                            evalloss, obj2, var_vio, violations2 = self.aug_lag(var_values,
                                        [newlst[scene][:-2]], self._udata_to_lst(eval_set),
                                            alpha, mu, curlam)

                            # evalloss, obj2, violations2, var_vio = unc_set.loss(
                            #     *var_values, *newlst[scene][:-2], alpha, eval_set, mu, curlam,
                            #     target=target_cvar)
                            lam[scene, :] = cvar_update
                            totloss = totloss + temploss
                            totevalloss = totevalloss + evalloss
                            optval += obj.item()
                            testval += obj2.item()
                            test_vio += violations2.item()
                            train_vio += violations.item()
                            violation_val += var_vio.item()
                            violation_train += cvar_update.item()

                        curlam = torch.maximum(curlam + mu*(torch.mean(lam, axis=0)),
                                               torch.zeros(self.num_g, dtype=float))
                        # curlam = np.maximum(curlam + mu*np.mean(lam, axis=0), 0)
                        mu = mu*mu_multiplier
                        totloss = totloss/num_scenarios
                        totloss.backward()
                        newrow = pd.Series(
                            {"step": steps,
                             "Loss_val": totloss.item(),
                             "Eval_val": totevalloss.item()/num_scenarios,
                             "Opt_val": optval/num_scenarios,
                             "Test_val": testval/num_scenarios,
                             "Violations": test_vio/num_scenarios,
                             "Violations_train": train_vio/num_scenarios,
                             "Violation_val": violation_val/num_scenarios,
                             "Violation_train": violation_train/num_scenarios,
                             "A_norm": np.mean(1/eps_tch.detach().numpy().copy()),
                             "Eps_vals": 1/eps_tch.detach().numpy().copy(),
                             "mu": mu,
                             "lam": curlam.detach().numpy().copy(),
                             "alpha": alpha.item(),
                             "alphagrad": alpha.grad,
                             "dfnorm": np.linalg.norm(eps_tch.grad),
                             "gradnorm": eps_tch.grad})
                        df = pd.concat([df, newrow.to_frame().T], ignore_index=True)

                        if save_iters:
                            T_iter.append(paramT_tch.detach().numpy().copy())
                            if not mro_set:
                                b_iter.append(paramb_tch.detach().numpy().copy())

                        if steps < step - 1:
                            opt.step()
                            opt.zero_grad()
                            torch.clamp(eps_tch, min=0.001)
                            if scheduler:
                                scheduler_.step(totloss)

                    self._trained = True
                    unc_set._trained = True

                    if not mro_set:
                        unc_set.paramT.value = (eps_tch*init).detach().numpy().copy()
                        unc_set.paramb.value = (
                            eps_tch*init_bval).detach().numpy().copy()
                    else:
                        unc_set.paramT.value = paramT_tch.detach().numpy().copy()
                self.new_prob = prob
        if eps:
            return_eps = eps_tch.detach().numpy().copy()
        else:
            return_eps = 1
        if not mro_set:
            return Result(self, prob, df, unc_set.paramT.value,
                          unc_set.paramb.value, return_eps, obj.item(), var_values,
                          T_iter=T_iter, b_iter=b_iter)
        else:
            return Result(self, prob, df, unc_set.paramT.value, None, return_eps, obj.item(),
                          var_values,
                          T_iter=T_iter)

    def grid(self, epslst=None, seed=1, init_A=None, init_b=None, init_alpha=-0.01,
             test_percentage=0.2, scenarios=None, num_scenarios=None, solver: Optional[str] = None):
        r"""
        Perform gridsearch to find optimal :math:`\epsilon`-ball around data.

        Parameters
        -----------
        epslst : np.array, optional
            The list of :math:`\epsilon` to iterate over. "Default np.logspace(-3, 1, 20)
        seed: int
            The seed to control the train test split. Default 1.
        solver: optional
            A solver to perform gradient-based learning

        Returns
        -------
        A pandas data frame with information on each :math:`\epsilon` having the following columns:
            Opt_val: float
                The objective value of the Robust Problem
            Loss_val: float
                The value of the loss function applied to the training data
            Eval_val: float
                The value of the loss function applied to the evaluation data
            Eps: float
                The epsilon value
        """
        # if enforce_dpp is False:
        #      warnings.warn("should enforce problem is dpp")
        if epslst is None:
            epslst = EPS_LST_DEFAULT


        if self.uncertain_parameters():
            unc_set = self.uncertain_parameters()[0].uncertainty_set

            if unc_set.data is None:
                raise ValueError("Cannot train without uncertainty set data")


            unc_reductions = []
            if type(self.objective) == Maximize:
                unc_reductions += [FlipObjective()]
            unc_reductions += [RemoveUncertainParameters()]
            newchain = UncertainChain(self, reductions=unc_reductions)
            prob, inverse_data = newchain.apply(self)
            if unc_set.paramT is not None:
                df = pd.DataFrame(columns=["Opt_val", "Eval_val", "Loss_val", "Eps"])
                if type(unc_set) == MRO:
                    mro_set = True
                else:
                    mro_set = False
                # setup train and test data
                train, test = train_test_split(unc_set.data, test_size=int(
                    unc_set.data.shape[0]*test_percentage), random_state=seed)
                val_dset = torch.tensor(train, requires_grad=True, dtype=DTYPE)
                eval_set = torch.tensor(test, requires_grad=True, dtype=DTYPE)
                # create cvxpylayer
                cvxpylayer = CvxpyLayer(prob, parameters=prob.parameters(),
                                        variables=self.variables())

                # paramlst = prob.parameters()
                y_parameters = self.y_parameters()
                num_scenarios = self.num_scenarios

                newlst = {}
                for scene in range(num_scenarios):
                    newlst[scene] = []
                    for i in range(len(y_parameters)):
                        newlst[scene].append(torch.tensor(np.array(y_parameters[i].data[scene,:]).astype(
                            float), requires_grad=True, dtype=DTYPE))
                    newlst[scene].append(0)
                    if not mro_set:
                        newlst[scene].append(0)


                # for scene in range(num_scenarios):
                #     newlst[scene] = []
                #     if not mro_set:
                #         for i in range(len(paramlst[:-2])):
                #             newlst[scene].append(torch.tensor(
                # np.array(scenarios[scene][i]).astype(
                #                 float)))
                #         newlst[scene].append(0)
                #         newlst[scene].append(0)
                #     else:
                #         for i in range(len(paramlst[:-1])):
                #             newlst[scene].append(torch.tensor(
                # np.array(scenarios[scene][i]).astype(
                #                 float)))
                #         newlst[scene].append(0)
                minval = 9999999
                var_vals = 0
                alpha = torch.tensor(init_alpha, requires_grad=True)

                if init_A is not None:
                    init = torch.tensor(init_A, requires_grad=True, dtype=DTYPE)
                else:
                    init = torch.tensor(np.eye(train.shape[1]), requires_grad=True,
                                        dtype=DTYPE)
                if init_b is not None:
                    init_bval = torch.tensor(init_b, requires_grad=True,
                                             dtype=DTYPE)
                else:
                    init_bval = torch.tensor(-np.mean(train, axis=0), requires_grad=True,
                                             dtype=DTYPE)

                for epss in epslst:
                    eps_tch1 = torch.tensor([[1/epss]], requires_grad=True, dtype=DTYPE)
                    totloss = 0
                    totevalloss = []
                    optval = []
                    testval = []
                    test_vio = []
                    train_vio = []
                    violation_val = []
                    violation_train = []

                    for scene in range(num_scenarios):
                        if not mro_set:
                            newlst[scene][-1] = eps_tch1[0][0]*init_bval
                            newlst[scene][-2] = eps_tch1[0][0]*init
                            paramT_tch = eps_tch1[0][0]*init
                        else:
                            if unc_set._uniqueA:
                                if init_A is None or (init_A is not None and init_A.shape[0] !=
                                                      (unc_set._K*unc_set._m)):
                                    paramT_tch = eps_tch1[0][0]*init
                                    paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                                else:
                                    paramT_tch = eps_tch1[0][0]*init
                            else:
                                paramT_tch = eps_tch1[0][0]*init
                            newlst[scene][-1] = paramT_tch
                        var_values = cvxpylayer(*newlst[scene],
                                                solver_args=LAYER_SOLVER)

                        temploss, obj, cvar_update, violations = self.aug_lag(var_values,
                                        [newlst[scene][:-2]], self._udata_to_lst(val_dset),
                                        alpha, mu=100, curlam=1000)
                        # temploss, obj, violations,cvar_update = unc_set.loss(
                        #     *var_values, *newlst[scene][:-2], alpha = torch.tensor(
                        #     init_alpha), data = val_dset)

                        evalloss, obj2, var_vio, violations2 = self.aug_lag(var_values,
                                    [newlst[scene][:-2]], self._udata_to_lst(eval_set),
                                    alpha, mu=100, lam=1000)
                        # evalloss, obj2, violations2, var_vio = unc_set.loss(
                        #     *var_values, *newlst[scene][:-2], alpha = torch.tensor(
                        #     init_alpha), data = eval_set)

                        totloss += temploss.item()
                        totevalloss.append(evalloss.item())
                        optval.append(obj.item())
                        testval.append(obj2.item())
                        test_vio.append(violations2.item())
                        train_vio.append(violations.item())
                        violation_val.append(var_vio.item())
                        violation_train.append(cvar_update.item())
                    totloss = totloss/num_scenarios
                    if totloss <= minval:
                        minval = temploss
                        mineps = eps_tch1.clone()
                        minT = paramT_tch.clone()
                        var_vals = var_values
                    newrow = pd.Series(
                        {"Loss_val": totloss,
                         "Eval_val": totevalloss,
                         "Opt_val": optval,
                         "Test_val": testval,
                         "Violations": test_vio,
                         "Violations_train": train_vio,
                         "Violation_val": violation_val,
                         "Violation_train": violation_train,
                            "Eps": 1/eps_tch1[0][0].detach().numpy().copy()
                         })
                    df = pd.concat([df, newrow.to_frame().T], ignore_index=True)

                self._trained = True
                unc_set._trained = True

                if not mro_set:
                    unc_set.paramT.value = (mineps*init).detach().numpy().copy()
                    unc_set.paramb.value = (
                        mineps[0]*init_bval).detach().numpy().copy()
                else:
                    unc_set.paramT.value = minT.detach().numpy().copy()
                self.new_prob = prob
        if not mro_set:
            return Result(self, prob, df, unc_set.paramT.value,
                          unc_set.paramb.value, mineps[0][0].detach().numpy().copy(),
                          minval, var_vals)
        else:
            return Result(self, prob, df, unc_set.paramT.value,
                          None, mineps[0][0].detach().numpy().copy(), minval, var_vals)

    def dualize_constraints(self):
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
