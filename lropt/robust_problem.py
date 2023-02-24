from abc import ABC
from typing import Optional

import numpy as np
import pandas as pd
import scipy as sc
import torch
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem
from cvxpy.reductions import Dcp2Cone, Qp2SymbolicQp
from cvxpy.reductions.flip_objective import FlipObjective
# from cvxpy.reductions.chain import Chain
from cvxpy.reductions.solvers.solving_chain import (SolvingChain,
                                                    construct_solving_chain)
from cvxpylayers.torch import CvxpyLayer
from sklearn.model_selection import train_test_split

from lropt.remove_uncertain.remove_uncertain import RemoveUncertainParameters
from lropt.settings import EPS_LST_DEFAULT, OPTIMIZERS
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.separate_uncertain_params import \
    Separate_Uncertain_Params
from lropt.uncertain_canon.uncertain_chain import UncertainChain
from lropt.uncertainty_sets.mro import MRO
from lropt.utils import unique_list


class RobustProblem(Problem):
    """Create a Robust Optimization Problem with uncertain variables"""

    def __init__(self, objective, constraints):
        self._trained = False
        self._values = None
        self._numvars = 0
        super(RobustProblem, self).__init__(objective, constraints)
        self._trained = False
        self._values = None
        self.new_prob = None

    @property
    def trained(self):
        return self._trained

    @property
    def param_values(self):
        return self._values

    def uncertain_parameters(self):
        """Find which variables are uncertain

        Returns
        -------
        num_params : int
            the number of unique uncertain parameters in robust problem
        """
        unc_params = []
        # TODO: Add also in cost
        for c in self.constraints:
            unc_params += [v for v in c.parameters()
                           if isinstance(v, UncertainParameter)]

        return unique_list(unc_params)
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
            # import ipdb
            # ipdb.set_trace()
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
        self, eps=False, fixb=True, step=45, lr=0.01, scheduler=True, momentum=0.8,
        optimizer="SGD", init_eps=None, init_A=None, init_b=None, save_iters=False, seed=1, init_lam=10, init_mu=10,
        mu_multiplier=1.02, init_alpha=-0.01,
        target_cvar=0., solver: Optional[str] = None
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
        # if enforce_dpp is False:
        #      warnings.warn("should enforce problem is dpp")

        # candidate_solvers = self._find_candidate_solvers(solver=solver, gp=False)
        # self._sort_candidate_solvers(candidate_solvers)
        # solving_chain = construct_solving_chain(self, candidate_solvers, gp=False,
        #                                         enforce_dpp=True,
        #                                         ignore_dpp=False,
        #                                         # Comment this for now. Useful
        #                                         # in next cvxpy release
        #                                         solver_opts=None
        #                                         )
        #

        T_iter = []
        b_iter = []
        if self.uncertain_parameters():
            # import ipdb
            # ipdb.set_trace()
            unc_set = self.uncertain_parameters()[0].uncertainty_set

            if unc_set.data is None:
                raise ValueError("Cannot train without uncertainty set data")

            # new_reductions = solving_chain.reductions
            # # Find position of Dcp2Cone or Qp2SymbolicQp
            # for idx in range(len(new_reductions)):
            #     if type(new_reductions[idx]) in [Dcp2Cone, Qp2SymbolicQp]:
            #         # Insert RemoveUncertainParameters before those reductions
            #         new_reductions.insert(idx, RemoveUncertainParameters())
            #         unc_reductions = new_reductions[:idx+1]
            #         break
        # return a chain instead (chain.apply, return the problem and inverse data)
            unc_reductions = []
            if type(self.objective) == Maximize:
                unc_reductions += [FlipObjective()]
            unc_reductions += [RemoveUncertainParameters()]

            newchain = UncertainChain(self, reductions=unc_reductions)
            prob, inverse_data = newchain.apply(self)
            if unc_set.paramT is not None:
                # import ipdb
                # ipdb.set_trace()
                if type(unc_set) == MRO:
                    mro_set = True
                else:
                    mro_set = False

                df = pd.DataFrame(columns=["step", "Opt_val", "Eval_val", "Loss_val", "Violations", "A_norm"])

                # setup train and test data
                train, test = train_test_split(unc_set.data, test_size=int(unc_set.data.shape[0]/4), random_state=seed)
                val_dset = torch.tensor(train, requires_grad=True, dtype=torch.double)
                eval_set = torch.tensor(test, requires_grad=True, dtype=torch.double)
                # create cvxpylayer
                cvxpylayer = CvxpyLayer(prob, parameters=prob.parameters(), variables=self.variables())
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
                        paramb_tch = torch.tensor(-init@np.mean(train, axis=0), requires_grad=True, dtype=torch.double)
                    else:
                        if init_eps and init_A is None:
                            init = (1/init_eps)*np.eye(1)
                        elif init_A is not None:
                            init = np.array(init_A)
                            if init_eps:
                                init = (1/init_eps)*init
                        else:
                            init = np.array([[np.cov(train.T)]])
                        paramb_tch = torch.tensor(-init@np.mean(train, axis=0), requires_grad=True, dtype=torch.double)
                    if init_b is not None:
                        paramb_tch = torch.tensor(np.array(init_b), requires_grad=True, dtype=torch.double)

                    # ALPHA
                    alpha = torch.tensor(init_alpha, requires_grad=True)
                    paramT_tch = torch.tensor(init, requires_grad=True, dtype=torch.double)
                    # paramb_tch = paramT_tch@torch.tensor(-np.mean(train, axis=0), requires_grad=True)
                    if fixb or mro_set:
                        if mro_set and unc_set._uniqueA:
                            if init_A is None:
                                paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                            elif init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m):
                                paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                        paramT = paramT_tch.detach().numpy()
                        paramT_tch = torch.tensor(paramT, requires_grad=True, dtype=torch.double)
                        variables = [paramT_tch, alpha]
                    else:
                        variables = [paramT_tch, paramb_tch, alpha]

                    opt = OPTIMIZERS[optimizer](variables, lr=lr, momentum=momentum)
                    # opt = OPTIMIZERS[optimizer](variables, lr=lr)
                    if scheduler:
                        scheduler_ = torch.optim.lr_scheduler.ReduceLRonPlateau(opt, patience=5)

                    paramlst = prob.parameters()
                    newlst = []
                    if not mro_set:
                        for i in paramlst[:-2]:
                            newlst.append(torch.tensor(np.array(i.value).astype(
                                np.float), requires_grad=True, dtype=torch.double))
                        newlst.append(paramT_tch)
                        newlst.append(paramb_tch)
                    else:
                        for i in paramlst[:-1]:
                            newlst.append(torch.tensor(np.array(i.value).astype(
                                np.float), requires_grad=True, dtype=torch.double))
                        newlst.append(paramT_tch)

                    # train
                    lam = init_lam
                    mu = init_mu
                    for steps in range(step):
                        # import ipdb
                        # ipdb.set_trace()
                        if not mro_set:
                            newlst[-1] = paramb_tch
                            newlst[-2] = paramT_tch
                        else:
                            newlst[-1] = paramT_tch
                        var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})
                        temploss, obj, violations, cvar_update = unc_set.loss(
                            *var_values, alpha, val_dset,  mu, lam, target=target_cvar)
                        evalloss, obj2, violations2, var_vio = unc_set.loss(
                            *var_values,  alpha, eval_set, mu, lam, target=target_cvar)
                        lam = np.maximum(lam + mu*(cvar_update - target_cvar), 0)
                        mu = mu*mu_multiplier
                        temploss.backward()
                        newrow = pd.Series(
                            {"step": steps,
                             "Loss_val": temploss.item(),
                             "Eval_val": evalloss.item(),
                             "Opt_val": obj.item(),
                             "Test_val": obj2.item(),
                             "Violations": violations2.item(),
                             "Violation_val": var_vio.item(),
                             "Violation_train": cvar_update.item(),
                             "A_norm": np.linalg.norm(paramT_tch.detach().numpy().copy()),
                             "mu": mu,
                             "lam": lam,
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

                        if steps < step - 1:
                            opt.step()
                            opt.zero_grad()
                            if scheduler:
                                scheduler_.step(temploss)

                    self._trained = True
                    unc_set._trained = True
                    unc_set.paramT.value = paramT_tch.detach().numpy().copy()
                    if not mro_set:
                        unc_set.paramb.value = paramb_tch.detach().numpy().copy()

                else:
                    # import ipdb
                    # ipdb.set_trace()
                    #
                    if init_eps:
                        eps_tch = torch.tensor(1/np.array(init_eps), requires_grad=True, dtype=torch.double)
                        # eps_tch.grad = torch.tensor(0., dtype=torch.double)
                        if mro_set:
                            if unc_set._uniqueA and eps_tch.shape == torch.Size([]):
                                eps_tch = eps_tch.repeat(unc_set._K)
                                eps_tch = eps_tch.detach().numpy()
                                eps_tch = torch.tensor(eps_tch, requires_grad=True, dtype=torch.double)
                    else:
                        eps_tch = torch.tensor(1., requires_grad=True, dtype=torch.double)
                        if mro_set and unc_set._uniqueA:
                            eps_tch = eps_tch.repeat(unc_set._K)
                            eps_tch = eps_tch.detach().numpy()
                            eps_tch = torch.tensor(eps_tch, requires_grad=True, dtype=torch.double)
                    if init_A is not None:
                        init = torch.tensor(init_A, requires_grad=True, dtype=torch.double)
                    else:
                        init = torch.tensor(np.eye(train.shape[1]), requires_grad=True, dtype=torch.double)
                    if init_b is not None:
                        init_bval = torch.tensor(init_b, requires_grad=True, dtype=torch.double)
                    else:
                        init_bval = -init@torch.tensor(np.mean(train, axis=0), dtype=torch.double)
                    if not mro_set:
                        paramb_tch = eps_tch*init_bval
                        paramT_tch = eps_tch*init
                    elif unc_set._uniqueA:
                        if init_A is None or (init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m)):
                            paramT_tch = eps_tch[0]*init
                            for k_ind in range(1, unc_set._K):
                                paramT_tch = torch.vstack((paramT_tch, eps_tch[k_ind]*init))
                            case = 0
                        else:
                            paramT_tch = eps_tch[0]*torch.tensor(init_A[0:unc_set._m, 0:unc_set._m], dtype=torch.double)
                            for k_ind in range(1, unc_set._K):
                                paramT_tch = torch.vstack(
                                    (paramT_tch, eps_tch[k_ind] *
                                        torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)*unc_set._m, 0:unc_set._m],
                                                     dtype=torch.double)))
                            case = 1
                    else:
                        paramT_tch = eps_tch*init
                        case = 2
                    alpha = torch.tensor(init_alpha, requires_grad=True)
                    variables = [eps_tch, alpha]
                    opt = OPTIMIZERS[optimizer](variables, lr=lr, momentum=momentum)
                    # opt = OPTIMIZERS[optimizer](variables, lr=lr)
                    # opt = torch.optim.SGD(variables, lr=lr, momentum=.8)
                    if scheduler:
                        scheduler_ = torch.optim.lr_scheduler.ReduceLRonPlateau(opt, patience=5)
                    # assign parameter values
                    paramlst = prob.parameters()
                    newlst = []
                    if not mro_set:
                        for i in paramlst[:-2]:
                            newlst.append(torch.tensor(np.array(i.value).astype(
                                np.float), requires_grad=True, dtype=torch.double))
                        newlst.append(paramT_tch)
                        newlst.append(paramb_tch)
                    else:
                        for i in paramlst[:-1]:
                            newlst.append(torch.tensor(np.array(i.value).astype(
                                np.float), requires_grad=True, dtype=torch.double))
                        newlst.append(paramT_tch)

                    # train
                    mu = init_mu
                    lam = init_lam
                    for steps in range(step):
                        # import ipdb
                        # ipdb.set_trace()
                        if not mro_set:
                            newlst[-1] = eps_tch*init_bval
                            newlst[-2] = eps_tch*init
                        else:
                            if case == 0:
                                paramT_tch = eps_tch[0]*init
                                for k_ind in range(1, unc_set._K):
                                    paramT_tch = torch.vstack((paramT_tch, eps_tch[k_ind]*init))
                            elif case == 1:
                                paramT_tch = eps_tch[0] * \
                                    torch.tensor(init_A[0:unc_set._m, 0:unc_set._m], dtype=torch.double)
                                for k_ind in range(1, unc_set._K):
                                    paramT_tch = torch.vstack(
                                        (paramT_tch, eps_tch[k_ind] *
                                            torch.tensor(init_A[(k_ind*unc_set._m):(k_ind+1)
                                                                * unc_set._m, 0:unc_set._m], dtype=torch.double)))
                            else:
                                paramT_tch = eps_tch*init
                            newlst[-1] = paramT_tch
                        var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})
                        temploss, obj, violations, cvar_update = unc_set.loss(
                            *var_values,  alpha, val_dset, mu, lam, target=target_cvar)
                        evalloss, obj2, violations2, var_vio = unc_set.loss(
                            *var_values, alpha, eval_set, mu, lam, target=target_cvar)
                        lam = np.maximum(lam + mu*(cvar_update - target_cvar), 0)
                        mu = mu*mu_multiplier
                        temploss.backward()
                        # eps_tch = torch.maximum(eps_tch, torch.tensor(0.0001))
                        newrow = pd.Series(
                            {"step": steps,
                             "Loss_val": temploss.item(),
                             "Eval_val": evalloss.item(),
                             "Opt_val": obj.item(),
                             "Violations": violations2.item(),
                             "Violation_val": var_vio.item(),
                             "Violation_train": cvar_update.item(),
                             "Test_val": obj2.item(),
                             "A_norm": np.mean(1/eps_tch.detach().numpy().copy()),
                             "Eps_vals": 1/eps_tch.detach().numpy().copy(),
                             "mu": mu,
                             "lam": lam,
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
                                scheduler_.step(temploss)

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
                          unc_set.paramb.value, return_eps, obj.item(), var_values, T_iter=T_iter, b_iter=b_iter)
        else:
            return Result(self, prob, df, unc_set.paramT.value, None, return_eps, obj.item(), var_values,
                          T_iter=T_iter)

    def grid(self, epslst=None, seed=1, init_A=None, init_b=None, init_alpha=-0.01, solver: Optional[str] = None):
        r"""
        performs gridsearch to find optimal :math:`\epsilon`-ball around data with respect to user-defined loss

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

        # candidate_solvers = self._find_candidate_solvers(solver=solver, gp=False)
        # self._sort_candidate_solvers(candidate_solvers)
        # solving_chain = construct_solving_chain(self, candidate_solvers,  gp=False,
        #                                         enforce_dpp=True,
        #                                         ignore_dpp=False,
        #                                         # Comment this for now. Useful
        #                                         # in next cvxpy release
        #                                         solver_opts=None
        #                                         )
        # #
        if self.uncertain_parameters():
            # import ipdb
            # ipdb.set_trace()
            unc_set = self.uncertain_parameters()[0].uncertainty_set

            if unc_set.data is None:
                raise ValueError("Cannot train without uncertainty set data")

        #     new_reductions = solving_chain.reductions
        #     # Find position of Dcp2Cone or Qp2SymbolicQp
        #     for idx in range(len(new_reductions)):
        #         if type(new_reductions[idx]) in [Dcp2Cone, Qp2SymbolicQp]:
        #             # Insert RemoveUncertainParameters before those reductions
        #             new_reductions.insert(idx, RemoveUncertainParameters())
        #             unc_reductions = new_reductions[:idx+1]
        #             break
        # # return a chain instead (chain.apply, return the problem and inverse data)
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
                train, test = train_test_split(unc_set.data, test_size=int(unc_set.data.shape[0]/4), random_state=seed)
                val_dset = torch.tensor(train, requires_grad=True, dtype=torch.double)
                eval_set = torch.tensor(test, requires_grad=True, dtype=torch.double)
                # create cvxpylayer
                cvxpylayer = CvxpyLayer(prob, parameters=prob.parameters(), variables=self.variables())
                eps_tch = torch.tensor([[epslst[0]]], requires_grad=True, dtype=torch.double)
                paramb_tch = eps_tch[0][0]*torch.tensor(-np.mean(train, axis=0), requires_grad=True, dtype=torch.double)
                paramT_tch = eps_tch[0][0]*torch.tensor(np.eye(train.shape[1]), requires_grad=True, dtype=torch.double)
                # assign parameter values
                paramlst = prob.parameters()
                newlst = []
                if not mro_set:
                    for i in paramlst[:-2]:
                        newlst.append(torch.tensor(np.array(i.value).astype(
                            np.float), requires_grad=True), dtype=torch.double)
                    newlst.append(paramT_tch)
                    newlst.append(paramb_tch)
                else:
                    for i in paramlst[:-1]:
                        newlst.append(torch.tensor(np.array(i.value).astype(
                            np.float), requires_grad=True), dtype=torch.double)
                    newlst.append(paramT_tch)
                minval = 9999999
                var_vals = 0

                if init_A is not None:
                    init = torch.tensor(init_A, requires_grad=True, dtype=torch.double)
                else:
                    init = torch.tensor(np.eye(train.shape[1]), requires_grad=True, dtype=torch.double)
                if init_b is not None:
                    init_bval = torch.tensor(init_b, requires_grad=True, dtype=torch.double)
                else:
                    init_bval = torch.tensor(-np.mean(train, axis=0), requires_grad=True, dtype=torch.double)

                for epss in epslst:
                    # import ipdb
                    eps_tch1 = torch.tensor([[1/epss]], requires_grad=True, dtype=torch.double)
                    # ipdb.set_trace()
                    if not mro_set:
                        newlst[-1] = eps_tch1[0][0]*init_bval
                        newlst[-2] = eps_tch1[0][0]*init
                    else:
                        if unc_set._uniqueA:
                            if init_A is None or (init_A is not None and init_A.shape[0] != (unc_set._K*unc_set._m)):
                                paramT_tch = eps_tch1[0][0]*init
                                paramT_tch = paramT_tch.repeat(unc_set._K, 1)
                            else:
                                paramT_tch = eps_tch1[0][0]*init
                        else:
                            paramT_tch = eps_tch1[0][0]*init
                        newlst[-1] = paramT_tch
                    var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})
                    temploss, obj, violations, cvar_update = unc_set.loss(
                        *var_values, torch.tensor(init_alpha), val_dset)
                    evalloss, obj2, violations2, var_vio = unc_set.loss(*var_values, torch.tensor(init_alpha), eval_set)
                    if temploss <= minval:
                        minval = temploss
                        mineps = eps_tch1.clone()
                        minT = paramT_tch.clone()
                        var_vals = var_values
                    newrow = pd.Series(
                        {"Loss_val": temploss.item(),
                            "Eval_val": evalloss.item(),
                            "Opt_val": obj.item(),
                            "Test_val": obj2.item(),
                            "Violations": violations2.item(),
                            "Violation_val": var_vio.item(),
                            "Violation_train": cvar_update.item(),
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
                          unc_set.paramb.value, mineps[0][0].detach().numpy().copy(), minval, var_vals)
        else:
            return Result(self, prob, df, unc_set.paramT.value,
                          None, mineps[0][0].detach().numpy().copy(), minval, var_vals)

    def dualize_constraints(self):
        # import ipdb
        # ipdb.set_trace()
        if self.uncertain_parameters():
            unc_reductions = []
            if type(self.objective) == Maximize:
                unc_reductions += [FlipObjective()]
            # Add separate uncertainty
            unc_reductions += [Separate_Uncertain_Params()]
            unc_reductions += [RemoveUncertainParameters()]
            newchain = UncertainChain(self, reductions=unc_reductions)
            prob, _ = newchain.apply(self)
            return prob
        return super(RobustProblem, self)

    def solve(self, solver: Optional[str] = None):
        # import ipdb
        # ipdb.set_trace()
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
            # unc_reductions = []
            # if type(self.objective) == Maximize:
            #     unc_reductions += [FlipObjective()]
            # unc_reductions += [RemoveUncertainParameters()]
            # newchain = UncertainChain(self, reductions=unc_reductions)
            # prob, _ = newchain.apply(self)

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
