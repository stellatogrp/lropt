from typing import Optional

import numpy as np
import pandas as pd
# import scipy as sc
import torch
from cvxpy.problems.problem import Problem
from cvxpy.reductions import Dcp2Cone, Qp2SymbolicQp
# from cvxpy.reductions.chain import Chain
from cvxpy.reductions.solvers.solving_chain import (SolvingChain,
                                                    construct_solving_chain)
from cvxpylayers.torch import CvxpyLayer
from sklearn.model_selection import train_test_split

from lro.remove_uncertain.remove_uncertain import RemoveUncertainParameters
from lro.uncertain import UncertainParameter
from lro.uncertain_canon.uncertain_chain import UncertainChain
from lro.utils import unique_list


class RobustProblem(Problem):
    """Problem with uncertain variables"""

    def __init__(self, objective, constraints):
        self._trained = False
        self._values = None
        super(RobustProblem, self).__init__(objective, constraints)
        self._trained = False
        self._values = None

    @property
    def trained(self):
        return self._trained

    @property
    def param_values(self):
        return self._values

    def uncertain_parameters(self):
        """Find which variables are uncertain"""
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
        solver_opts: Optional[dict] = None
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
                                                # Comment this for now. Useful
                                                # in next cvxpy release
                                                solver_opts=solver_opts
                                                )
        #
        if self.uncertain_parameters():
            # import ipdb
            # ipdb.set_trace()
            new_reductions = solving_chain.reductions
            # Find position of Dcp2Cone or Qp2SymbolicQp
            for idx in range(len(new_reductions)):
                if type(new_reductions[idx]) in [Dcp2Cone, Qp2SymbolicQp]:
                    # Insert RemoveUncertainParameters before those reductions
                    new_reductions.insert(idx, RemoveUncertainParameters())
                    break
        # return a chain instead (chain.apply, return the problem and inverse data)
        return SolvingChain(reductions=new_reductions)

    def train(
        self, eps=False, step=45, lr=0.001, solver: Optional[str] = None, gp: bool = False,
        enforce_dpp: bool = True, ignore_dpp: bool = False,
        solver_opts: Optional[dict] = None
    ):
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
                                                # Comment this for now. Useful
                                                # in next cvxpy release
                                                solver_opts=solver_opts
                                                )
        #
        if self.uncertain_parameters():
            # import ipdb
            # ipdb.set_trace()
            unc_set = self.uncertain_parameters()[0].uncertainty_set
            new_reductions = solving_chain.reductions
            # Find position of Dcp2Cone or Qp2SymbolicQp
            for idx in range(len(new_reductions)):
                if type(new_reductions[idx]) in [Dcp2Cone, Qp2SymbolicQp]:
                    # Insert RemoveUncertainParameters before those reductions
                    new_reductions.insert(idx, RemoveUncertainParameters())
                    unc_reductions = new_reductions[:idx+1]
                    break
        # return a chain instead (chain.apply, return the problem and inverse data)

            newchain = UncertainChain(self, reductions=unc_reductions)
            prob, inverse_data = newchain.apply(self)
            if unc_set.paramT is not None:
                df = pd.DataFrame(columns=["steps", "Opt_val", "Eval_val", "Loss_val", "Rnorm"])
                df1 = pd.DataFrame(columns=["steps", "Opt_val", "Eval_val", "Loss_val", "Rnorm"])

                # setup train and test data
                train, test = train_test_split(unc_set.data, test_size=int(unc_set.data.shape[0]/5))
                val_dset = torch.tensor(train, requires_grad=True)
                eval_set = torch.tensor(test, requires_grad=True)
                # create cvxpylayer
                cvxpylayer = CvxpyLayer(prob, parameters=prob.parameters(), variables=self.variables())
                if not eps:
                    # initialize parameters to train
                    if len(np.shape(np.cov(train.T))) >= 1:
                        # init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(train.T)))
                        init = 0.2*np.eye(train.shape[1])
                        paramb_tch = torch.tensor(init@np.mean(train, axis=0), requires_grad=True)
                    else:
                        init = np.array([[np.cov(train.T)]])
                        paramb_tch = torch.tensor(init@np.mean(train, axis=0), requires_grad=True)

                    paramT_tch = torch.tensor(init, requires_grad=True)
                    variables = [paramT_tch, paramb_tch]
                    opt = torch.optim.SGD(variables, lr=lr, momentum=.8)

                    paramlst = prob.parameters()
                    newlst = []
                    for i in paramlst[:-2]:
                        newlst.append(torch.tensor(np.array(i.value).astype(np.float), requires_grad=True))
                    newlst.append(paramT_tch)
                    newlst.append(paramb_tch)

                    # train
                    for steps in range(step):
                        # import ipdb
                        # ipdb.set_trace()
                        totloss = 0
                        objv = 0
                        splits = 1
                        for sets in range(splits):
                            var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})
                            temploss, obj = unc_set.loss(*var_values, val_dset)
                            evalloss, _ = unc_set.loss(*var_values, eval_set)
                            objv += obj
                            totloss += temploss
                        totloss = totloss/splits
                        totloss.backward()
                        newrow = pd.Series(
                            {"steps": steps,
                             "Loss_val": totloss.item(),
                             "Eval_val": evalloss.item(),
                             "Opt_val": objv.item()/splits,
                             "Rnorm": np.linalg.norm(paramT_tch.detach().numpy().copy())
                             })
                        df = df.append(newrow, ignore_index=True)
                        if steps < step - 1:
                            opt.step()
                            opt.zero_grad()
                    self._values = {'T': paramT_tch.detach().numpy().copy(), 'b': paramb_tch.detach().numpy().copy()}
                    self._trained = True
                    unc_set._trained = True
                    var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})

                else:
                    # import ipdb
                    # ipdb.set_trace()
                    #
                    # eps_tch = torch.tensor([[1/np.mean(np.cov(train.T))]], requires_grad=True)
                    eps_tch = torch.tensor([[1/5.]], requires_grad=True)
                    paramb_tch = eps_tch[0]*torch.tensor(np.mean(train, axis=0), requires_grad=True)
                    paramT_tch = eps_tch*torch.tensor(np.eye(train.shape[1]), requires_grad=True)
                    variables = [eps_tch]
                    opt = torch.optim.SGD(variables, lr=lr, momentum=.8)

                    # assign parameter values
                    paramlst = prob.parameters()
                    newlst = []
                    for i in paramlst[:-2]:
                        newlst.append(torch.tensor(np.array(i.value).astype(np.float), requires_grad=True))
                    newlst.append(paramT_tch)
                    newlst.append(paramb_tch)

                    # train
                    for steps in range(step):
                        # import ipdb
                        # ipdb.set_trace()
                        totloss = 0
                        objv = 0
                        splits = 1
                        for sets in range(splits):
                            newlst[-1] = eps_tch[0]*torch.tensor(np.mean(train, axis=0), requires_grad=True)
                            newlst[-2] = eps_tch*torch.tensor(np.eye(train.shape[1]), requires_grad=True)
                            var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})
                            temploss, obj = unc_set.loss(*var_values, val_dset)
                            evalloss, _ = unc_set.loss(*var_values, eval_set)
                            objv += obj
                            totloss += temploss
                        totloss = totloss/splits
                        totloss.backward()
                        newrow = pd.Series(
                            {"steps": steps,
                             "Loss_val": totloss.item(),
                             "Eval_val": evalloss.item(),
                             "Opt_val": objv.item()/splits,
                             "Rnorm": eps_tch[0][0].detach().numpy().copy()
                             })
                        df = df.append(newrow, ignore_index=True)
                        if steps < step - 1:
                            opt.step()
                            opt.zero_grad()

                    eps_tch = torch.tensor([[1/5.]], requires_grad=True)
                    paramb_tch = eps_tch[0]*torch.tensor(np.mean(train, axis=0), requires_grad=True)
                    paramT_tch = eps_tch*torch.tensor(np.eye(train.shape[1]), requires_grad=True)

                    # assign parameter values
                    paramlst = prob.parameters()
                    newlst = []
                    for i in paramlst[:-2]:
                        newlst.append(torch.tensor(np.array(i.value).astype(np.float), requires_grad=True))
                    newlst.append(paramT_tch)
                    newlst.append(paramb_tch)
                    epslst = np.linspace(0.1, 5, 20)
                    # train
                    for epss in epslst:
                        # import ipdb
                        eps_tch = torch.tensor([[epss]], requires_grad=True)
                        # ipdb.set_trace()
                        totloss = 0
                        objv = 0
                        splits = 1
                        for sets in range(splits):
                            newlst[-1] = eps_tch[0]*torch.tensor(np.mean(train, axis=0), requires_grad=True)
                            newlst[-2] = eps_tch*torch.tensor(np.eye(train.shape[1]), requires_grad=True)
                            var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})
                            temploss, obj = unc_set.loss(*var_values, val_dset)
                            evalloss, _ = unc_set.loss(*var_values, eval_set)
                            objv += obj
                            totloss += temploss
                        totloss = totloss/splits
                        newrow = pd.Series(
                            {"steps": steps,
                             "Loss_val": totloss.item(),
                             "Eval_val": evalloss.item(),
                             "Opt_val": objv.item()/splits,
                             "Rnorm": eps_tch[0][0].detach().numpy().copy()
                             })
                        df1 = df1.append(newrow, ignore_index=True)
                    self._values = {'T': (eps_tch*torch.tensor(np.eye(train.shape[1]))).detach().numpy().copy(), 'b': (
                        eps_tch[0]*torch.tensor(np.mean(train, axis=0))).detach().numpy().copy()}
                    self._trained = True
                    unc_set._trained = True
        return df, df1
