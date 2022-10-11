from typing import Optional

import numpy as np
import torch
from cvxpy.problems.problem import Problem
from cvxpy.reductions import Dcp2Cone, Qp2SymbolicQp
# from cvxpy.reductions.chain import Chain
from cvxpy.reductions.solvers.solving_chain import (SolvingChain,
                                                    construct_solving_chain)
from cvxpylayers.torch import CvxpyLayer

from lro.remove_uncertain.remove_uncertain import RemoveUncertainParameters
from lro.uncertain import UncertainParameter
from lro.uncertain_canon.uncertain_chain import UncertainChain
from lro.utils import unique_list


class RobustProblem(Problem):
    """Problem with uncertain variables"""

    def uncertain_parameters(self):
        """Find which variables are uncertain"""
        unc_params = []
        # TODO: Add also in cost
        for c in self.constraints:
            unc_params += [v for v in c.parameters()
                           if isinstance(v, UncertainParameter)]

        return unique_list(unc_params)
# create function for only remove_uncertain reduction

    def _construct_chain_old(
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
                    unc_reductions = new_reductions[:idx+1]
                    break
        # return a chain instead (chain.apply, return the problem and inverse data)

        newchain = UncertainChain(self, reductions=unc_reductions)
        prob, inverse_data, newparam = newchain.apply(self)
        if newparam is not None:
            # create cvxpylayer
            cvxpylayer = CvxpyLayer(prob, parameters=prob.parameters(), variables=prob.variables())

            # designate parameter to train
            dim = newparam.shape[0]
            paramT_tch = torch.tensor(np.eye(dim), requires_grad=True)
            variables = [paramT_tch]
            opt = torch.optim.SGD(variables, lr=.02, momentum=.8)

            # assign parameter values
            paramlst = prob.parameters()
            newlst = []
            for i in paramlst[:-1]:
                newlst.append(torch.tensor(i.value, requires_grad=True))
            newlst.append(paramT_tch)

            # solve
            var_values = cvxpylayer(*newlst, solver_args={'solve_method': 'ECOS'})

            # loss function?
            for i in range(len(prob.variables())):
                prob.variables[i].value = var_values[i].value
                prob.objective.value
                opt
        return SolvingChain(reductions=new_reductions)
