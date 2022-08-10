from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Maximize
from cvxpy.error import SolverError, DCPError, DGPError
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.reductions import (Chain, Dcp2Cone,
                              FlipObjective,
                              Dgp2Dcp,  # Qp2SymbolicQp,
                              CvxAttr2Constr,
                              Complex2Real
                              )
from cvxpy.reductions.complex2real import complex2real
#  from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from rcvx.remove_uncertain.remove_uncertain import RemoveUncertainParameters
from rcvx.uncertain import UncertainParameter


def construct_robust_intermediate_chain(problem, candidates, gp=False):
    """
    Builds a chain that rewrites a problem into an intermediate
    representation suitable for numeric reductions.

    This chain includes a reduction to bring the robust
    problem into convex form.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    candidates : dict
        Dictionary of candidate solvers divided in qp_solvers
        and conic_solvers.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.

    Returns
    -------
    Chain
        A Chain that can be used to convert the problem to
        an intermediate form.

    Raises
    ------
    DCPError
        Raised if the problem is not DCP and `gp` is False.
    DGPError
        Raised if the problem is not DGP and `gp` is True.
    """

    reductions = []
    if len(problem.variables()) == 0:
        return Chain(reductions=reductions)

    # TODO Reduce boolean constraints.
    # TODO: Reenable complex2real
    if complex2real.accepts(problem):
        reductions += [Complex2Real()]

    if gp:
        reductions += [Dgp2Dcp()]

    if not gp and not problem.is_dcp():
        append = ""
        append = (" However, the problem does follow DGP rules. "
                  "Consider calling this function with `gp=True`.")
        raise DCPError("Problem does not follow DCP rules." + append)

    elif gp and not problem.is_dgp():
        append = ""
        if problem.is_dcp():
            append = (" However, the problem does follow DCP rules. "
                      "Consider calling this function with `gp=False`.")
        raise DGPError("Problem does not follow DGP rules." + append)

    # Dcp2Cone and Qp2SymbolicQp require problems to minimize their objectives.
    if type(problem.objective) == Maximize:
        reductions += [FlipObjective()]

    #  if problem.uncertain_parameters():
    #      reductions += [RemoveUncertainParameters()]

    #  # First, attempt to canonicalize the problem to a linearly constrained QP.
    #  if candidates['qp_solvers'] and qp2symbolic_qp.accepts(problem):
    #      reductions += [CvxAttr2Constr(),
    #                     Qp2SymbolicQp()]
    #      return Chain(reductions=reductions)

    # Canonicalize it to conic problem.
    if not candidates['conic_solvers']:
        raise SolverError("Problem could not be reduced to a QP, and no "
                          "conic solvers exist among candidate solvers "
                          "(%s)." % candidates)
    reductions += [Dcp2Cone()]
    if problem.uncertain_parameters():
        reductions += [RemoveUncertainParameters()]
    reductions += [Dcp2Cone(),
                   CvxAttr2Constr()]

    return Chain(reductions=reductions)


class RobustProblem(Problem):
    """Problem with uncertain variables"""

    def uncertain_parameters(self):
        """Find which variables are uncertain"""
        unc_params = []
        # TODO: Add also in cost
        for c in self.constraints:
            unc_params += [v for v in c.parameters()
                           if isinstance(v, UncertainParameter)]

        # Pick unique elements
        seen = set()
        return [seen.add(obj.id) or obj
                for obj in unc_params if obj.id not in seen]

    def _construct_chains(self, solver=None, gp=False):
        """
        Construct the chains required to reformulate and solve the problem.

        In particular, this function

        #. finds the candidate solvers
        #. constructs the intermediate chain suitable for numeric reductions.
        #. constructs the solving chain that performs the
           numeric reductions and solves the problem.

        Parameters
        ----------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        gp : bool, optional
            If True, the problem is parsed as a Disciplined Geometric Program
            instead of as a Disciplined Convex Program.
        """

        chain_key = (solver, gp)

        if chain_key != self._cached_chain_key:
            try:
                candidate_solvers = self._find_candidate_solvers(solver=solver,
                                                                 gp=gp)

                self._intermediate_chain = \
                    construct_robust_intermediate_chain(self,
                                                        candidate_solvers,
                                                        gp=gp)
                self._intermediate_problem, self._intermediate_inverse_data = \
                    self._intermediate_chain.apply(self)

                self._solving_chain = \
                    construct_solving_chain(self._intermediate_problem,
                                            candidate_solvers)

                self._cached_chain_key = chain_key

            except Exception as e:
                raise e
