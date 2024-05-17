import cvxpy as cp
from cvxpy import SCS, Variable
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction

from lropt.robust_problem import RobustProblem
from lropt.uncertain_canon.max_of_uncertain import sum_of_max_of_uncertain
from lropt.uncertain_canon.utils import standard_invert, unique_list
from lropt.uncertain_parameter import UncertainParameter


class RemoveSumOfMaxOfUncertain(Reduction):
    def apply(self, problem: RobustProblem, solver=SCS):
        """Removes sum_of_max_of_uncertain constraints by creating
        a copy of the constraint for each term in the maximum."""
        def _gen_objective_constraints(problem):
            """
            This function generates canon objective and new constraints
            to deal with uncertainty in the objective
            """
            if self.has_unc_param(problem.objective.expr):
                epigraph_obj = Variable()
                epi_cons = problem.objective.expr <= epigraph_obj
                new_constraints = [epi_cons] + problem.constraints
                canon_objective = cp.Minimize(epigraph_obj)
            else:
                canon_objective = problem.objective
                new_constraints = problem.constraints
            return canon_objective, new_constraints


        problem = RobustProblem(problem.objective, problem.constraints)
        inverse_data = InverseData(problem)

        canon_objective, new_constraints = _gen_objective_constraints(problem)
        epigraph_problem = RobustProblem(canon_objective,new_constraints)

        new_constraints = []
        for constraint in epigraph_problem.constraints:
            if isinstance(constraint, sum_of_max_of_uncertain):
                for arg in constraint.args:
                    canon_constr = arg <= 0
                    new_constraints += [canon_constr]
                    inverse_data.cons_id_map.update({constraint.id: canon_constr.id})
            else:
                new_constraints += [constraint]

        new_problem = RobustProblem(objective=epigraph_problem.objective, \
                                    constraints=new_constraints)

        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        return standard_invert(solution=solution, inverse_data=inverse_data)

    def count_unq_uncertain_param(self, expr) -> int:
        unc_params = [v for v in expr.parameters() if isinstance(v, UncertainParameter)]
        return len(unique_list(unc_params))

    def has_unc_param(self, expr) -> bool:
        if not isinstance(expr, int) and not isinstance(expr, float):
            return self.count_unq_uncertain_param(expr) >= 1
        else:
            return False
