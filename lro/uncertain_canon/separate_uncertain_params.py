from cvxpy import Variable, problems
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.constraints.nonpos import Inequality
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from lro.uncertain import UncertainParameter
from lro.uncertain_canon.separate_uncertainty import \
    SEPARATION_METHODS as sep_methods
from lro.utils import unique_list


class Separate_Uncertain_Params(Reduction):

    def __init__(self, problem=None) -> None:
        super(Separate_Uncertain_Params, self).__init__(problem=problem)

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint."""
        inverse_data = InverseData(problem)
        # import ipdb
        # ipdb.set_trace()
        canon_objective, canon_constraints = problem.objective, []

        for constraint in problem.constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # ipdb.set_trace()
            if self.has_unc_param(constraint):

                unc_lst, std_lst, is_max = self.separate_uncertainty(constraint)

                assert (is_max == 0)
                original_constraint = 0

                for unc_function in unc_lst:
                    unc_var = Variable()
                    canon_constraints += [unc_function <= unc_var]

                    original_constraint += unc_var

                for std_function in std_lst:
                    original_constraint += std_function

                canon_constr = original_constraint <= 0
                canon_constraints += [canon_constr]

            else:
                # canon_constr, aux_constr = self.canonicalize_tree(
                #     constraint, 0)
                canon_constr = constraint
                canon_constraints += [canon_constr]

            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

        # import ipdb
        # ipdb.set_trace()

        new_problem = problems.problem.Problem(canon_objective,
                                               canon_constraints)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}

        return Solution(solution.status, solution.opt_val, pvars, dvars,
                        solution.attr)

    def count_unq_uncertain_param(self, expr):
        unc_params = []
        if isinstance(expr, Inequality):
            unc_params += [v for v in expr.parameters()
                           if isinstance(v, UncertainParameter)]
            return len(unique_list(unc_params))

        else:
            unc_params += [v for v in expr.parameters()
                           if isinstance(v, UncertainParameter)]
        return len(unique_list(unc_params))

    def has_unc_param(self, expr):
        if not isinstance(expr, int) and not isinstance(expr, float):
            return self.count_unq_uncertain_param(expr) >= 1
        else:
            return 0

    def separate_uncertainty(self, expr):
        '''separate cvxpy expression into subexpressions with uncertain parameters and without.
        Input:
            expr :
                a cvxpy expression
        Output:
            unc_lst :
                EX: :math:`[g_1(u_1,x), g_2(u_1,x)]`
                a list of cvxpy multiplication expressions from expr each containing one uncertain parameter
            std_lst :
                Ex: :math:`[h_1(x),h_2(x)]`
                any other cvxpy expressions

        The original expr is equivalent to the sum of expressions in unc_lst and std_lst
            '''
        if self.count_unq_uncertain_param(expr) == 0:
            return [], [expr], 0

        elif len(expr.args) == 0:
            assert (self.has_unc_param(expr))
            return [expr], [], 0

        elif type(expr) not in sep_methods:
            raise ValueError("DRP error: not able to process non multiplication/additions")

        if type(expr) == maximum:
            return [expr], [], 0

        func = sep_methods[type(expr)]
        return func(self, expr)