from cvxpy import Variable, problems
from cvxpy import sum as cp_sum
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.separate_uncertainty import \
    SEPARATION_METHODS as sep_methods
from lropt.utils import unique_list


class Separate_Uncertain_Params(Reduction):

    def __init__(self, problem=None) -> None:
        super(Separate_Uncertain_Params, self).__init__(problem=problem)

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint."""
        inverse_data = InverseData(problem)
        canon_objective, canon_constraints = problem.objective, []

        for constraint in problem.constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # ipdb.set_trace()
            if self.has_unc_param(constraint):

                unique_unc_params = self.get_unq_uncertain_param(constraint)
                num_unc_params = len(unique_unc_params)

                unc_dict = {}
                for unc_param in unique_unc_params:
                    unc_dict[unc_param] = 0
                unc_epi = Variable(num_unc_params)

                unc_lst, std_lst, is_max = self.separate_uncertainty(constraint)

                assert (is_max == 0)
                original_constraint = 0

                for unc_function in unc_lst:
                    param_lst = self.get_unq_uncertain_param(unc_function)
                    # assert len(param_lst) == 1, "two different parameters multiplied violates lropt ruleset"
                    unc_param = param_lst[0]

                    unc_dict[unc_param] += unc_function

                for i, unc_param in enumerate(unique_unc_params):
                    canon_constraints += [unc_dict[unc_param] <= unc_epi[i]]

                original_constraint += sum(std_lst) + cp_sum(unc_epi)

                canon_constr = original_constraint <= 0
                canon_constraints += [canon_constr]

            else:
                canon_constr = constraint
                canon_constraints += [canon_constr]

            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

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

    def get_unq_uncertain_param(self, expr):
        return unique_list([v for v in expr.parameters()
                            if isinstance(v, UncertainParameter)])

    def count_unq_uncertain_param(self, expr):
        return len(self.get_unq_uncertain_param(expr))

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
