from cvxpy import Variable, problems
from cvxpy.atoms.affine import Sum
# from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.unary_operators import NegExpression
# Type Checking
from cvxpy.constraints.nonpos import Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from lro.uncertain import UncertainParameter
from lro.utils import unique_list


class Uncertain_Canonicalization(Reduction):
    """Recursively canonicalize each expression in a problem.
    This reduction recursively canonicalizes every expression tree in a
    problem, visiting each node. At every node, this reduction first
    canonicalizes its arguments; it then canonicalizes the node, using the
    canonicalized arguments.
    The attribute `canon_methods` is a dictionary
    mapping node types to functions that canonicalize them; the signature
    of these canonicalizing functions must be
        def canon_func(expr, canon_args) --> (new_expr, constraints)
    where `expr` is the `Expression` (node) to canonicalize, canon_args
    is a list of the canonicalized arguments of this expression,
    `new_expr` is a canonicalized expression, and `constraints` is a list
    of constraints introduced while canonicalizing `expr`.
    Attributes:
    ----------
        canon_methods : dict
            A dictionary mapping node types to canonicalization functions.
        problem : Problem
            A problem owned by this reduction.
    """

    def __init__(self, canon_methods, problem=None) -> None:
        super(Uncertain_Canonicalization, self).__init__(problem=problem)
        self.canon_methods = canon_methods

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint."""
        inverse_data = InverseData(problem)

        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective)

        for constraint in problem.constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint

            # if self.has_unc_param(constraint):
            # import ipdb
            # ipdb.set_trace()
            # unc_lst, std_lst = self.separate_uncertainty(constraint)
            # result_expr = self.remove_uncertainty(unc_lst, std_lst)

            canon_constr, aux_constr = self.canonicalize_tree(
                constraint, 0)
            canon_constraints += aux_constr + [canon_constr]
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

    def canonicalize_tree(self, expr, var):
        """Recursively canonicalize an Expression."""
        # TODO don't copy affine expressions?
        if type(expr) == cvxtypes.partial_problem():
            canon_expr, constrs = self.canonicalize_tree(
                expr.args[0].objective.expr, var)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(constr, var)
                constrs += [canon_constr] + aux_constr
        else:
            canon_args = []
            constrs = []
            for arg in expr.args:
                canon_arg, c = self.canonicalize_tree(arg, var)
                canon_args += [canon_arg]
                constrs += c
            canon_expr, c = self.canonicalize_expr(expr, canon_args, var)
            constrs += c
        return canon_expr, constrs

    def canonicalize_expr(self, expr, args, var):
        """Canonicalize an expression, w.r.t. canonicalized arguments."""
        # Constant trees are collapsed, but parameter trees are preserved.
        if isinstance(expr, Expression) and (
                expr.is_constant() and not expr.parameters()):
            return expr, []
        elif type(expr) in self.canon_methods:
            return self.canon_methods[type(expr)](expr, args, var)
        else:
            return expr.copy(args), []

    def canon_tree_uncertain(self, u_list, uvar, n_list):
        num = len(u_list)
        if len(uvar.shape) >= 1:
            z = Variable((num+1, uvar.shape([0])))
        else:
            z = Variable(num+1)
        aux_const = [Sum(z, axis=1) == 0]
        aux_expr, new_const = uvar.conjugate(z[num])
        aux_const += new_const
        for ind in range(len(u_list)):
            # if len(u_list[ind].variables()) (check if has variable)

            new_expr, new_const = self.canonicalize_tree(u_list[ind], z[ind])
            aux_expr += new_expr
            aux_const += new_const
        for expr in n_list:
            aux_expr += expr
        return aux_expr, aux_const

    def count_unq_uncertain_param(self, expr):
        unc_params = []
        unc_params += [v for v in expr.parameters()
                       if isinstance(v, UncertainParameter)]

        return len(unique_list(unc_params))

    def has_unc_param(self, expr):
        return self.count_unq_uncertain_param(expr) == 1

    def separate_uncertainty(self, expr):
        '''takes in a constraint or expression and returns 3 lists:
            unc_lst :
                EX: [g_1(u_1,x), g_2(u_1,x)]
                a list of lists corresponding to the number of
                unique uncertain parameters in the expression
            std_lst :
                Ex: [h_1(x),h_2(x)]
                any other functions without uncertainty
            '''

        # Check Initial Conditions
        if isinstance(expr, Inequality):
            return self.separate_uncertainty(expr.args[0] - expr.args[1])
        elif self.count_unq_uncertain_param(expr) == 0:
            return ([], [expr])
        elif self.count_unq_uncertain_param(expr) > 1:
            raise ValueError("DRP error: Cannot have multiple uncertain params in the same expr")
        elif len(expr.args) == 0:
            assert (self.has_unc_param(expr))
            return ([expr], [])

        elif isinstance(expr, multiply):
            if self.has_unc_param(expr.args[0]) and self.has_unc_param(expr.args[1]):
                raise ValueError("DRP error: Cannot have uncertainty multiplied by each other")
            c = 1
            if self.has_unc_param(expr.args[0]):
                unc_lst, std_lst = self.separate_uncertainty(expr.args[0])
                if isinstance(expr.args[1], NegExpression):
                    c = -1
                    expr.args[1] = expr.args[1].args[0]
                elif isinstance(expr.args[1], Constant):
                    c = expr.args[1].value
                    expr.args[1] = 1
                new_unc_lst = [c * g_u * expr.args[1] for g_u in unc_lst]
                new_std_lst = [c * h_x * expr.args[1] for h_x in std_lst]
                return (new_unc_lst, new_std_lst)
            else:
                unc_lst, std_lst = self.separate_uncertainty(expr.args[1])
                if isinstance(expr.args[0], NegExpression):
                    c = -1
                    expr.args[0] = expr.args[0].args[0]
                elif isinstance(expr.args[0], Constant):
                    c = expr.args[0].value
                    expr.args[0] = 1
                new_unc_lst = [c * expr.args[0] * g_u for g_u in unc_lst]
                new_std_lst = [c * expr.args[0] * h_x for h_x in std_lst]
                return (new_unc_lst, new_std_lst)

        elif isinstance(expr, MulExpression):
            import ipdb
            ipdb.set_trace()
            if self.has_unc_param(expr.args[0]) and self.has_unc_param(expr.args[1]):
                raise ValueError("DRP error: Cannot have uncertainty multiplied by each other")
            c = 1
            if self.has_unc_param(expr.args[0]):
                unc_lst, std_lst = self.separate_uncertainty(expr.args[0])
                if isinstance(expr.args[1], NegExpression):
                    c = -1
                    expr.args[1] = expr.args[1].args[0]
                elif isinstance(expr.args[1], Constant):
                    c = expr.args[1].value
                    expr.args[1] = 1
                new_unc_lst = [c * g_u @ expr.args[1] for g_u in unc_lst]
                new_std_lst = [c * h_x @ expr.args[1] for h_x in std_lst]
                return (new_unc_lst, new_std_lst)
            else:
                unc_lst, std_lst = self.separate_uncertainty(expr.args[1])
                if isinstance(expr.args[0], NegExpression):
                    c = -1
                    expr.args[0] = expr.args[0].args[0]
                elif isinstance(expr.args[0], Constant):
                    c = expr.args[0].value
                    expr.args[0] = 1
                new_unc_lst = [c * expr.args[0] @ g_u for g_u in unc_lst]
                new_std_lst = [c * expr.args[0] @ h_x for h_x in std_lst]
                return (new_unc_lst, new_std_lst)

        elif isinstance(expr, AddExpression):
            if self.has_unc_param(expr.args[0]) and self.has_unc_param(expr.args[1]):
                unc_lst_0, std_lst_0 = self.separate_uncertainty(expr.args[0])
                unc_lst_1, std_lst_1 = self.separate_uncertainty(expr.args[1])
                return (unc_lst_0 + unc_lst_1, std_lst_0 + std_lst_1)
            elif self.has_unc_param(expr.args[0]):
                unc_lst, std_lst = self.separate_uncertainty(expr.args[0])
                std_lst.append(expr.args[1])
                return (unc_lst, std_lst)
            else:
                unc_lst, std_lst = self.separate_uncertainty(expr.args[1])
                std_lst.append(expr.args[0])
                return (unc_lst, std_lst)
        elif isinstance(expr, NegExpression):
            return ([-1 * expr.args[0]], [])
        else:
            raise ValueError("DRP error: not able to process non multiplication/additions")
