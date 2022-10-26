from cvxpy import Variable, problems
# from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression
# Type Checking
from cvxpy.constraints.nonpos import Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from lro.remove_uncertain.atom_canonicalizers.mul_canon import \
    mul_canon_transform
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
        # import ipdb
        # ipdb.set_trace()
        canon_objective, canon_constraints = self.canonicalize_tree(
            problem.objective, 0)

        for constraint in problem.constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint
            # import ipdb
            # ipdb.set_trace()
            if self.has_unc_param(constraint):
                # import ipdb
                # ipdb.set_trace()
                unc_lst, std_lst = self.separate_uncertainty(constraint)
                unc_params = []
                unc_params += [v for v in unc_lst[0].parameters()
                               if isinstance(v, UncertainParameter)]
                canon_constr, aux_constr = self.remove_uncertainty(unc_lst, unc_params[0], std_lst)
                canon_constraints += aux_constr + [canon_constr]
                # import ipdb
                # ipdb.set_trace()
                # if unc_params[0].uncertainty_set.data is not None and not unc_params[0].uncertainty_set.trained:
                #     raise ValueError("You must first train the uncertainty with problem.train()")
                if unc_params[0].uncertainty_set.trained:
                    unc_params[0].uncertainty_set.paramT.value = problem.param_values['T']
                    unc_params[0].uncertainty_set.paramb.value = problem.param_values['b']
            else:
                # canon_constr, aux_constr = self.canonicalize_tree(
                #     constraint, 0)
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

    def remove_uncertainty(self, unc_lst, uvar, std_lst):
        "canonicalize each term separately with inf convolution"
        # import ipdb
        # ipdb.set_trace()
        num = len(unc_lst)
        if len(unc_lst[0].shape) >= 1:
            num_constr = unc_lst[0].shape[0]
        else:
            num_constr = 1
        trans = uvar.uncertainty_set.affine_transform
        if trans:
            if len(trans['A'].shape) > 1:
                shape = trans['A'].shape[1]
            else:
                shape = 1
        elif len(uvar.shape) >= 1:
            shape = uvar.shape[0]
        else:
            shape = 1
        if shape == 1:
            z = Variable(num)
        else:
            z = Variable((num, shape))
        z_cons = 0
        z_new_cons = {}
        new_vars = {}
        aux_expr = 0
        aux_const = []
        j = 0
        for ind in range(num):
            # if len(unc_lst[ind].variables()) (check if has variable)
            u_expr, cons = self.remove_const(unc_lst[ind], 1)
            uvar = mul_canon_transform(uvar, cons)
            new_expr, new_const = self.canonicalize_tree(u_expr, z[ind])
            if self.has_unc_param(new_expr):
                assert (num_constr == shape)
                if shape == 1:
                    new_vars[ind] = Variable((num_constr, shape))
                else:
                    new_vars[ind] = Variable((num_constr, shape))
                for idx in range(num_constr):
                    # import ipdb
                    # ipdb.set_trace()
                    new_expr, new_const = uvar.isolated_unc(idx, new_vars[ind][idx], num_constr)
                    aux_expr = aux_expr + new_expr
                    aux_const += new_const
                    if j == 1:
                        z_new_cons[idx] += new_vars[ind][idx]
                    else:
                        z_new_cons[idx] = new_vars[ind][idx]

                j = 1
            else:
                aux_expr = aux_expr + new_expr
                aux_const += new_const
                z_cons += z[ind]
        z_unc = Variable((num_constr, shape))
        if j == 1:
            for ind in range(num_constr):
                aux_const += [z_cons + z_new_cons[ind] == -z_unc[ind]]
        else:
            aux_const += [z_cons == -z_unc[0]]
        new_expr, new_const = uvar.conjugate(z_unc, num_constr)
        aux_const += new_const
        aux_expr = aux_expr + new_expr
        for expr in std_lst:
            aux_expr = aux_expr + expr
        return aux_expr <= 0, aux_const

    def count_unq_uncertain_param(self, expr):
        unc_params = []
        if isinstance(expr, Inequality):
            unc_params += [v for v in expr.parameters()
                           if isinstance(v, UncertainParameter)]
            return len(unique_list(unc_params))
        # elif expr.is_constant():
        #     return 0
        else:
            unc_params += [v for v in expr.parameters()
                           if isinstance(v, UncertainParameter)]
        return len(unique_list(unc_params))

    def has_unc_param(self, expr):
        if not isinstance(expr, int) and not isinstance(expr, float):
            return self.count_unq_uncertain_param(expr) == 1
        else:
            return 0

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
        # import ipdb
        # ipdb.set_trace()
        # Check Initial Conditions
        if isinstance(expr, Inequality):
            return self.separate_uncertainty(expr.args[0] + -1*expr.args[1])
        elif self.count_unq_uncertain_param(expr) == 0:
            return ([], [expr])
        elif self.count_unq_uncertain_param(expr) > 1:
            raise ValueError("DRP error: Cannot have multiple uncertain params in the same expr")
        elif len(expr.args) == 0:
            assert (self.has_unc_param(expr))
            return ([expr], [])

        elif isinstance(expr, multiply):
            if self.has_unc_param(expr.args[0]) and \
                    self.has_unc_param(expr.args[1]):
                raise ValueError("DRP error: Cannot have uncertainty multiplied by each other")
            if self.has_unc_param(expr.args[0]):
                unc_lst, std_lst = self.separate_uncertainty(expr.args[0])
                if isinstance(expr.args[1], NegExpression):
                    new_unc_lst = [-1*expr.args[1].args[0] * g_u for g_u in unc_lst]
                    new_std_lst = [-1*expr.args[1].args[0] * h_x for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                elif isinstance(expr.args[1], Promote):
                    new_unc_lst = [expr.args[1].value[0] * g_u for g_u in unc_lst]
                    new_std_lst = [expr.args[1].value[0] * h_x for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                new_unc_lst = [expr.args[1] * g_u for g_u in unc_lst]
                new_std_lst = [expr.args[1] * h_x for h_x in std_lst]
                return (new_unc_lst, new_std_lst)
            else:
                unc_lst, std_lst = self.separate_uncertainty(expr.args[1])
                if isinstance(expr.args[0], NegExpression):
                    new_unc_lst = [-1*expr.args[0].args[0] * g_u for g_u in unc_lst]
                    new_std_lst = [-1*expr.args[0].args[0] * h_x for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                elif isinstance(expr.args[0], Promote):
                    new_unc_lst = [expr.args[0].value[0] * g_u for g_u in unc_lst]
                    new_std_lst = [expr.args[0].value[0] * h_x for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                new_unc_lst = [expr.args[0] * g_u for g_u in unc_lst]
                new_std_lst = [expr.args[0] * h_x for h_x in std_lst]
                return (new_unc_lst, new_std_lst)

        elif isinstance(expr, MulExpression):
            # import ipdb
            # ipdb.set_trace()
            if self.has_unc_param(expr.args[0]) and \
                    self.has_unc_param(expr.args[1]):
                raise ValueError("DRP error: Cannot have uncertainty multiplied by each other")
            if self.has_unc_param(expr.args[0]):
                unc_lst, std_lst = self.separate_uncertainty(expr.args[0])
                if isinstance(expr.args[1], NegExpression):
                    new_unc_lst = [-1 * g_u @ expr.args[1].args[0] for g_u in unc_lst]
                    new_std_lst = [-1 * h_x @ expr.args[1].args[0] for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                elif isinstance(expr.args[1], Promote):
                    new_unc_lst = [g_u * expr.args[1].value[0] for g_u in unc_lst]
                    new_std_lst = [h_x * expr.args[1].value[0] for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                new_unc_lst = [g_u @ expr.args[1] for g_u in unc_lst]
                new_std_lst = [h_x @ expr.args[1] for h_x in std_lst]
                return (new_unc_lst, new_std_lst)
            else:
                unc_lst, std_lst = self.separate_uncertainty(expr.args[1])
                if isinstance(expr.args[0], NegExpression):
                    new_std_lst = [-1 * h_x @ expr.args[0].args[0] for h_x in std_lst]
                    new_unc_lst = [-1 * g_x @ expr.args[0].args[0] for g_x in unc_lst]
                    return (new_unc_lst, new_std_lst)
                elif isinstance(expr.args[0], Promote):
                    new_unc_lst = [expr.args[0].value[0] * g_u for g_u in unc_lst]
                    new_std_lst = [expr.args[0].value[0] * h_x for h_x in std_lst]
                    return (new_unc_lst, new_std_lst)
                new_unc_lst = [expr.args[0] @ g_u for g_u in unc_lst]
                new_std_lst = [expr.args[0] @ h_x for h_x in std_lst]
                return (new_unc_lst, new_std_lst)

        elif isinstance(expr, AddExpression):
            if self.has_unc_param(expr.args[0]) and \
                    self.has_unc_param(expr.args[1]):
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
            unc_lst, std_lst = self.separate_uncertainty(expr.args[0])
            new_unc_lst = [-1 * g_u for g_u in unc_lst]
            new_std_lst = [-1 * h_x for h_x in std_lst]
            return (new_unc_lst, new_std_lst)
        elif isinstance(expr, Promote):
            raise ValueError("DRP error: not able to process non multiplication/additions")
        else:
            # import ipdb
            # ipdb.set_trace()
            raise ValueError("DRP error: not able to process non multiplication/additions")

    def remove_const(self, expr, cons):
        # import ipdb
        # ipdb.set_trace()
        if len(expr.args) == 0:
            return expr, cons
        elif isinstance(expr, multiply):
            if expr.args[0].is_constant() and not self.has_unc_param(expr.args[0]):
                if expr.args[0].is_scalar():
                    cons = (cons*expr.args[0]).value
                elif isinstance(expr.args[0], Promote):
                    cons = (cons*expr.args[0].args[0]).value
                return self.remove_const(expr.args[1], cons)
            if expr.args[1].is_constant() and not self.has_unc_param(expr.args[1]):
                if expr.args[1].is_scalar():
                    cons = (cons*expr.args[1]).value
                elif isinstance(expr.args[1], Promote):
                    cons = (cons*expr.args[1].args[0]).value
                return self.remove_const(expr.args[0], cons)
            if isinstance(expr.args[0], NegExpression) and isinstance(expr.args[1], NegExpression):
                return self.remove_const(expr.args[0].args[0]*expr.args[1].args[0], cons)
            elif isinstance(expr.args[0], NegExpression):
                cons = (-1*cons).value
                return self.remove_const(expr.args[0].args[0], cons)
            elif isinstance(expr.args[1], NegExpression):
                cons = (-1*cons).value
                return self.remove_const(expr.args[1].args[0], cons)
            else:
                expr1, cons = self.remove_const(expr.args[0], cons)
                expr2, cons = self.remove_const(expr.args[1], cons)
                return expr1*expr2, cons
        elif isinstance(expr, MulExpression):
            expr1, cons = self.remove_const(expr.args[0], cons)
            expr2, cons = self.remove_const(expr.args[1], cons)
            return expr1*expr2, cons
        else:
            return expr, cons
