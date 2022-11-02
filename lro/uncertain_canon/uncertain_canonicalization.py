from cvxpy import Variable, problems
# from cvxpy.expressions.variable import Variable
# from cvxpy.atoms.affine.add_expr import AddExpression
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
from lro.uncertain_canon.remove_constant import \
    REMOVE_CONSTANT_METHODS as rm_const_methods
from lro.uncertain_canon.separate_uncertainty import \
    SEPARATION_METHODS as sep_methods
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
        num_unc_fns = len(unc_lst)
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
            z = Variable(num_unc_fns)
        else:
            z = Variable((num_unc_fns, shape))
        z_cons = 0
        z_new_cons = {}
        new_vars = {}
        aux_expr = 0
        aux_const = []
        j = 0
        for ind in range(num_unc_fns):

            # if len(unc_lst[ind].variables()) (check if has variable)
            u_expr, cons = self.remove_const(unc_lst[ind], 1)
            uvar = mul_canon_transform(uvar, cons)
            new_expr, new_const = self.canonicalize_tree(u_expr, z[ind])
            if self.has_unc_param(new_expr):
                # ONLY HERE IN ISOLATED U CASE
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
                # import ipdb
                # ipdb.set_trace()
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
        '''separate cvxpy expression into subexpressions with uncertain parameters and without.
        Input:
            expr :
                a cvxpy expression
        Output:
            unc_lst :
                EX: [g_1(u_1,x), g_2(u_1,x)]
                a list of cvxpy multiplication expressions from expr each containing one uncertain parameter
            std_lst :
                Ex: [h_1(x),h_2(x)]
                any other cvxpy expressions

        The original expr is equivalnet to the sum of expressions in unc_lst and std_lst
            '''
        # Check Initial Conditions
        if self.count_unq_uncertain_param(expr) == 0:
            return ([], [expr])
        elif self.count_unq_uncertain_param(expr) > 1:
            raise ValueError("DRP error: Cannot have multiple uncertain params in the same expr")
        elif len(expr.args) == 0:
            assert (self.has_unc_param(expr))
            return ([expr], [])

        elif type(expr) not in sep_methods:
            raise ValueError("DRP error: not able to process non multiplication/additions")

        func = sep_methods[type(expr)]
        return func(self, expr)

    def remove_const(self, expr, cons):
        '''remove the constants at the beginning of an expression with uncertainty'''
        # import ipdb
        # ipdb.set_trace()
        if len(expr.args) == 0:
            return expr, cons

        if type(expr) not in rm_const_methods:
            return expr, cons
        else:
            func = rm_const_methods[type(expr)]
            return func(self, expr, cons)
