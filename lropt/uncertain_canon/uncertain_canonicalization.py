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

from lropt.remove_uncertain.atom_canonicalizers.mul_canon import \
    mul_canon_transform
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.remove_constant import \
    REMOVE_CONSTANT_METHODS as rm_const_methods
from lropt.uncertain_canon.separate_uncertainty import \
    SEPARATION_METHODS as sep_methods
from lropt.uncertainty_sets.budget import Budget
from lropt.uncertainty_sets.mro import MRO
from lropt.uncertainty_sets.polyhedral import Polyhedral
from lropt.utils import unique_list


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
            problem.objective, 0, 1)

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
                unc_lst, std_lst, is_max = self.separate_uncertainty(constraint)
                if is_max:
                    unc_lst_tot = []
                    for cons_idx in range(len(unc_lst)):
                        unc_lst_tot += unc_lst[cons_idx]
                else:
                    unc_lst_tot = unc_lst
                if len(unc_lst_tot[0].shape) >= 1:
                    element_shape = unc_lst_tot[0].shape[0]
                else:
                    element_shape = 1
                unc_params = []
                unc_params += [v for v in unc_lst_tot[0].parameters()
                               if isinstance(v, UncertainParameter)]
                if is_max == 1:
                    if not type(unc_params[0].uncertainty_set) == MRO:
                        canon_constr, aux_constr, lmda = self.remove_uncertainty(
                            unc_lst[0], unc_params[0], std_lst[0], element_shape)
                        canon_constraints += aux_constr + [canon_constr]

                        for new_cons_idx in range(1, len(unc_lst)):
                            canon_constr, aux_constr, new_lmda = self.remove_uncertainty(
                                unc_lst[new_cons_idx], unc_params[0], std_lst[new_cons_idx], element_shape)
                            canon_constraints += aux_constr + [canon_constr]

                            if type(unc_params[0].uncertainty_set) == Budget:
                                canon_constraints += [lmda[0] == new_lmda[0], lmda[1] == new_lmda[1]]

                            elif not type(unc_params[0].uncertainty_set) == Polyhedral:
                                canon_constraints += [lmda == new_lmda]
                    else:
                        canon_constr, aux_constr, lmda, sval = self.remove_uncertainty_mro(
                            unc_lst[0], unc_params[0], std_lst[0], element_shape)
                        canon_constraints += aux_constr + [canon_constr]

                        for new_cons_idx in range(1, len(unc_lst)):
                            canon_constr, aux_constr, new_lmda, new_sval = self.remove_uncertainty_mro(
                                unc_lst[new_cons_idx], unc_params[0], std_lst[new_cons_idx], element_shape)
                            canon_constraints += aux_constr
                            canon_constraints += [lmda == new_lmda]
                            canon_constraints += [sval == new_sval]

                else:
                    # import ipdb
                    # ipdb.set_trace()
                    if not type(unc_params[0].uncertainty_set) == MRO:
                        canon_constr, aux_constr, lmbda = self.remove_uncertainty(
                            unc_lst, unc_params[0], std_lst, element_shape)
                        canon_constraints += aux_constr + [canon_constr]
                    else:
                        canon_constr, aux_constr, lmbda, sval = self.remove_uncertainty_mro(
                            unc_lst, unc_params[0], std_lst, element_shape)
                        canon_constraints += aux_constr + [canon_constr]
                # import ipdb
                # ipdb.set_trace()
                # if unc_params[0].uncertainty_set.data is not None and not unc_params[0].uncertainty_set.trained:
                #     raise ValueError("You must first train the uncertainty with problem.train()")
                # if unc_params[0].uncertainty_set.trained:
                #     unc_params[0].uncertainty_set.paramT.value = problem.param_values['T']
                #     unc_params[0].uncertainty_set.paramb.value = problem.param_values['b']
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

    def canonicalize_tree(self, expr, var, cons):
        """Recursively canonicalize an Expression."""
        # TODO don't copy affine expressions?
        if type(expr) == cvxtypes.partial_problem():
            canon_expr, constrs = self.canonicalize_tree(
                expr.args[0].objective.expr, var, cons)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(constr, var, cons)
                constrs += [canon_constr] + aux_constr
        else:
            canon_args = []
            constrs = []
            for arg in expr.args:
                canon_arg, c = self.canonicalize_tree(arg, var, cons)
                canon_args += [canon_arg]
                constrs += c
            canon_expr, c = self.canonicalize_expr(expr, canon_args, var, cons)
            constrs += c
        return canon_expr, constrs

    def canonicalize_expr(self, expr, args, var, cons):
        """Canonicalize an expression, w.r.t. canonicalized arguments."""
        # Constant trees are collapsed, but parameter trees are preserved.
        if isinstance(expr, Expression) and (
                expr.is_constant() and not expr.parameters()):
            return expr, []
        elif type(expr) in self.canon_methods:
            return self.canon_methods[type(expr)](expr, args, var, cons)
        else:
            return expr.copy(args), []

    def remove_uncertainty(self, unc_lst, uvar, std_lst, num_constr):
        "canonicalize each term separately with inf convolution"
        # import ipdb
        # ipdb.set_trace()
        u_shape = self.get_u_shape(uvar)
        num_unc_fns = len(unc_lst)
        if num_unc_fns > 0:
            if u_shape == 1:
                z = Variable(num_unc_fns)
            else:
                z = Variable((num_unc_fns, u_shape))
            z_cons = 0
            z_new_cons = {}
            new_vars = {}
            aux_expr = 0
            aux_constraint = []
            has_isolated = 0
            for ind in range(num_unc_fns):

                # if len(unc_lst[ind].variables()) (check if has variable)
                u_expr, constant = self.remove_constant(unc_lst[ind])

                # uvar = mul_canon_transform(uvar, cons)
                new_expr, new_constraint = self.canonicalize_tree(u_expr, z[ind], constant)
                if self.has_unc_param(new_expr):
                    uvar = mul_canon_transform(uvar, constant)
                    new_vars[ind] = Variable((num_constr, u_shape))
                    for idx in range(num_constr):
                        # import ipdb
                        # ipdb.set_trace()
                        new_expr, new_constraint = uvar.isolated_unc(idx, new_vars[ind][idx], num_constr)
                        aux_expr = aux_expr + new_expr
                        aux_constraint += new_constraint
                        if has_isolated == 1:
                            z_new_cons[idx] += new_vars[ind][idx]
                        else:
                            z_new_cons[idx] = new_vars[ind][idx]
                    has_isolated = 1
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    aux_expr = aux_expr + new_expr
                    aux_constraint += new_constraint
                    z_cons += z[ind]

            z_unc = Variable((num_constr, u_shape))
            if has_isolated == 1:
                for idx in range(num_constr):
                    aux_constraint += [z_cons + z_new_cons[idx] == -z_unc[idx]]
            else:
                aux_constraint += [z_cons == -z_unc[0]]
            new_expr, new_constraint, lmbda = uvar.conjugate(z_unc, num_constr, k_ind=0)
            aux_expr = aux_expr + new_expr
            aux_constraint = aux_constraint + new_constraint
        else:
            aux_expr, aux_constraint, lmbda = uvar.conjugate(u_shape, num_constr, k_ind=0)
        for expr in std_lst:
            aux_expr = aux_expr + expr
        return aux_expr <= 0, aux_constraint, lmbda

    def remove_uncertainty_mro(self, unc_lst, uvar, std_lst, num_constr):
        "canonicalize each term separately with inf convolution"
        # import ipdb
        # ipdb.set_trace()
        u_shape = self.get_u_shape(uvar)
        num_unc_fns = len(unc_lst)
        if num_unc_fns > 0:
            if u_shape == 1:
                z = Variable(num_unc_fns)
            else:
                z = Variable((num_unc_fns, u_shape))
            z_cons = 0
            z_new_cons = {}
            new_vars = {}
            aux_expr = 0
            aux_constraint = []
            has_isolated = 0
            for ind in range(num_unc_fns):
                # if len(unc_lst[ind].variables()) (check if has variable)
                u_expr, constant = self.remove_constant(unc_lst[ind])

                # uvar = mul_canon_transform(uvar, cons)
                new_expr, new_constraint = self.canonicalize_tree(u_expr, z[ind], constant)
                if self.has_unc_param(new_expr):
                    uvar = mul_canon_transform(uvar, constant)
                    new_vars[ind] = Variable((num_constr, u_shape))
                    for idx in range(num_constr):
                        # import ipdb
                        # ipdb.set_trace()
                        new_expr, new_constraint = uvar.isolated_unc(idx, new_vars[ind][idx], num_constr)
                        aux_expr = aux_expr + new_expr
                        aux_constraint += new_constraint
                        if has_isolated == 1:
                            z_new_cons[idx] += new_vars[ind][idx]
                        else:
                            z_new_cons[idx] = new_vars[ind][idx]
                    has_isolated = 1
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    aux_expr = aux_expr + new_expr
                    aux_constraint += new_constraint
                    z_cons += z[ind]

            z_unc = {}
            for k_ind in range(uvar.uncertainty_set._K):
                z_unc[k_ind] = Variable((num_constr, u_shape))
                if has_isolated == 1:
                    for idx in range(num_constr):
                        aux_constraint += [z_cons + z_new_cons[idx] == -z_unc[k_ind][idx]]
                else:
                    aux_constraint += [z_cons == -z_unc[k_ind][0]]
                new_expr, new_constraint, lmbda, sval = uvar.conjugate(z_unc[k_ind], num_constr, k_ind)
                cur_expr = aux_expr + new_expr
                for expr in std_lst:
                    cur_expr = cur_expr + expr
                aux_constraint = aux_constraint + new_constraint + [cur_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho*lmbda + uvar.uncertainty_set._w@sval
        else:
            aux_constraint = []
            for k_ind in range(uvar.uncertainty_set._K):
                aux_expr, new_constraint, lmbda, sval = uvar.conjugate(u_shape, num_constr, k_ind)
                cur_expr = aux_expr
                for expr in std_lst:
                    cur_expr = cur_expr + expr
                aux_constraint = aux_constraint + new_constraint + [cur_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho*lmbda + uvar.uncertainty_set._w@sval
        return fin_expr <= 0, aux_constraint, lmbda, sval

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

    def get_u_shape(self, uvar):
        trans = uvar.uncertainty_set.affine_transform

        # find shape of uncertainty parameter
        if trans:
            if len(trans['A'].shape) > 1:
                u_shape = trans['A'].shape[1]
            else:
                u_shape = 1
        elif len(uvar.shape) >= 1:
            u_shape = uvar.shape[0]
        else:
            u_shape = 1

        return u_shape

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
        # import ipdb
        # ipdb.set_trace()
        # Check Initial Conditions
        if self.count_unq_uncertain_param(expr) == 0:
            return [], [expr], 0
        # elif self.count_unq_uncertain_param(expr) > 1:
        #     raise ValueError("DRP error: Cannot have multiple uncertain params in the same expr")
        elif len(expr.args) == 0:
            assert (self.has_unc_param(expr))
            return [expr], [], 0

        elif type(expr) not in sep_methods:
            raise ValueError("DRP error: not able to process non multiplication/additions")
        # import ipdb
        # ipdb.set_trace()
        func = sep_methods[type(expr)]
        return func(self, expr)

    def remove_constant(self, expr, constant=1):
        '''remove the constants at the beginning of an expression with uncertainty'''
        # import ipdb
        # ipdb.set_trace()
        if len(expr.args) == 0:
            return expr, constant

        if type(expr) not in rm_const_methods:
            return expr, constant
        else:
            func = rm_const_methods[type(expr)]
            return func(self, expr, constant)
