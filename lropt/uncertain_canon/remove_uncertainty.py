
import numpy as np
from cvxpy import Variable, problems
from cvxpy.constraints.nonpos import Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction

from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.atom_canonicalizers import CANON_METHODS as remove_uncertain_methods
from lropt.uncertain_canon.atom_canonicalizers.mul_canon import mul_canon_transform
from lropt.uncertain_canon.remove_constant import REMOVE_CONSTANT_METHODS as rm_const_methods
from lropt.uncertain_canon.utils import standard_invert
from lropt.uncertainty_sets.mro import MRO
from lropt.utils import unique_list


class RemoveUncertainty(Reduction):
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

    def __init__(self, canon_methods=remove_uncertain_methods, problem=None) -> None:
        super(RemoveUncertainty, self).__init__(problem=problem)
        self.canon_methods = canon_methods

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint."""
        inverse_data = InverseData(problem)
        canon_constraints = []
        lmbda, sval = (None, None)
        for cons_num, constraint in enumerate(problem.constraints):
            # canon_constr is the constraint rexpressed in terms of its canonicalized arguments,
            # and aux_constr are the constraints generated while canonicalizing the arguments of the
            # original constraint
            if self.has_unc_param(constraint):
                cur_cons_data = problem._cons_data[cons_num]
                canon_constr, lmbda, sval = self.remove_uncertainty(cur_cons_data,
                                                                    canon_constraints, lmbda, sval)
            else:
                canon_constr = constraint
                canon_constraints += [canon_constr]

            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

        new_problem = problems.problem.Problem(problem.objective, canon_constraints)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        return standard_invert(solution=solution, inverse_data=inverse_data)

    def canonicalize_tree(self, expr, var, cons):
        """Recursively canonicalize an Expression."""
        # TODO don't copy affine expressions?
        if type(expr) == cvxtypes.partial_problem():
            canon_expr, constrs = self.canonicalize_tree(
                expr.args[0].objective.expr, var, cons)
            for constr in expr.args[0].constraints:
                canon_constr, aux_constr = self.canonicalize_tree(
                    constr, var, cons)
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


    def remove_uncertain_terms(self, uvar, k_num,z_cons, aux_constraint, u_shape, smaller_u_shape):
        "add constraints for the uncertain term conjugates"
        supp_cons = {}
        z_unc = {}
        for k_ind in range(k_num):
            z_unc[k_ind] = Variable(smaller_u_shape)
            supp_cons[k_ind] = Variable(u_shape)
            if uvar.uncertainty_set.a is not None:
                aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                + uvar.uncertainty_set.a.T@supp_cons[k_ind] == -z_unc[k_ind]]
            else:
                aux_constraint += [z_cons + supp_cons[k_ind] == -z_unc[k_ind]]
        return aux_constraint, z_unc, supp_cons

    def remove_uncertainty_sets(self, uvar,u_shape, k_num, z_cons,
                                supp_cons, z_unc,aux_expr,
                                    aux_constraint, cur_cons_data,
                                    is_mro, has_uncertain):
        "add constraints for the conjugate of the uncertainty set"
        for k_ind in range(k_num):
            terms = (z_unc[k_ind],supp_cons[k_ind],uvar.uncertainty_set.b) \
                if has_uncertain else (u_shape,0,None)
            new_expr, new_constraint, lmbda, sval = uvar.conjugate(terms[0],terms[1], k_ind)
            aux_expr = aux_expr + new_expr
            if terms[2] is not None:
                aux_expr = aux_expr - uvar.uncertainty_set.b@(z_cons) \
                    - supp_cons[k_ind]@uvar.uncertainty_set.b
            # add certian terms
            for expr in cur_cons_data['std_lst']:
                aux_expr = aux_expr + expr
            aux_constraint = aux_constraint + new_constraint
            fin_expr = aux_expr
            if is_mro:
                aux_constraint += [aux_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho*lmbda + uvar.uncertainty_set._w@sval
        return fin_expr, aux_constraint, lmbda, sval

    def remove_uncertainty_helper(self, cur_cons_data, uvar,is_mro):
        "canonicalize each term separately with inf convolution"
        u_shape = self.get_u_shape(uvar)
        smaller_u_shape = uvar.uncertainty_set._dimension
        k_num = 1 if not is_mro else uvar.uncertainty_set._K
        merged_list = cur_cons_data['unc_lst'] + cur_cons_data['unc_isolated']
        num_unc_fns = len(merged_list)
        aux_constraint = []
        aux_expr = 0
        if num_unc_fns > 0:
            if u_shape == 1:
                z = Variable(num_unc_fns)
            else:
                z = Variable((num_unc_fns, u_shape))
            z_cons = np.zeros(u_shape)

            # generate conjugate variables and constraints for uncertain terms
            for ind in range(num_unc_fns):
                u_expr, constant = self.remove_constant(merged_list[ind])
                new_expr, new_constraint = self.canonicalize_tree(
                    u_expr, z[ind], constant)

                if self.has_unc_param(new_expr):
                    uvar = mul_canon_transform(uvar, constant)
                    new_expr, new_constraint = uvar.isolated_unc(z[ind])

                aux_expr = aux_expr + new_expr
                aux_constraint += new_constraint
                z_cons = z_cons + z[ind]

            aux_constraint, z_unc, supp_cons = \
                self.remove_uncertain_terms(uvar=uvar, k_num=k_num,
                        z_cons=z_cons, aux_constraint=aux_constraint,
                        u_shape=u_shape, smaller_u_shape=smaller_u_shape)

            # add constraints for uncertainty set
            fin_expr, aux_constraint, lmbda, sval = \
                self.remove_uncertainty_sets(uvar=uvar,u_shape=u_shape,
                k_num=k_num, z_cons=z_cons, supp_cons=supp_cons,
                z_unc=z_unc,aux_expr = aux_expr,
                aux_constraint=aux_constraint,
                cur_cons_data = cur_cons_data,
                is_mro= is_mro,has_uncertain=True)
        else:
            # No uncertain term, conjudate only the uncertainty set
            fin_expr, aux_constraint, lmbda, sval = \
                self.remove_uncertainty_sets(uvar=uvar,u_shape=u_shape,
            k_num=k_num,z_cons = None,supp_cons= None, z_unc = None,
            aux_expr = aux_expr, aux_constraint=aux_constraint,
            cur_cons_data = cur_cons_data, is_mro= is_mro, has_uncertain=False)
        return fin_expr <= 0, aux_constraint, lmbda, sval

    def count_unq_uncertain_param(self, expr):
        unc_params = []
        if isinstance(expr, Inequality):
            unc_params += [v for v in expr.parameters() if isinstance(v, UncertainParameter)]
            return len(unique_list(unc_params))

        else:
            unc_params += [v for v in expr.parameters() if isinstance(v, UncertainParameter)]
        return len(unique_list(unc_params))

    def has_unc_param(self, expr):
        if not isinstance(expr, int) and not isinstance(expr, float):
            return self.count_unq_uncertain_param(expr) >= 1
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

    def remove_uncertainty(self, cur_cons_data, canon_constraints, lmbda, sval):
        """
        This function removes uncertainty. Connects constraints if is mro and
        has a maximum constraint
        """

        unc_param = cur_cons_data['unc_param']
        is_mro = type(unc_param.uncertainty_set) == MRO
        canon_constr, aux_constr, new_lmbda, new_sval = \
            self.remove_uncertainty_helper(cur_cons_data, unc_param,is_mro)
        canon_constraints += aux_constr + [canon_constr]
        if lmbda is None:
            lmbda = new_lmbda
            sval = new_sval
        else:
            if is_mro:
                canon_constraints += [lmbda == new_lmbda]
                canon_constraints += [sval == new_sval]

        return canon_constr, lmbda, sval

    def remove_constant(self, expr, constant=1):
        '''remove the constants at the beginning of an expression with uncertainty'''
        if len(expr.args) == 0:
            return expr, constant

        if type(expr) not in rm_const_methods:
            return expr, constant
        else:
            func = rm_const_methods[type(expr)]
            return func(self, expr, constant)
