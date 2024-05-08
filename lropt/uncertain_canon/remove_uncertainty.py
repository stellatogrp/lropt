
import numpy as np
from cvxpy import Variable, problems
from cvxpy.atoms.affine.promote import Promote
from cvxpy.constraints.nonpos import Inequality
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction

# from lropt.uncertain_canon.atom_canonicalizers.mul_canon import mul_canon_transform
# from lropt.uncertain_canon.remove_constant import REMOVE_CONSTANT_METHODS as rm_const_methods
from lropt.uncertain_canon.utils import standard_invert
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.mro import MRO
from lropt.utils import unique_list


class RemoveUncertainty(Reduction):
    """Recursively canonicalize each expression in a problem.
    This reduction recursively canonicalizes every expression tree in a
    problem, visiting each node. At every node, this reduction first
    canonicalizes its arguments; it then canonicalizes the node, using the
    canonicalized arguments.
    mapping node types to functions that canonicalize them; the signature
    of these canonicalizing functions must be
        def canon_func(expr, canon_args) --> (new_expr, constraints)
    where `expr` is the `Expression` (node) to canonicalize, canon_args
    is a list of the canonicalized arguments of this expression,
    `new_expr` is a canonicalized expression, and `constraints` is a list
    of constraints introduced while canonicalizing `expr`.
    Attributes:
    ----------
        problem : Problem
            A problem owned by this reduction.
    """

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

    def remove_uncertain_terms(self, uvar, k_num,z_cons, aux_constraint, u_shape, smaller_u_shape):
        "add constraints for the conjugates of the uncertain terms"
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
            # add certain terms
            for expr in cur_cons_data['std_lst']:
                aux_expr = aux_expr + expr
            aux_constraint = aux_constraint + new_constraint
            fin_expr = aux_expr
            if is_mro:
                aux_constraint += [aux_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho*lmbda + uvar.uncertainty_set._w@sval
        return fin_expr, aux_constraint, lmbda, sval

    def mulexpression_canon_transform(self,u, P):
        "adjust affine transform by the data matrix"
        if len(P.shape) == 1:
            P = np.reshape(P,(1,P.shape[0]))
        uset = u.uncertainty_set
        if uset.affine_transform_temp:
            uset.affine_transform_temp['b'] = P@uset.affine_transform_temp['b']
            uset.affine_transform_temp['A'] = P@uset.affine_transform_temp['A']
        else:
            uset.affine_transform_temp = {'A': P, 'b': np.zeros(np.shape(P)[0])}
        return u

    def mul_canon_transform(self,u, c):
        "adjust affine transform by the data scalar"
        uset = u.uncertainty_set
        if isinstance(c, Promote):
            c = c.value[0]
        if uset.affine_transform_temp:
            uset.affine_transform_temp['b'] = c*uset.affine_transform_temp['b']
            uset.affine_transform_temp['A'] = c*uset.affine_transform_temp['A']
        else:
            if len(u.shape) == 0:
                uset.affine_transform_temp = {'A': c*np.eye(1), 'b': 0}
            else:
                uset.affine_transform_temp = {'A': c*np.eye(u.shape[0]), 'b': np.zeros(u.shape[0])}
        return u

    def canonicalize_mul(self, z_cons,u_shape,uvar,transform_data,
                              aux_expr, aux_constraint,var,is_isolated):
        """canonicalize the uncertain terms by adjusting the affine transform,
        then applying the canon_method"""
        z = Variable(u_shape)
        if u_shape==1:
            uvar = self.mul_canon_transform(uvar,transform_data)
        else:
            uvar = self.mulexpression_canon_transform(uvar,transform_data)
        if is_isolated:
            new_expr, new_constraint = uvar.isolated_unc(z)
        else:
            new_expr, new_constraint = uvar.remove_uncertain(var,z)
        aux_expr = aux_expr + new_expr
        aux_constraint += new_constraint
        z_cons = z_cons + z
        return z_cons, aux_expr, aux_constraint

    def remove_uncertainty_helper(self, cur_cons_data, uvar,is_mro):
        "remove the uncertain terms and the uncertainty set"
        u_shape = self.get_u_shape(uvar)
        smaller_u_shape = uvar.uncertainty_set._dimension
        k_num = 1 if not is_mro else uvar.uncertainty_set._K
        aux_constraint = []
        aux_expr = 0
        z_cons = np.zeros(u_shape)
        if cur_cons_data['has_uncertain_mult'] or ['has_uncertain_isolated']:
            # canonicalize uncertain constraints that are multiplied againt x
            if cur_cons_data['has_uncertain_mult']:
                z_cons, aux_expr, aux_constraint = self.canonicalize_mul(
                    z_cons = z_cons, u_shape=u_shape,uvar=uvar,
                    transform_data= cur_cons_data["unc_term"],
                    aux_expr=aux_expr, aux_constraint=aux_constraint,
                    var = cur_cons_data["var"],is_isolated=False)

            # canonicalize isolated uncertian constrains
            if cur_cons_data['has_uncertain_isolated']:
                z_cons, aux_expr, aux_constraint = self.canonicalize_mul(
                    z_cons = z_cons, u_shape=u_shape,uvar=uvar,
                    transform_data= cur_cons_data["unc_isolated"],
                    aux_expr=aux_expr, aux_constraint=aux_constraint,
                    var = None,is_isolated=True)

            # relate the conjugate variables
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
            # No uncertain term, conjugate only the uncertainty set
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
            return False

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
