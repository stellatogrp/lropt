
import operator

import numpy as np
from cvxpy import Variable, problems
from cvxpy.atoms.affine.promote import Promote
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction

from lropt.uncertain_canon.utils import cross_product, standard_invert
from lropt.uncertainty_sets.mro import MRO
from lropt.uncertainty_sets.scenario import Scenario

# from lropt.uncertain_canon.atom_canonicalizers.mul_canon import mul_canon_transform
# from lropt.uncertain_canon.remove_constant import REMOVE_CONSTANT_METHODS as rm_const_methods
from lropt.utils import has_unc_param


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

    def accepts(self,problem):
        return True

    def apply(self, problem):
        """Recursively canonicalize the objective and every constraint."""
        inverse_data = InverseData(problem)
        canon_constraints = []
        lmbda, sval = (None, None)
        for cons_num, constraint in enumerate(problem.constraints):
            # canon_constr is the constraint rexpressed in terms of its canonicalized arguments,
            # and aux_constr are the constraints generated while canonicalizing the arguments of the
            # original constraint
            if has_unc_param(constraint):
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

    def remove_uncertain_terms(self, u_info, k_num,z_cons, aux_constraint):
        """add constraints for the conjugates of the uncertain terms
            for each uncertain param
        """
        def _gen_a_mult(uvar,z, supp):
            "determines whether or not to multiply the u_term by a"
            if uvar.uncertainty_set.a is not None:
                return uvar.uncertainty_set.a.T@z + uvar.uncertainty_set.a.T@supp
            else:
                return z + supp
        supp_cons = {}
        z_unc = {}
        for u_ind, uvar in enumerate(u_info["list"]):
            u_shape, smaller_u_shape = u_info[u_ind]
            supp_cons[u_ind] = {}
            z_unc[u_ind] = {}
            for k_ind in range(k_num):
                z_unc[u_ind][k_ind] = Variable(smaller_u_shape)
                supp_cons[u_ind][k_ind] = Variable(u_shape)

                aux_constraint += [_gen_a_mult(uvar,z_cons[u_ind],
                        supp_cons[u_ind][k_ind]) == -z_unc[u_ind][k_ind]]
        return aux_constraint, z_unc, supp_cons

    def remove_uncertainty_sets(self, u_info, k_num, z_cons,
                                supp_cons, z_unc,aux_expr,
                                    aux_constraint, cur_cons_data,
                                    is_mro):
        """add constraints for the conjugate of the uncertainty set
        for each uncertain param"""
        for k_ind in range(k_num):
            for u_ind, uvar in enumerate(u_info['list']):
                terms = (z_unc[u_ind][k_ind],supp_cons[u_ind][k_ind],uvar.uncertainty_set.b)
                new_expr, new_constraint, lmbda, sval = uvar.conjugate(terms[0],terms[1], k_ind)
                aux_expr = aux_expr + new_expr
                aux_constraint = aux_constraint + new_constraint
                if terms[2] is not None:
                    aux_expr = aux_expr - uvar.uncertainty_set.b@(
                        z_cons[u_ind])- \
                            supp_cons[u_ind][k_ind]@uvar.uncertainty_set.b
            # add certain terms
            for expr in cur_cons_data['std_lst']:
                aux_expr = aux_expr + expr
            fin_expr = aux_expr
            if is_mro:
                uvar = u_info['list'][0]
                aux_constraint += [aux_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho_mult*\
                    uvar.uncertainty_set.rho*lmbda + uvar.uncertainty_set._w@sval
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
            uset.affine_transform_temp = {'A': P, 'b': None}
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
                uset.affine_transform_temp = {'A': c*np.eye(1), 'b': None}
            else:
                uset.affine_transform_temp = {'A': c*np.eye(u.shape[0]), 'b': None}
        return u

    def canonicalize_mul(self, z_cons_u,u_shape,uvar,transform_data,
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
        z_cons_u = z_cons_u + z
        return z_cons_u, aux_expr, aux_constraint

    def remove_uncertainty_helper(self, cur_cons_data, is_mro):
        """remove the uncertain terms and the uncertainty set
        for each uncertain param"""
        u_info = {}
        u_info["list"] = cur_cons_data["unc_param_list"]
        k_num = 1 if not is_mro else u_info["list"][0].uncertainty_set._K
        aux_constraint = []
        aux_expr = 0
        z_cons = {}
        for u_ind, uvar in enumerate(u_info["list"]):
            u_shape = self.get_u_shape(uvar)
            smaller_u_shape = uvar.uncertainty_set._dimension
            u_info[u_ind] = (u_shape, smaller_u_shape)
            z_cons[u_ind] = np.zeros(u_shape)
            # if cur_cons_data[u_ind]['has_uncertain_mult'] or \
            #     cur_cons_data[u_ind]['has_uncertain_isolated']:
            #     # canonicalize uncertain constraints that are multiplied against x
            if cur_cons_data[u_ind]['has_uncertain_mult']:
                z_cons[u_ind], aux_expr, aux_constraint = self.canonicalize_mul(
                    z_cons_u = z_cons[u_ind], u_shape=u_shape,uvar=uvar,
                    transform_data= cur_cons_data[u_ind]["unc_term"],
                    aux_expr=aux_expr, aux_constraint=aux_constraint,
                    var = cur_cons_data["var"],is_isolated=False)

            # canonicalize isolated uncertain constrains
            if cur_cons_data[u_ind]['has_uncertain_isolated']:
                z_cons[u_ind], aux_expr, aux_constraint = self.canonicalize_mul(
                    z_cons_u = z_cons[u_ind], u_shape=u_shape,uvar=uvar,
                    transform_data= cur_cons_data[u_ind]["unc_isolated"],
                    aux_expr=aux_expr, aux_constraint=aux_constraint,
                    var = None,is_isolated=True)

        # relate the conjugate variables
        aux_constraint, z_unc, supp_cons = \
            self.remove_uncertain_terms(u_info = u_info, k_num=k_num,
                                        z_cons=z_cons,
                                        aux_constraint=aux_constraint)

        # add constraints for uncertainty set
        fin_expr, aux_constraint, lmbda, sval = \
            self.remove_uncertainty_sets(u_info=u_info,
            k_num=k_num, z_cons=z_cons, supp_cons=supp_cons,
            z_unc=z_unc,aux_expr = aux_expr,
            aux_constraint=aux_constraint,
            cur_cons_data = cur_cons_data,
            is_mro= is_mro)
            # else:
            #     # No uncertain term, conjugate only the uncertainty set
            #     fin_expr, aux_constraint, lmbda, sval = \
            #         self.remove_uncertainty_sets(uvar=uvar,u_shape=u_shape,
            #     k_num=k_num,z_cons = None,supp_cons= None, z_unc = None,
            #     aux_expr = aux_expr, aux_constraint=aux_constraint,
            #     cur_cons_data = cur_cons_data, is_mro= is_mro, has_uncertain=False)
        return fin_expr <= 0, aux_constraint, lmbda, sval

    def scenario_helper(self,cur_cons_data):
        """For the scenario approach, duplicate the constraint for each
        uncertainty realization. If there are multiple uncertain parameters,
        use the cartesian product of the realizations, unless cartesian is set
        to false."""
        aux_constraint = []
        data_list = []
        other_list = []
        u_info = {}
        u_info['op'] = []
        u_info['cartesian'] = []
        u_info['other'] = []
        u_info['cart_ind'] = []
        u_info['other_ind'] = []
        cur_vars = cur_cons_data["var"]
        for u_ind, uvar in enumerate(cur_cons_data["unc_param_list"]):
            if uvar.uncertainty_set._cartesian:
                u_info['cartesian'].append(uvar)
                data_list.append(uvar.uncertainty_set.data)
                u_info['cart_ind'].append(u_ind)
            else:
                u_info['other'].append(uvar)
                other_list.append(uvar.uncertainty_set.data)
                u_info['other_ind'].append(u_ind)
            if len(uvar.shape)!=0 and uvar.shape[0]>1:
                u_info['op'].append(operator.matmul)
            else:
                u_info['op'].append(operator.mul)

        if len(data_list)>=1:
            cross_list = cross_product(*data_list)
            num_reps = len(cross_list)
        else:
            num_reps = 0
        num_reps_other = 0
        if len(other_list)!= 0:
            num_reps_other = other_list[0].shape[0]
            for cur_array in other_list[1:]:
                assert num_reps_other == cur_array.shape[0], "if not cartesian,\
                      arrays must have the same number of realizations"
        if num_reps_other!=0 and num_reps != 0:
            assert num_reps == num_reps_other, "if not cartesian,\
                      arrays must have the same number of realizations"

        num_reps = np.maximum(num_reps, num_reps_other)
        for reps in range(num_reps):
            cur_con = cur_cons_data["std_lst"][0]
            for cur_ind, uvar in enumerate(u_info['cartesian']):
            # add uncertain constraints that are multiplied against x
                u_ind = u_info['cart_ind'][cur_ind]
                if cur_cons_data[u_ind]['has_uncertain_mult']:
                    cur_con  = cur_con + cur_vars@(
                        u_info['op'][u_ind](
                            cur_cons_data[u_ind]["unc_term"],
                            cross_list[reps][cur_ind]))

            # canonicalize isolated uncertain constrains
                if cur_cons_data[u_ind]['has_uncertain_isolated']:
                    cur_con = cur_con + u_info['op'][u_ind](
                        cur_cons_data[u_ind]["unc_isolated"],
                        cross_list[reps][cur_ind])

            for cur_ind, uvar in enumerate(u_info['other']):
                u_ind = u_info['other_ind'][cur_ind]
                if cur_cons_data[u_ind]['has_uncertain_mult']:
                    cur_con  = cur_con + cur_vars@(
                        u_info['op'][u_ind](
                            cur_cons_data[u_ind]["unc_term"],
                            other_list[cur_ind][reps]))

            # canonicalize isolated uncertain constrains
                if cur_cons_data[u_ind]['has_uncertain_isolated']:
                    cur_con = cur_con \
                        + u_info['op'][u_ind](
                            cur_cons_data[u_ind]["unc_isolated"],
                            other_list[cur_ind][reps])

            aux_constraint += [cur_con <= 0]
        return aux_constraint[0], aux_constraint[1:], 0, None


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

        unc_param_list = cur_cons_data['unc_param_list']
        for unc_param in unc_param_list:
            is_mro = isinstance(unc_param.uncertainty_set, MRO)
            if is_mro and (len(unc_param_list) != 1):
                raise ValueError("Multiple uncertainty sets is not " + \
                                  "supported for MRO uncertainty")

        if isinstance(unc_param.uncertainty_set, Scenario):
            canon_constr, aux_constr, new_lmbda, new_sval = \
                self.scenario_helper(cur_cons_data)
        else:
            canon_constr, aux_constr, new_lmbda, new_sval = \
                self.remove_uncertainty_helper(cur_cons_data, is_mro)
        canon_constraints += aux_constr + [canon_constr]
        if lmbda is None:
            lmbda = new_lmbda
            sval = new_sval
        else:
            if is_mro:
                canon_constraints += [lmbda == new_lmbda]
                canon_constraints += [sval == new_sval]

        return canon_constr, lmbda, sval
