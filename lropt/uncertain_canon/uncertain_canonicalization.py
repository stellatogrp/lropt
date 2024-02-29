from enum import Enum
import numpy as np
from cvxpy import Variable, problems
from cvxpy.constraints.nonpos import Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution

from lropt.remove_uncertain.atom_canonicalizers.mul_canon import mul_canon_transform
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.remove_constant import REMOVE_CONSTANT_METHODS as rm_const_methods
from lropt.uncertain_canon.separate_uncertainty import SEPARATION_METHODS as sep_methods
from lropt.uncertainty_sets.mro import MRO
from lropt.utils import unique_list


class RemoveUncertaintyMode():
    """
    This is a helper class that helps decide on has_matrix/has_isolated mode.
    """
    class Status(Enum):
        NONE = 0
        HAS_ISOLATED = 1
        HAS_MATRIX = 2

    def __init__(self):
        self._has_matrix = False
        self._has_isolated = False
    
    def seen_matrix(self, seen):
        if seen: self._has_matrix = True
    
    def seen_isolated(self, seen):
        if seen: self._has_isolated = True
    
    def get_status(self) -> Status:
        if self._has_matrix:
            return RemoveUncertaintyMode.Status.HAS_MATRIX
        elif self._has_isolated:
            return RemoveUncertaintyMode.Status.HAS_ISOLATED
        return RemoveUncertaintyMode.Status.NONE

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
        def _gen_objective_constraints(problem):
            """
            This function generates canon objective and new constraints
            """
            if self.has_unc_param(problem.objective.expr):
                epigraph_obj = Variable()
                epi_cons = problem.objective.expr <= epigraph_obj
                new_constraints = [epi_cons] + problem.constraints
                canon_objective = Minimize(epigraph_obj)
            else:
                canon_objective = problem.objective
                new_constraints = problem.constraints
            return canon_objective, new_constraints
        
        inverse_data = InverseData(problem)
        # import ipdb
        # ipdb.set_trace()
        # canon_objective, canon_constraints = self.canonicalize_tree(
        #     problem.objective, 0, 1)

        canon_objective, new_constraints = _gen_objective_constraints(problem)
        canon_constraints = []
        
        for constraint in new_constraints:
            # canon_constr is the constraint rexpressed in terms of
            # its canonicalized arguments, and aux_constr are the constraints
            # generated while canonicalizing the arguments of the original
            # constraint
            if self.has_unc_param(constraint):
                unc_lst, std_lst, is_max = self.separate_uncertainty(constraint)
                canon_constr = self.remove_uncertainty(unc_lst, std_lst, is_max, canon_constraints)
            else:
                canon_constr = constraint
                canon_constraints += [canon_constr]

            inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

        new_problem = problems.problem.Problem(canon_objective, canon_constraints)
        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                 if vid in solution.primal_vars}
        dvars = {orig_id: solution.dual_vars[vid]
                 for orig_id, vid in inverse_data.cons_id_map.items()
                 if vid in solution.dual_vars}

        return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)

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

    #TODO (Irina): Please go over all the functions and make sure the names make sense, and update
    #the dosctrings where relevant
    def _update_new_constraint(self, unc_lst, new_vars, z_new_cons, z_cons,
                               num_constr, u_shape, ind, z, uvar, aux_expr, aux_constraint):
        """
        TODO: Complete the docstring
        This is an internal function that updates new_expr and new_constraint. It is used while
        removing uncertainty from an expression (remove_uncertainty_expr).

        Returns:
            new_expr

            new_constraint

            has_matrix

            has_isolated
        """
        def _is_mat(expr: Expression) -> bool:
            """
            This is a helper function that returns True if the expression is a 2D matrix and false
            otherwise.
            """
            return len(expr.shape)==2

        def _gen_var_mat(self, new_expr, mat_flag):
            """
            This is a helper function that creates var_mat.
            """
            if not mat_flag:
                return None
            if self.has_unc_param(new_expr.args[0]):
                var_mat = new_expr.args[1]
            elif self.has_unc_param(new_expr.args[1]):
                var_mat = new_expr.args[0]
            return var_mat
        
        def _update_new_vars(mat_flag, ind, z_new_cons, col, new_vars, num_constr, u_shape):
            new_var_ind = Variable((num_constr, u_shape))
            if mat_flag:
                if ind==0:
                    z_new_cons[col] = {}
                new_vars[col] = {}
                new_vars[col][ind] = new_var_ind
            else:
                new_vars[ind] = new_var_ind
        
        def _gen_new_expr_constraint(uvar, total_inds, col, i, new_vars, ind, var_mat,
                                     num_constr_cols, num_constr):
            """
            This is a helper function that generates new_expr and new_constraint for different cases
            """
            if mat_flag:
                new_expr, new_constraint = uvar.isolated_unc_matrix(total_inds,
                        col, i, new_vars[col][ind][i], var_mat[col][i],num_constr_cols, num_constr)
            else:
                new_expr, new_constraint = uvar.isolated_unc(i, new_vars[ind][i],num_constr)
            return new_expr, new_constraint
        
        def _update_z_new_cons(mat_flag, z_new_cons, new_vars, col, ind, i):
            """
            This is a helper function that updates z_new_cons.
            """
            if mat_flag:
                if ind >=1:
                    z_new_cons[col][i] += new_vars[col][ind][i]
                else:
                    z_new_cons[col][i] = new_vars[col][ind][i]
            else:
                if ind == 1:
                    z_new_cons[i] += new_vars[ind][i]
                else:
                    z_new_cons[i] = new_vars[ind][i]

        has_matrix = False
        has_isolated = False

        #Check if has variable
        u_expr, constant = self.remove_constant(unc_lst[ind])
        new_expr, new_constraint = self.canonicalize_tree(u_expr, z[ind], constant)

        if self.has_unc_param(new_expr):
            uvar = mul_canon_transform(uvar, constant)
        mat_flag = _is_mat(unc_lst[ind])
        #Vector is like a 2D matrix with one dimension == 1
        num_constr_cols = unc_lst[ind].shape[1] if mat_flag else 1
        if not self.has_unc_param(new_expr):
            aux_expr = aux_expr + new_expr
            aux_constraint += new_constraint
            z_cons = z_cons + z[ind]
            return has_matrix, has_isolated, num_constr_cols, aux_expr, aux_constraint, uvar, z_cons
        var_mat = _gen_var_mat(self, new_expr, mat_flag)
        total_inds = 0 #Total element (takes into account rows and columns). Used only if mat_flag.
        
        for col in range(num_constr_cols):
            _update_new_vars(mat_flag, ind, z_new_cons, col, new_vars, num_constr, u_shape)
            for i in range(num_constr):
                new_expr, new_constraint = _gen_new_expr_constraint(uvar=uvar,
                        total_inds=total_inds, col=col, i=i, new_vars=new_vars, ind=ind,
                        var_mat=var_mat, num_constr_cols=num_constr_cols, num_constr=num_constr)
                _update_z_new_cons(mat_flag, z_new_cons, new_vars, col, ind, i)
                aux_expr = aux_expr + new_expr
                aux_constraint += new_constraint
                total_inds += 1
        if mat_flag:
            has_matrix = True
        else:
            has_isolated = True
        
        return has_matrix, has_isolated, num_constr_cols, aux_expr, aux_constraint, uvar, z_cons
    
    def _gen_aux_constraint(self, status, std_lst, num_constr_cols, num_constr, u_shape,
                            smaller_u_shape, uvar, z_cons, z_new_cons, z_unc, supp_cons,
                            aux_expr, aux_constraint):
        if status.get_status()==RemoveUncertaintyMode.Status.HAS_MATRIX:
            for expr in std_lst:
                aux_expr = aux_expr + expr
            supp_cons = {}
            z_unc = {}
            for col in range(num_constr_cols):
                supp_cons[col] = Variable((num_constr, u_shape))
                z_unc[col] =  Variable((num_constr, smaller_u_shape))
                for idx in range(num_constr):
                    if uvar.uncertainty_set.a is not None:
                        aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                            + uvar.uncertainty_set.a.T@supp_cons[col][idx] \
                            + uvar.uncertainty_set.a.T@z_new_cons[col][idx] == \
                            -z_unc[col][idx]]
                    else:
                        aux_constraint += [z_cons + supp_cons[col][idx]\
                            +z_new_cons[col][idx] == -z_unc[col][idx]]

        elif status.get_status()==RemoveUncertaintyMode.Status.HAS_ISOLATED:
            for idx in range(num_constr):
                if uvar.uncertainty_set.a is not None:
                    aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                    + uvar.uncertainty_set.a.T@supp_cons[idx] \
                    +uvar.uncertainty_set.a.T@z_new_cons[idx] == \
                        -z_unc[idx]]
                else:
                    aux_constraint += [z_cons + supp_cons[idx] +
                                    z_new_cons[idx] == -z_unc[idx]]
        else:
            #[0] is needed to make the dimensions match (demote from 2D to 1D) (every time [0])
            if uvar.uncertainty_set.a is not None:
                aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                        + uvar.uncertainty_set.a.T@supp_cons[0] == -z_unc[0]]
            else:
                aux_constraint += [z_cons + supp_cons[0] == -z_unc[0]]
        
        return aux_expr, aux_constraint

    def _conj_and_process(self, status, num_constr_cols, uvar, z_unc, supp_cons, num_constr, z_cons,
                          aux_expr, aux_constraint):
        """
        This is a helper functino that conjugates and prepares aux_expr, aux_constraint, lmbda.
        """
        #If not a matrix, pad in a list so the 0-th element is taken once during the for loop
        if status.get_status() != RemoveUncertaintyMode.Status.HAS_MATRIX:
            z_unc = [z_unc]
            supp_cons = [supp_cons]

        for col in range(num_constr_cols):
            new_expr, new_constraint, lmbda = uvar.conjugate(z_unc[col], supp_cons[col],
                                                             num_constr, k_ind=0)
            if uvar.uncertainty_set.b is not None:
                new_expr = new_expr - uvar.uncertainty_set.b@(z_cons) \
                    - supp_cons[col]@uvar.uncertainty_set.b
            if status.get_status() == RemoveUncertaintyMode.Status.HAS_MATRIX:
                if col < num_constr_cols - 1:
                    #TODO (Irina): Are matrix_expr_to_constr and 
                    matrix_expr_to_constr = [aux_expr[:,col] + new_expr <=0]
                    aux_constraint = aux_constraint + new_constraint \
                        + matrix_expr_to_constr

        if status.get_status() == RemoveUncertaintyMode.Status.HAS_MATRIX:
            return aux_expr[:,col], aux_constraint + new_constraint, lmbda
        
        if status.get_status() == RemoveUncertaintyMode.Status.HAS_MATRIX:
            #TODO (Irina): Originally this read aux_expr[:,col], and col is a iteration index. Was
            #this on purpose? At the last iteration we should get col==num_constr_cols-1
            #(it's not a good habbit using the loop index outside the loop)
            aux_expr = aux_expr[:,num_constr_cols-1] 
        else: 
            aux_expr = aux_expr + new_expr
        aux_constraint = aux_constraint + new_constraint
        return aux_expr, aux_constraint, lmbda

    def remove_uncertainty_expr(self, unc_lst, uvar, num_constr, std_lst):
        """
        TODO Irina, please make sure the docstring is correct.
        This function removes uncertainty from a signle expression.
        """
        u_shape = self.get_u_shape(uvar)
        smaller_u_shape = uvar.uncertainty_set._dimension
        num_unc_expr = len(unc_lst) #Number of uncertain expressions
        #A status object that keeps track of whether we have seen a matrix or isolated
        status = RemoveUncertaintyMode()
        #If the uncertain list is empty,  only need to conjugate the uncertainty set.
        if num_unc_expr==0: 
            aux_expr, aux_constraint, lmbda = uvar.conjugate(u_shape, 0, num_constr, k_ind=0)
            return aux_expr, aux_constraint, lmbda, status

        #Parameter initialization
        #Everything with z is a dual variable
        #If u is a scalar/vector, create a scalar/vector variable. Otherwise, 2D matrix.
        z_dim = num_unc_expr if u_shape==1 else (num_unc_expr, u_shape)
        z = Variable(z_dim)
        z_cons = np.zeros(u_shape)
        z_new_cons = {}
        new_vars = {}
        aux_expr = 0
        aux_constraint = []
        num_constr_cols = None #Default value to be overwritten later
        for ind in range(num_unc_expr):
            curr_has_matrix, curr_has_isolated, num_constr_cols, aux_expr, aux_constraint, uvar, z_cons = \
                        self._update_new_constraint(unc_lst=unc_lst,
                        new_vars=new_vars, z_new_cons=z_new_cons, z_cons=z_cons,
                        num_constr=num_constr, u_shape=u_shape, ind=ind, z=z, uvar=uvar,
                        aux_expr=aux_expr, aux_constraint=aux_constraint)
            #Update has_matrix and has_isolated to True if the current result is True
            status.seen_matrix(curr_has_matrix)
            status.seen_isolated(curr_has_isolated)
            
        #Create new dual variables for the uncertainty set and for the support
        z_unc = Variable((num_constr, smaller_u_shape))
        supp_cons = Variable((num_constr, u_shape))

        aux_expr, aux_constraint = self._gen_aux_constraint(status=status, std_lst=std_lst,
                            num_constr_cols=num_constr_cols, num_constr=num_constr, u_shape=u_shape,
                            smaller_u_shape=smaller_u_shape, uvar=uvar, z_cons=z_cons,
                            z_new_cons=z_new_cons, z_unc=z_unc, supp_cons=supp_cons,
                            aux_expr=aux_expr, aux_constraint=aux_constraint)

        aux_expr, aux_constraint, lmbda = self._conj_and_process(status=status,
                            num_constr_cols=num_constr_cols, uvar=uvar, z_unc=z_unc,
                            supp_cons=supp_cons, num_constr=num_constr, z_cons=z_cons,
                            aux_expr=aux_expr, aux_constraint=aux_constraint)
        return aux_expr, aux_constraint, lmbda, status

    def remove_uncertainty_simple(self, unc_lst, uvar, std_lst, num_constr):
        """
        Canonicalize each term separately with inf convolution:
        It removes the uncertainty in a single constraint.
        This function creates a variable for each expression in a constraint,
        and for the uncertainty set. 
        If there are uncertain parameters after the initial canonicalize_tree call, we 
        deal with the uncertain parameter through an additional procedure.
        After canonicalizing all terms, we return the new expression and constraints.

        Paramteres:
            unc_lst:
                List of uncertain expressions that need to be canonicalized.
            uvar:
                The uncertain parameter.
            std_lst:
                List of expressions that do not involve uncertainty.
            num_constr:
                The dimensions of the constraint.

        Returns:
            canon_constr:
                Reformulated constraint (without uncertainty).
            aux_constraint:
                Additional constraints that are needed.
            lmbda:
                A variable from the reformulation that is required for remove_uncertainty.
        """

        aux_expr, aux_constraint, lmbda, status = self.remove_uncertainty_expr(unc_lst=unc_lst,
                                    uvar=uvar, num_constr=num_constr, std_lst=std_lst)
        #Adding the non uncertain terms to the expression list
        #Amit: For some reason it happens only for non-matrix case?
        if status.get_status() != RemoveUncertaintyMode.Status.HAS_MATRIX:
            for expr in std_lst:
                aux_expr = aux_expr + expr
        return aux_expr <= 0, aux_constraint, lmbda

    def remove_uncertainty_mro(self, unc_lst, uvar, std_lst, num_constr):
        "canonicalize each term separately with inf convolution"
        u_shape = self.get_u_shape(uvar)
        smaller_u_shape = uvar.uncertainty_set._dimension
        num_unc_fns = len(unc_lst)
        if num_unc_fns > 0:
            if u_shape == 1:
                z = Variable(num_unc_fns)
            else:
                z = Variable((num_unc_fns, u_shape))
            z_cons = np.zeros(u_shape)
            z_new_cons = {}
            new_vars = {}
            aux_expr = 0
            aux_constraint = []
            has_isolated = 0
            has_matrix = 0
            for ind in range(num_unc_fns):

                # if len(unc_lst[ind].variables()) (check if has variable)
                u_expr, constant = self.remove_constant(unc_lst[ind])

                # uvar = mul_canon_transform(uvar, cons)
                new_expr, new_constraint = self.canonicalize_tree(
                    u_expr, z[ind], constant)

                if self.has_unc_param(new_expr) and len(unc_lst[ind].shape)==2:
                    uvar = mul_canon_transform(uvar, constant)
                    if self.has_unc_param(new_expr.args[0]):
                        var_mat = new_expr.args[1]
                    elif self.has_unc_param(new_expr.args[1]):
                        var_mat = new_expr.args[0]
                    num_constr_cols = unc_lst[ind].shape[1]
                    total_inds = 0
                    for col in range(num_constr_cols):
                        if ind == 0:
                            z_new_cons[col] = {}
                        new_vars[col] = {}
                        new_vars[col][ind] = Variable((num_constr, u_shape))
                        for idx in range(num_constr):
                            new_expr, new_constraint = \
                                uvar.isolated_unc_matrix(total_inds, \
                                  col, idx, new_vars[col][ind][idx], \
                                    var_mat[col][idx],num_constr_cols,\
                                        num_constr)
                            total_inds += 1
                            aux_expr = aux_expr + new_expr
                            aux_constraint += new_constraint
                            if  ind >=1 :
                                z_new_cons[col][idx] += new_vars[col][ind][idx]
                            else:
                                z_new_cons[col][idx] = new_vars[col][ind][idx]
                    has_matrix = 1

                elif self.has_unc_param(new_expr):
                    uvar = mul_canon_transform(uvar, constant)
                    new_vars[ind] = Variable((num_constr, u_shape))
                    for idx in range(num_constr):
                        new_expr, new_constraint = uvar.isolated_unc(idx, new_vars[ind][idx],
                                                                     num_constr)
                        aux_expr = aux_expr + new_expr
                        aux_constraint += new_constraint
                        if ind == 1:
                            z_new_cons[idx] += new_vars[ind][idx]
                        else:
                            z_new_cons[idx] = new_vars[ind][idx]
                    has_isolated = 1
                else:
                    aux_expr = aux_expr + new_expr
                    aux_constraint += new_constraint
                    z_cons = z_cons + z[ind]

            supp_cons = {}
            z_unc = {}
            if has_matrix == 1:
                for expr in std_lst:
                    aux_expr = aux_expr + expr
                for col in range(num_constr_cols):
                    supp_cons[col] = {}
                    z_unc[col] =  {}
                    for k_ind in range(uvar.uncertainty_set._K):
                        supp_cons[col][k_ind] = Variable((num_constr, u_shape))
                        z_unc[col][k_ind] =  Variable((num_constr, smaller_u_shape))
                        for idx in range(num_constr):
                            if uvar.uncertainty_set.a is not None:
                                aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                                + uvar.uncertainty_set.a.T@supp_cons[col][k_ind][idx] \
                                +uvar.uncertainty_set.a.T@z_new_cons[col][idx] == \
                                    -z_unc[col][k_ind][idx]]
                            else:
                                aux_constraint += [z_cons + supp_cons[col][k_ind][idx]\
                    +z_new_cons[col][idx] == -z_unc[col][k_ind][idx]]

                        new_expr, new_constraint, lmbda, sval = uvar.conjugate(
                        z_unc[col][k_ind], supp_cons[col][k_ind], num_constr, k_ind)
                        if uvar.uncertainty_set.b is not None:
                            new_expr = new_expr - uvar.uncertainty_set.b@(z_cons) \
                                - supp_cons[col][k_ind]@uvar.uncertainty_set.b
                        matrix_expr_to_constr = [aux_expr[:,col] + new_expr <=0]
                        aux_constraint = aux_constraint + new_constraint \
                                + matrix_expr_to_constr
                    if col < num_constr_cols - 1:
                        fin_expr = [uvar.uncertainty_set.rho_mult*\
                                    uvar.uncertainty_set.rho*lmbda +\
                                     uvar.uncertainty_set._w@sval <=0]
                        aux_constraint = aux_constraint + fin_expr
                return uvar.uncertainty_set.rho_mult*\
                    uvar.uncertainty_set.rho*lmbda + \
                    uvar.uncertainty_set._w@sval <=0, aux_constraint,lmbda,sval

            elif has_isolated == 1:
                for k_ind in range(uvar.uncertainty_set._K):
                    z_unc[k_ind] = Variable((num_constr, smaller_u_shape))
                    supp_cons[k_ind] = Variable((num_constr, u_shape))
                    for idx in range(num_constr):
                        if uvar.uncertainty_set.a is not None:
                            aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                            + uvar.uncertainty_set.a.T@supp_cons[k_ind][idx] \
                            +uvar.uncertainty_set.a.T@z_new_cons[idx] == \
                                -z_unc[k_ind][idx]]
                        else:
                            aux_constraint += [z_cons + supp_cons[k_ind][idx] +
                                        z_new_cons[idx] == -z_unc[k_ind][idx]]
            else:
                for k_ind in range(uvar.uncertainty_set._K):
                    z_unc[k_ind] = Variable((num_constr, smaller_u_shape))
                    supp_cons[k_ind] = Variable((num_constr, u_shape))
                    if uvar.uncertainty_set.a is not None:
                        aux_constraint += [uvar.uncertainty_set.a.T@z_cons \
                            + uvar.uncertainty_set.a.T@supp_cons[k_ind][0] == -z_unc[k_ind][0]]
                    else:
                        aux_constraint += [z_cons + supp_cons[k_ind][0] == -z_unc[k_ind][0]]
            for k_ind in range(uvar.uncertainty_set._K):
                new_expr, new_constraint, lmbda, sval = uvar.conjugate(
                    z_unc[k_ind],supp_cons[k_ind], num_constr, k_ind)
                cur_expr = aux_expr + new_expr
                if uvar.uncertainty_set.b is not None:
                    cur_expr = cur_expr - uvar.uncertainty_set.b@(z_cons) \
                        - supp_cons[k_ind]@uvar.uncertainty_set.b
                for expr in std_lst:
                    cur_expr = cur_expr + expr
                aux_constraint = aux_constraint + new_constraint+\
                      [cur_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho_mult*\
                    uvar.uncertainty_set.rho*lmbda +\
                          uvar.uncertainty_set._w@sval
        else:
            aux_constraint = []
            for k_ind in range(uvar.uncertainty_set._K):
                aux_expr, new_constraint, lmbda, sval = uvar.conjugate(
                    u_shape, 0, num_constr, k_ind)
                cur_expr = aux_expr
                for expr in std_lst:
                    cur_expr = cur_expr + expr
                aux_constraint = aux_constraint + \
                    new_constraint + [cur_expr <= 0]
                fin_expr = uvar.uncertainty_set.rho_mult*\
                    uvar.uncertainty_set.rho*lmbda +\
                          uvar.uncertainty_set._w@sval
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

    def separate_uncertainty(self, expr):
        '''separate cvxpy expression into subexpressions with uncertain parameters and without.
        Input:
            expr :
                a cvxpy expression
        Output:
            unc_lst :
                Ex: :math:`[g_1(u_1,x), g_2(u_1,x)]`
                a list of cvxpy multiplication expressions from expr each containing one uncertain
                parameter
            std_lst :
                Ex: :math:`[h_1(x),h_2(x)]`
                any other cvxpy expressions

        The original expr is equivalent to the sum of expressions in unc_lst and std_lst
            '''
        # Check Initial Conditions
        if self.count_unq_uncertain_param(expr) == 0:
            return [], [expr], 0
        # elif self.count_unq_uncertain_param(expr) > 1:
        #     raise ValueError("DRP error: Cannot have multiple uncertain params in the same expr")
        elif len(expr.args) == 0:
            assert (self.has_unc_param(expr))
            return [expr], [], 0

        elif type(expr) not in sep_methods:
            raise ValueError(
                "DRP error: not able to process non multiplication/additions")
        func = sep_methods[type(expr)]
        return func(self, expr)
    
    #TODO (Irina): Please fill/update/assert the docstrings of all the helper functions below
    # (just the description, no need to go over args/returns)
    def remove_uncertainty(self, unc_lst, std_lst, is_max, canon_constraints):
        """
        This function removes uncertainty. See Appendix A.1.3.
        """

        def _gen_merged_unc_lst(is_max, unc_lst):
            """
            This function returns a merged uncertainty list.
            """
            if is_max:
                unc_lst_merged = []
                for cons_idx in range(len(unc_lst)):
                    unc_lst_merged += unc_lst[cons_idx]
            else:
                unc_lst_merged = unc_lst
            return unc_lst_merged
        
        def _calc_constraint_shape(unc_lst_merged):
            """
            This function calculates the shape of the constraint.
            """
            if len(unc_lst_merged[0].shape) >= 1:
                constraint_shape = unc_lst_merged[0].shape[0]
            else:
                constraint_shape = 1
            return constraint_shape

        def _add_canon_constraint(is_mro, canon_constraints, new_cons_idx=None,
                                  lmbda=None, sval=None):
            new_sval = None
            if new_cons_idx is None:
                #not is_max
                unc_lst_pass = unc_lst
                std_lst_pass = std_lst
            else:
                unc_lst_pass = unc_lst[new_cons_idx]
                std_lst_pass = std_lst[new_cons_idx]
            
            if is_mro:
                canon_constr, aux_constr, new_lmbda, new_sval = self.remove_uncertainty_mro(
                    unc_lst_pass, unc_params[0], std_lst_pass, constraint_shape)
            else:
                canon_constr, aux_constr, new_lmbda = self.remove_uncertainty_simple(
                    unc_lst_pass, unc_params[0], std_lst_pass, constraint_shape)
                
            if is_mro and new_cons_idx:
                canon_constraints += aux_constr
                canon_constraints += [lmbda == new_lmbda]
                canon_constraints += [sval == new_sval]
            else:
                canon_constraints += aux_constr + [canon_constr]
            return new_lmbda, new_sval, canon_constr


        unc_lst_merged = _gen_merged_unc_lst(is_max, unc_lst)
        constraint_shape = _calc_constraint_shape(unc_lst_merged)
        unc_params= [v for v in unc_lst_merged[0].parameters() if isinstance(v, UncertainParameter)]
        is_mro = type(unc_params[0].uncertainty_set) == MRO
        new_cons_idx = 0 if is_max else None
        lmbda, sval, canon_constr = _add_canon_constraint(is_mro, canon_constraints, new_cons_idx)
        if is_max:
            for new_cons_idx in range(1, len(unc_lst)):
                lmbda, sval, canon_constr = _add_canon_constraint(is_mro, canon_constraints,
                                                                  new_cons_idx, lmbda, sval)
        return canon_constr

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
