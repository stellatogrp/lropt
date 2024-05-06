import operator
from typing import Union

import cvxpy as cp
import numpy as np
from cvxpy import SCS, Parameter, Variable
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from numpy import ndarray
from scipy.sparse import csr_matrix

from lropt import Parameter as LroptParameter
from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.uncertain_canonicalization.utils import (
    promote_expr,
    tensor_reshaper,
)
from lropt.uncertain_canon.utils import standard_invert
from lropt.utils import unique_list

PARAM_TYPES = (UncertainParameter, LroptParameter, Parameter)
LROPT_PARAMETER_TYPES = (UncertainParameter, LroptParameter)
CERTAIN_PARAMETER_TYPES = (LroptParameter, Parameter)



class UncertainCanonicalization(Reduction):
    def apply(self, problem: RobustProblem, solver=SCS):
        """Separate the conic constraint into part with uncertainty and without."""
        def _unc_param_to_canon(problem:RobustProblem) -> dict:
            """
            This function creates a dictionary from uncertain paramater index to the uncertain
            parameter and its original canonicalize method, and changes the canonicalize method
            to the one of cp.Parameter (instead of RobustProblem).
            This is needed because we need to override the canonicalize function of each
            uncertain parameter for the separation, but will need to restore the original
            canonicalize function before returning from apply.
            """
            # Note: I save unc_canon_dict as a dictionary to make sure the keys are immutable
            # objects (integers), even though we do not use the keys at any point.
            unc_canon_dict = dict()
            for i,u in enumerate(problem.uncertain_parameters()):
                unc_canon_dict[i] = (u, u.canonicalize)
                u.canonicalize = super(UncertainParameter, u).canonicalize
            return unc_canon_dict

        def _gen_objective_constraints(problem):
            """
            This function generates canon objective and new constraints
            """
            if self.has_unc_param(problem.objective.expr):
                epigraph_obj = Variable()
                epi_cons = problem.objective.expr <= epigraph_obj
                new_constraints = [epi_cons] + problem.constraints
                canon_objective = cp.Minimize(epigraph_obj)
            else:
                canon_objective = problem.objective
                new_constraints = problem.constraints
            return canon_objective, new_constraints

        def _get_tensors(problem: RobustProblem, solver = SCS) -> ndarray:
            """
            This inner function generates A_tensor: the 3D tensor of A. It also generates b,c
            """
            def _gen_param_vec(param_prob: Reduction) -> list:
                """
                This is a helper function that generates the parameters vector.
                This vector will be multiplied by T_Ab to get a vector containing A and b of the
                reformulated conic problem.
                """
                def _select_target(param: Union[PARAM_TYPES], param_vec_dict: dict,
                                                T_Ab_dict: dict) -> tuple[list, list]:
                    """
                    This is a helper function that determines whether to add the new parameter and
                    columns to the uncertain parameters or the certain parameters.
                    """
                    if isinstance(param, LROPT_PARAMETER_TYPES):
                        return_key = type(param)
                    else:
                        return_key = Parameter

                    return param_vec_dict[return_key], T_Ab_dict[return_key]

                def _gen_param_value(param: Union[PARAM_TYPES]) \
                                        -> np.ndarray | Union[LROPT_PARAMETER_TYPES]:
                    """
                    This is a helper function that returns the uncertain parameter if the input is
                    an uncertain parameter, or the parameter's value for known parameters.
                    """

                    if isinstance(param, LROPT_PARAMETER_TYPES):
                        return param
                    elif param.value is not None:
                        return param.value
                    return param

                def _safe_hstack(vec: list) -> np.ndarray | Hstack:
                    """
                    This is a helper function that hstacks the elements of vec or returns None if
                    vec is empty.
                    """
                    #Empty vector - return None
                    if not vec:
                        return None
                    #A vector of uncertain parameters needs cvxpy hstack
                    if isinstance(vec[0], LROPT_PARAMETER_TYPES):
                        return cp.hstack(vec)
                    return np.hstack(vec)

                def _safe_gen_vecAb(T_Ab_dict: dict, param_vec_dict: dict,
                                    param_type: Union[CERTAIN_PARAMETER_TYPES]):
                    """
                    This function safely generates vecAb = T_Ab @ vec_param, or returns None if
                    vec_param is empty.
                    """

                    T_Ab = T_Ab_dict[param_type]
                    param_vec = param_vec_dict[param_type]
                    if param_vec is None or (isinstance(param_vec, ndarray) and len(param_vec)==0):
                        return None
                    if param_type == Parameter:
                        #No need to check if it's 0 because param_vec is never empty
                        return T_Ab @ param_vec
                    elif param_type == LroptParameter:
                        #For LROPT Parameters need to be treated like Uncertain Parameters in
                        #this function.
                        T_Ab = T_Ab[0]
                        curr_vecAb = T_Ab @ param_vec
                        #For LROPT Parameters, need to pad 1D vectors into 2D vectors.
                        return promote_expr(curr_vecAb)

                n_var = param_prob.reduced_A.var_len
                T_Ab = param_prob.A
                T_Ab = tensor_reshaper(T_Ab, n_var)
                param_vec_dict = {param_type: [] for param_type in PARAM_TYPES}
                T_Ab_dict = {param_type: [] for param_type in PARAM_TYPES}
                running_param_size = 0 #This is a running counter that keeps track of the total size
                                        #of all the parameters seen so far.
                for param in param_prob.parameters:
                    param_size = param_prob.param_id_to_size[param.id]
                    param_vec_target, T_Ab_target = _select_target(param, param_vec_dict, T_Ab_dict)
                    param_val = _gen_param_value(param)
                    param_vec_target.append(param_val)
                    T_Ab_target.append(T_Ab[:, running_param_size:running_param_size+param_size])
                    running_param_size += param_size

                #Add the parameter-free element:
                #The last element is always 1, represents the free element (not a parameter)
                param_vec_dict[Parameter].append(1)
                T_Ab_dict[Parameter].append(T_Ab[:, running_param_size:])

                #Stack all variables. Certain is never empty - always has the free element
                for param_type in PARAM_TYPES:
                    param_vec_dict[param_type] = _safe_hstack(param_vec_dict[param_type])
                    T_Ab_dict[param_type] = _safe_hstack(T_Ab_dict[param_type])
                vec_Ab_certain       = _safe_gen_vecAb(T_Ab_dict, param_vec_dict, Parameter)
                vec_Ab_certain_param = _safe_gen_vecAb(T_Ab_dict, param_vec_dict, LroptParameter)

                return vec_Ab_certain, vec_Ab_certain_param, T_Ab_dict

            def _finalize_expressions(vec_Ab: ndarray | Expression, n_var: int) -> tuple:
                """
                This is a helper function that generates A, b from vec_Ab.
                """
                if vec_Ab is None:
                    return 0, 0
                Ab_dim = (-1, n_var+1) #+1 for the free parameter
                Ab = vec_Ab.reshape(Ab_dim, order='C')
                # note minus sign for different conic form in A_rec
                if not isinstance(Ab, Expression):
                    Ab = Ab.tocsr()
                A_rec = -Ab[:, :-1]
                b_rec = Ab[:, -1]
                return A_rec, b_rec

            def _finalize_expressions_uncertain(T_Ab,n_var):
                """
                This is a helper function that generates A_unc
                """
                if T_Ab is None:
                    return None,None
                # Ab_dim = (-1, n_var+1) #+1 for the free parameter
                num_rows = T_Ab[0].shape[0]
                num_constraints = num_rows//(n_var+1)
                # num = T_Ab[0].shape[1]
                A_list = []
                b_list = []
                for i in range(num_constraints):
                    cur_T = -T_Ab[0][i*(n_var+1):(i+1)*(n_var+1),:]
                    A_list.append(cur_T[:-1,])
                    b_list.append(cur_T[-1,])
                # for i in range(num):
                #     temp = T_Ab[0][:n_var+1,i]
                #     Ab = temp.reshape(Ab_dim, order='C')
                #     Ab = Ab.tocsr()
                #     shape = Ab.shape[0]
                #     A_dic[i] = -Ab[:, :-1]
                #     b_dic[i] = Ab[:, -1]
                # return np.vstack([vstack([A_dic[i][j] \
                #                 for i in range(num)]).T for j in \
                #                     range(shape)]), \
                #                         np.vstack([vstack([b_dic[i][j] \
                #                 for i in range(num)]).T for j in range(shape)])
                return A_list, b_list

            data = problem.get_problem_data(solver=solver)
            param_prob = data[0]["param_prob"]
            cones = data[0]["dims"]
            canon_variables = param_prob.variables
            vec_Ab_certain, vec_Ab_certain_param, T_Ab_dict = _gen_param_vec(param_prob)
            n_var = param_prob.reduced_A.var_len
            A_rec_certain, b_rec = _finalize_expressions(vec_Ab_certain, n_var=n_var)
            A_rec_certain_param, b_rec_param = _finalize_expressions(vec_Ab_certain_param,
                                                                     n_var=n_var)
            A_rec_certain_total = A_rec_certain + promote_expr(A_rec_certain_param)
            b_rec_total = b_rec + promote_expr(b_rec_param)
            A_rec_uncertain, b_unc = _finalize_expressions_uncertain(T_Ab_dict[UncertainParameter],
                                                                     n_var=n_var)
            return A_rec_certain_total, A_rec_uncertain, b_rec_total, b_unc, cones, canon_variables

        def _gen_objective(problem: RobustProblem) -> Expression:
            #TODO: update this function to reformulate the objective
            return problem.objective

        def _gen_constraints(A_rec_certain: ndarray, \
                             A_rec_uncertain: Expression, b_rec: ndarray, b_unc,
                                        variables: list[Variable], cones,\
                                              u: UncertainParameter) \
                                              -> list[Expression]:
            """
            This is a helper function that generates a new constraint.
            """
            def _append_constraint(constraints: list[Constraint], A: csr_matrix,
                                   variables_stacked: Hstack, b_rec: csr_matrix,
                                   term_unc: Expression | int = 0, \
                                    term_unc_b: int = 0,\
                                          zero: bool = False, cons_data: dict = None) -> None:
                """
                This is a helper function that appends the i-th constraint.
                """
                cons_data['has_uncertain_isolated'] = (not isinstance(term_unc_b, int))
                cons_data['has_uncertain_mult'] = (not isinstance(term_unc, int))
                cons_data['unc_lst'] = [term_unc] if cons_data['has_uncertain_mult'] else []
                cons_data['unc_isolated'] = [term_unc_b] if \
                    cons_data['has_uncertain_isolated'] else []
                cons_data['std_lst'] = [A@variables_stacked - b_rec]

                #TODO: update this to the version commented out. We need to edit gen_torch_exp
                if zero:
                    constraints += [A@variables_stacked == b_rec]
                else:
                    constraints += [A@variables_stacked + term_unc + term_unc_b <= b_rec]

                # cons_func = cp.Zero if zero else cp.NonPos
                # constraints += [cons_func(A@variables_stacked + term_unc + term_unc_b - b_rec)]

            def _gen_term_unc(cones_zero: int, u: UncertainParameter, A_rec_uncertain: np.ndarray,
                                                i: int, variables_stacked: Hstack,
                                                b_unc: np.ndarray) -> tuple:
                """
                This is a helper function that generates term_unc and term_unc_b.
                """
                term_unc = 0
                term_unc_b = 0
                if (i<cones_zero) or (A_rec_uncertain is None):
                    return term_unc, term_unc_b

                if len(u.shape)!=0 and u.shape[0]>1:
                    op = operator.matmul
                else:
                    op = operator.mul

                if A_rec_uncertain[i].nnz != 0:
                    term_unc = variables_stacked@(op(A_rec_uncertain[i],u))

                if b_unc[i].nnz != 0:
                    term_unc_b = op(b_unc[i],u)

                return term_unc, term_unc_b

            # s = Variable(A_rec_certain.shape[0])
            variables_stacked = cp.hstack(variables)
            # u = cp.hstack([u])
            constraints = []
            cons_data = {}

            for i in range(cones.zero + cones.nonneg):
                term_unc, term_unc_b = _gen_term_unc(cones_zero=cones.zero, u=u,
                                                     A_rec_uncertain=A_rec_uncertain, i=i,
                                                     variables_stacked=variables_stacked,
                                                     b_unc=b_unc)
                zero = (i < cones.zero)
                cons_data[i] = {}
                _append_constraint(constraints=constraints, A=A_rec_certain[i],
                                   variables_stacked=variables_stacked, b_rec=b_rec[i],
                                   term_unc=term_unc, \
                                    term_unc_b=term_unc_b,zero=zero, \
                                        cons_data = cons_data[i])
            return constraints, cons_data

        def _restore_param_canon(unc_canon_dict: dict) -> None:
            """
            This function restores the canonicalize function of each uncertain parameter.
            """
            for u, orig_canon in unc_canon_dict.values():
                u.canonicalize = orig_canon

        def _gen_canon_robust_problem(problem: RobustProblem, \
                        A_rec_certain: ndarray,A_rec_uncertain: \
                        Expression, b_rec: ndarray, b_unc, cones,
                        variables) -> tuple:
            """
            This is a helper function that generates the new problem, new constraints
            (need to add cone constraints to it), and the new slack variable.
            """
            # variables = problem.variables()
            u = problem.uncertain_parameters()[0]
            new_objective = _gen_objective(problem)
            new_constraints, cons_data = _gen_constraints(A_rec_certain=A_rec_certain,
                                    A_rec_uncertain=A_rec_uncertain, b_rec=b_rec, b_unc=b_unc,
                                    variables=variables, cones=cones, u=u)
            return new_objective, new_constraints, cons_data

        problem = RobustProblem(problem.objective,
                                     problem.constraints)
        inverse_data = InverseData(problem)

        #Change uncertain paramter to use its original canonicalize
        unc_canon_dict = _unc_param_to_canon(problem)

        canon_objective, new_constraints = _gen_objective_constraints(problem)
        epigraph_problem = RobustProblem(canon_objective,new_constraints)

        #Get A, b tensors (A separated to uncertain and certain parts).
        A_rec_certain, A_rec_uncertain, b_rec, b_unc,cones,variables \
            = _get_tensors(epigraph_problem, solver=solver)

        #Change uncertain parameter to use its new canonicalize
        _restore_param_canon(unc_canon_dict)

        #TODO: What do I return? Need help from Irina. Need to verify
        #new objective: c@x
        #new constraint: (A_certain + A_uncertain@u)@x = b
        #Create a new problem with new objective and new constraint


        new_objective, new_constraints, \
            cons_data = _gen_canon_robust_problem(epigraph_problem, \
                                                  A_rec_certain,\
                        A_rec_uncertain, b_rec,b_unc,cones,variables)
        new_problem = RobustProblem(objective=new_objective, \
                                    constraints=new_constraints,\
                                          cons_data = cons_data)
        #TODO: Update this
        # #Map the ids
        # inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        #TODO: Go back to this function and make sure it is correct. If it is, need to write a new
        #function and consolidate with uncertain_canonicalization.invert
        return standard_invert(solution=solution, inverse_data=inverse_data)

    def count_unq_uncertain_param(self, expr):
        unc_params = [v for v in expr.parameters()
                           if isinstance(v, UncertainParameter)]
        return len(unique_list(unc_params))

    def has_unc_param(self, expr):
        if not isinstance(expr, int) and not isinstance(expr, float):
            return self.count_unq_uncertain_param(expr) >= 1
        else:
            return 0
