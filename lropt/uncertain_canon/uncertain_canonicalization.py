import operator
from typing import Union

import cvxpy as cp
import numpy as np
import scipy.sparse as scsparse
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
from lropt.uncertain_canon.utils import (
    CERTAIN_ID,
    gen_constraint_by_type,
    promote_expr,
    reshape_tensor,
    scalarize,
    standard_invert,
)
from lropt.uncertain_parameter import UncertainParameter

PARAM_TYPES = (UncertainParameter, LroptParameter, Parameter)
LROPT_PARAMETER_TYPES = (UncertainParameter, LroptParameter)
CERTAIN_PARAMETER_TYPES = (LroptParameter, Parameter)



class UncertainCanonicalization(Reduction):
    def accepts(self,problem):
        return True
    def apply(self, problem: RobustProblem, solver=SCS):
        """Separate the conic constraint into part with uncertainty and without."""

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
                    # elif param.value is not None:
                    #     return param.value
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
                    if isinstance(vec[0], LROPT_PARAMETER_TYPES) or \
                        isinstance(vec[0], CERTAIN_PARAMETER_TYPES):
                        return cp.hstack([cp.vec(param) for param in vec])
                    if not scsparse.issparse(vec[0]):
                        return csr_matrix(np.hstack(vec))
                    return scsparse.hstack(vec,format='csr')

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
                    # if param_type == Parameter:
                    #     #No need to check if it's 0 because param_vec is never empty
                    #     if param_vec.size > 1:
                    #         return T_Ab @ param_vec.T
                    #     return T_Ab @ param_vec
                    # elif param_type == LroptParameter:
                    #For LROPT Parameters need to be treated like Uncertain Parameters in
                    #this function.
                    #T_Ab = T_Ab[0]
                    curr_vecAb = T_Ab @ param_vec
                    #For LROPT Parameters, need to pad 1D vectors into 2D vectors.
                    return promote_expr(curr_vecAb)

                n_var = param_prob.reduced_A.var_len
                T_Ab = param_prob.A
                T_Ab = reshape_tensor(T_Ab, n_var)
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
                # note minus sign for different conic form in A
                if not isinstance(Ab, Expression):
                    Ab = Ab.tocsr() #TODO: This changes coo_matrix to csr, might be inefficient
                A_certain = -Ab[:, :-1]
                b_certain = Ab[:, -1]
                return A_certain, b_certain

            def _finalize_expressions_uncertain(T_Ab,n_var):
                """
                This is a helper function that generates A_unc
                """
                if T_Ab is None:
                    return None,None
                num_rows = T_Ab.shape[0]
                num_constraints = num_rows//(n_var+1)
                A_list = []
                b_list = []
                for i in range(num_constraints):
                    cur_T = -T_Ab[i*(n_var+1):(i+1)*(n_var+1),:]
                    A_list.append(cur_T[:-1,])
                    b_list.append(cur_T[-1,])
                return A_list, b_list

            data = problem.get_problem_data(solver=solver)
            param_prob = data[0]["param_prob"]
            cones = data[0]["dims"]
            canon_variables = param_prob.variables
            vec_Ab_certain, vec_Ab_certain_param, T_Ab_dict = _gen_param_vec(param_prob)
            n_var = param_prob.reduced_A.var_len
            A_certain, b_certain = _finalize_expressions(vec_Ab_certain, n_var=n_var)
            A_certain_param, b_certain_param = _finalize_expressions(vec_Ab_certain_param,
                                                                     n_var=n_var)
            A_certain_total = A_certain + promote_expr(A_certain_param)
            b_certain_total = b_certain + promote_expr(b_certain_param)
            A_uncertain, b_uncertain = _finalize_expressions_uncertain(
                                                    T_Ab_dict[UncertainParameter],n_var=n_var)
            return A_certain_total, A_uncertain, b_certain_total, b_uncertain, cones, \
                                                    canon_variables

        def _gen_objective(problem: RobustProblem) -> Expression:
            #TODO: update this function to reformulate the objective
            return problem.objective

        def _gen_constraints(A_certain: ndarray, A_uncertain: Expression,
                        b_certain: ndarray,b_uncertain,
                        variables: list[Variable], cones,
                        cons_data: dict, initial_index: int,
                        u: UncertainParameter)\
                            -> list[Expression]:
            """
            This is a helper function that generates a new constraint.
            Each constraint is associated with a dictionary, cons_data, that
            contains information on the uncertain terms within it.
            """
            def _append_constraint(constraints: list[Constraint], A: any, variables_stacked:
                                   Hstack, b_certain: any, term_unc: any,
                                   term_unc_b: any, cons_case: str,
                                   cons_size: int, cons_uncertain_data_dict: dict)\
                                                                                            -> None:
                """
                This is a helper function that appends the i-th constraint.
                """
                if cons_case == "soc":
                    cons_uncertain_data_dict['std_lst'] = []
                    soc_vec = []
                    for j in range(cons_size):
                        cons_uncertain_data_dict['std_lst'].append(
                            A[j]@variables_stacked - b_certain[j])
                        if isinstance(b_certain[j],csr_matrix):
                            b_term = b_certain[j].toarray()[0][0]
                        else:
                            b_term = scalarize(b_certain[j])
                        soc_vec.append(A[j]@variables_stacked + term_unc[j]
                                       + scalarize(term_unc_b[j])- b_term)
                    # if soc_vec[0].shape == (1,1):
                    #     epi_term = -soc_vec[0][0]
                    # else:
                    #     epi_term = -soc_vec[0]
                    constraints += [cp.SOC(-soc_vec[0], cp.vstack(soc_vec[1:]))]

                else:
                    cons_uncertain_data_dict['std_lst'] = [A@variables_stacked - b_certain]
                    cons_func = cp.Zero if (cons_case == "zero") else cp.NonPos
                    term_unc_b = scalarize(term_unc_b)
                    b_certain = scalarize(b_certain)
                    constraints += [cons_func(A@variables_stacked \
                                    + term_unc + term_unc_b - b_certain)]

            def _gen_term_unc(cones_zero: int, u: UncertainParameter,
                        A_uncertain: np.ndarray,i: int,
                        variables_stacked: Hstack,
                     b_uncertain: np.ndarray,
                     cons_uncertain_data_dict: dict) -> tuple:
                """
                This is a helper function that generates term_unc and term_unc_b.
                The dictionary cons_uncertain_data is populated with these uncertain terms.
                """
                term_unc = 0
                term_unc_b = 0
                if 'has_uncertain_isolated' not in cons_uncertain_data_dict:
                    cons_uncertain_data_dict['has_uncertain_isolated'] = False
                    cons_uncertain_data_dict['has_uncertain_mult'] = False
                    cons_uncertain_data_dict['unc_param'] = u
                if (i<cones_zero) or (A_uncertain is None):
                    return term_unc, term_unc_b

                if len(u.shape)!=0 and u.shape[0]>1:
                    op = operator.matmul
                else:
                    op = operator.mul

                if A_uncertain[i].nnz != 0:
                    term_unc = variables_stacked@(op(A_uncertain[i],u))
                    if cons_uncertain_data_dict['has_uncertain_mult']:
                        cons_uncertain_data_dict['unc_term'].append(A_uncertain[i])
                    else:
                        cons_uncertain_data_dict['has_uncertain_mult'] = True
                        cons_uncertain_data_dict['var'] = variables_stacked
                        cons_uncertain_data_dict['unc_term'] = [A_uncertain[i]]

                if b_uncertain[i].nnz != 0:
                    term_unc_b = op(b_uncertain[i],u)
                    if cons_uncertain_data_dict['has_uncertain_isolated']:
                        cons_uncertain_data_dict['unc_isolated'].append(b_uncertain[i])
                    else:
                        cons_uncertain_data_dict['has_uncertain_isolated'] = True
                        cons_uncertain_data_dict['unc_isolated'] = [b_uncertain[i]]

                return term_unc, term_unc_b

            variables_stacked = cp.hstack([cp.vec(var) for var in variables])
            constraints = []
            running_ind = 0
            total_constraint_num = cones.zero + cones.nonneg + len(cones.soc)
            for i in range(total_constraint_num):
                if (i < cones.zero):
                    cons_case = "zero"
                elif (i < (cones.zero + cones.nonneg)):
                    cons_case = "nonneg"
                elif (i < (total_constraint_num)):
                    cons_case = "soc"
                    cur_size = cones.soc[i-(cones.zero + cones.nonneg)]

                cons_data[initial_index+i] = {}
                if cons_case == "soc":
                    term_unc = []
                    term_unc_b = []
                    for j in range(cur_size):
                        term_unc_temp, term_unc_b_temp \
                            = _gen_term_unc(cones_zero=cones.zero,
                                            u=u,A_uncertain=A_uncertain,
                                        i=int(running_ind+j),
                                        variables_stacked=variables_stacked,
                                         b_uncertain=b_uncertain,
                                         cons_uncertain_data_dict=cons_data[initial_index+i])
                        term_unc.append(term_unc_temp)
                        term_unc_b.append(term_unc_b_temp)

                    _append_constraint(constraints=constraints,
                        A=A_certain[running_ind:(running_ind+cur_size)],
                        variables_stacked=variables_stacked,
                        b_certain=b_certain[running_ind:(running_ind+cur_size)],
                                    term_unc=term_unc, term_unc_b=term_unc_b,
                                    cons_case=cons_case,cons_size=cur_size,
                                    cons_uncertain_data_dict=cons_data[initial_index+i])

                else:
                    term_unc, term_unc_b = _gen_term_unc(cones_zero=cones.zero,
                                                 u=u,
                                             A_uncertain=A_uncertain, i=i,
                                            variables_stacked=variables_stacked,
                                            b_uncertain=b_uncertain,
                                            cons_uncertain_data_dict=cons_data[initial_index+i])

                    _append_constraint(constraints=constraints,
                                       A=A_certain[running_ind],
                                       variables_stacked=variables_stacked,
                                       b_certain=b_certain[running_ind],
                                    term_unc=term_unc, term_unc_b=term_unc_b,
                                    cons_case=cons_case,cons_size=1,
                                    cons_uncertain_data_dict=cons_data[initial_index+i])

                if i < (cones.zero + cones.nonneg):
                    running_ind += 1
                else:
                    running_ind += cur_size

            return constraints, cons_data, int(total_constraint_num+initial_index)

        def _gen_canon_robust_problem(problem: RobustProblem,
                                      A_certain: ndarray, A_uncertain: \
                        Expression, b_certain: ndarray, b_uncertain, cones,
                        variables,cons_data: dict, initial_index: int) -> tuple:
            """
            This is a helper function that generates the new problem, new constraints
            (need to add cone constraints to it), and the new slack variable.
            """
            # variables = problem.variables()
            u = problem.uncertain_parameters()[0]
            new_objective = _gen_objective(problem)
            new_constraints, cons_data_updated, total_cons_num =\
                _gen_constraints(A_certain=A_certain,
                                    A_uncertain=A_uncertain,
                                    b_certain=b_certain,
                                    b_uncertain=b_uncertain,
                                    variables=variables, cones=cones,
                                    cons_data=cons_data,
                                    initial_index=initial_index,u=u)
            return new_objective, new_constraints, cons_data_updated, total_cons_num

        def _gen_dummy_problem(objective: Expression,
                        constraints: list[Constraint],
                        cons_data: dict, initial_index: int) \
                                                -> RobustProblem:
            """
            This internal function creates a dummy problem from a given problem and a list of
            constraints.
            """
            dummy_problem = RobustProblem(objective=objective, constraints=constraints, verify_y_parameters=False)
            #Get A, b tensors (A separated to uncertain and certain parts).
            A_certain, A_uncertain, b_certain, b_uncertain, cones,variables \
                                                = _get_tensors(dummy_problem, solver=solver)

            new_objective, new_constraints, cons_data_updated, total_cons_num \
                = _gen_canon_robust_problem(dummy_problem,
                                                    A_certain, A_uncertain,
                                                    b_certain,b_uncertain,
                                                    cones, variables,cons_data,
                                                    initial_index)
            return new_constraints, cons_data_updated, total_cons_num

        inverse_data = InverseData(problem)

        #TODO (AMIT) WORK HERE!!
        #_get_tensors and _gen_canon_robust_problem need to be called, instead
        # of problem, on a new dummy problem which has the same objective and
        # only one constraint.
        #I create these dummy problems for every uncertain constraint with max,
        # whose constraints are all the previous constraints that have the id
        # of this max uncertain.
        #Then another single dummy problem whose constraints are all the
        # uncertain constraints without max.
        #Total number of dummy problems: #max + 1. I just need to take all the
        # constraints of all the dummy problems.

        # Dictionary to store the uncertainty status and information of each
        # constraint. Index by the constraint number
        cons_data = {}
        total_cons_number = 0
        new_constraints = []
        #constraints_by_type is a dictionary from ID of the uncertain max constraint to all of its
        #constraints. There are two special IDs: UNCERTAIN_NO_MAX_ID and CERTAIN_ID for the list of
        #all uncertain non-max constraints/certain constraints, respectively.
        constraints_by_type = gen_constraint_by_type()
        for id in problem.constraints_by_type.keys():
            if not problem.constraints_by_type[id]: #Nothing to do without constraints
                continue
            if id==CERTAIN_ID:
                dummy_constraints = problem.constraints_by_type[CERTAIN_ID]
                total_cons_number += len(dummy_constraints)
            else:
                dummy_constraints, cons_data, total_cons_number = \
                    _gen_dummy_problem(objective=problem.objective,
                                    constraints=problem.constraints_by_type[id],
                                    cons_data=cons_data,
                                    initial_index = total_cons_number)
            new_constraints += dummy_constraints
            constraints_by_type[id] = dummy_constraints
            # A_certain, A_uncertain, b_certain, b_uncertain, cones,variables \
            #                                         = _get_tensors(problem, solver=solver)

            # new_objective, new_constraints, cons_data = _gen_canon_robust_problem(problem,
            #                                         A_certain, A_uncertain, b_certain,b_uncertain,
            #                                         cones, variables)
        eval_exp = getattr(problem, "eval_exp", None)
        #TODO (AMIT): WORK HERE!!!
        # The constraints of the returned problem is all the constraints of the
        # dummy problem plus the certain constraints. The objective is
        # unchanged.
        new_problem = RobustProblem(objective=problem.objective, constraints=new_constraints,
                                                cons_data=cons_data, eval_exp=eval_exp)
        new_problem.constraints_by_type = constraints_by_type

        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        return standard_invert(solution=solution, inverse_data=inverse_data)
