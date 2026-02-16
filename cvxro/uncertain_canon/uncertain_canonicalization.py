import operator
from typing import Union

import cvxpy as cp
import numpy as np
import scipy.sparse as scsparse
from cvxpy import Parameter, Variable
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from numpy import ndarray
from scipy.sparse import csr_matrix

from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_canon.utils import (
    CERTAIN_ID,
    gen_constraint_by_type,
    promote_expr,
    reshape_tensor,
    scalarize,
    standard_invert,
)
from cvxro.uncertain_parameter import UncertainParameter

PARAM_TYPES = (UncertainParameter, ContextParameter, Parameter)
LROPT_PARAMETER_TYPES = (UncertainParameter, ContextParameter)
CERTAIN_PARAMETER_TYPES = (ContextParameter, Parameter)



class UncertainCanonicalization(Reduction):
    def accepts(self,problem):
        return True
    def apply(self, problem: RobustProblem, solver="CLARABEL"):
        """Separate the conic constraint into part with uncertainty and without."""

        def _get_tensors(problem: RobustProblem, solver = "CLARABEL") -> ndarray:
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
                        return cp.hstack([cp.vec(param, order="F") for param in vec])
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
                for param_type in CERTAIN_PARAMETER_TYPES:
                    param_vec_dict[param_type] = _safe_hstack(param_vec_dict[param_type])
                    T_Ab_dict[param_type] = _safe_hstack(T_Ab_dict[param_type])

                vec_Ab_certain       = _safe_gen_vecAb(T_Ab_dict, param_vec_dict, Parameter)
                vec_Ab_certain_param = _safe_gen_vecAb(T_Ab_dict, param_vec_dict, ContextParameter)

                return vec_Ab_certain, vec_Ab_certain_param,\
                      T_Ab_dict[UncertainParameter], \
                        param_vec_dict[UncertainParameter]

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

            def _finalize_expressions_uncertain(T_Ab, n_var):
                """
                This is a helper function that generates dicts A_unc and b_unc.
                A list is generated for each uncertain parameter.
                """
                A_dict = {}
                b_dict = {}
                num_params = len(T_Ab)
                if num_params == 0:
                    return None,None

                num_rows = T_Ab[0].shape[0]
                num_constraints = num_rows//(n_var+1)

                for param in range(num_params):
                    A_dict[param] = []
                    b_dict[param] = []
                    for i in range(num_constraints):
                        cur_T = -T_Ab[param][i*(n_var+1):(i+1)*(n_var+1),:]
                        A_dict[param].append(cur_T[:-1,])
                        b_dict[param].append(cur_T[-1,])
                return A_dict, b_dict

            data = problem.get_problem_data(solver=solver)
            param_prob = data[0]["param_prob"]
            cones = data[0]["dims"]
            canon_variables = param_prob.variables
            vec_Ab_certain, vec_Ab_certain_param, T_Ab_list_unc,\
                  param_vec_list_unc = _gen_param_vec(param_prob)
            n_var = param_prob.reduced_A.var_len
            A_certain, b_certain = _finalize_expressions(vec_Ab_certain, n_var=n_var)
            A_certain_param, b_certain_param = _finalize_expressions(vec_Ab_certain_param,
                                                                     n_var=n_var)
            A_certain_total = promote_expr(A_certain)\
                + promote_expr(A_certain_param)
            b_certain_total = promote_expr(b_certain) \
                + promote_expr(b_certain_param)
            A_uncertain_dict, b_uncertain_dict \
                  = _finalize_expressions_uncertain(T_Ab_list_unc, n_var=n_var)
            return A_certain_total, A_uncertain_dict, b_certain_total,\
                b_uncertain_dict, param_vec_list_unc, cones, canon_variables

        def _gen_objective(problem: RobustProblem) -> Expression:
            #TODO: update this function to reformulate the objective
            return problem.objective

        def _gen_constraints(A_certain: ndarray, A_uncertain,
                        b_certain: ndarray,b_uncertain, uncertain_params,
                        variables: list[Variable], cones,
                        cons_data: dict, initial_index: int)\
                            -> list[Expression]:
            """
            This is a helper function that generates a new constraint.
            Each constraint is associated with a dictionary, cons_data, that
            contains information on the uncertain terms within it.
            """
            def _append_constraint(constraints: list[Constraint], A: any, variables_stacked:
                                   Hstack, b_certain: any, term_unc: any,
                                   term_unc_b: any, cons_case: str,
                                   cons_uncertain_data_dict: dict)\
                                                                                            -> None:
                """
                This is a helper function that appends the i-th constraint.
                """

                cons_uncertain_data_dict['std_lst'] = [A@variables_stacked - b_certain]
                term_unc_b = scalarize(term_unc_b)
                b_certain = scalarize(b_certain)
                expr = A@variables_stacked + term_unc + term_unc_b - b_certain
                constraint = (expr==0) if cons_case=="zero" else (expr<=0)
                constraints += [constraint]

            def _gen_term_unc(cones_zero: int,
                        A_uncertain: np.ndarray,i: int,
                        variables_stacked: Hstack,
                     b_uncertain: np.ndarray,
                     uncertain_params,
                     cons_uncertain_data_dict: dict) -> tuple:
                """
                This is a helper function that generates term_unc and
                  term_unc_b for all uncertain params in the constraint.
                The dictionary cons_uncertain_data_dict is populated with
                these uncertain terms.
                """
                term_unc = 0
                term_unc_b = 0

                if (i<cones_zero) or (A_uncertain is None):
                    return term_unc, term_unc_b

                cons_uncertain_data_dict['unc_param_list'] = []
                cons_uncertain_data_dict['var'] = variables_stacked

                # running number of uncertain parameters in this constraint
                cur_i = 0
                for ind, u in enumerate(uncertain_params):
                    if len(u.shape)!=0:
                        op = operator.matmul
                    else:
                        op = operator.mul

                    A_nnz = (A_uncertain[ind][i].nnz != 0)
                    b_nnz = (b_uncertain[ind][i].nnz != 0)

                    if A_nnz or b_nnz:
                        cons_uncertain_data_dict['unc_param_list'].append(u)
                        cons_uncertain_data_dict[cur_i] = {}
                        cons_uncertain_data_dict[cur_i]['has_uncertain_mult'] = False
                        cons_uncertain_data_dict[cur_i]['has_uncertain_isolated'] = False

                        if A_nnz:
                            term_unc = term_unc + variables_stacked@(op(A_uncertain[ind][i],u))
                            cons_uncertain_data_dict[cur_i]['has_uncertain_mult'] = True
                            cons_uncertain_data_dict[cur_i]['unc_term'] = A_uncertain[ind][i]

                        if b_nnz:
                            term_unc_b = term_unc_b + op(b_uncertain[ind][i],u)
                            cons_uncertain_data_dict[cur_i]['has_uncertain_isolated'] = True
                            cons_uncertain_data_dict[cur_i]['unc_isolated'] = b_uncertain[ind][i]

                        cur_i += 1

                return term_unc, term_unc_b

            variables_stacked = cp.hstack([cp.vec(var, order="F") for var in variables])
            constraints = []
            running_ind = 0
            total_constraint_num = cones.zero + cones.nonneg
            for i in range(total_constraint_num):
                if (i < cones.zero):
                    cons_case = "zero"
                elif (i < (cones.zero + cones.nonneg)):
                    cons_case = "nonneg"

                cons_data[initial_index+i] = {}

                term_unc, term_unc_b = _gen_term_unc(cones_zero=cones.zero,
                                            A_uncertain=A_uncertain, i=i,
                                        variables_stacked=variables_stacked,
                                        b_uncertain=b_uncertain,
                                        uncertain_params = uncertain_params,
                                        cons_uncertain_data_dict=cons_data[initial_index+i])

                _append_constraint(constraints=constraints,
                                    A=A_certain[running_ind],
                                    variables_stacked=variables_stacked,
                                    b_certain=b_certain[running_ind],
                                term_unc=term_unc, term_unc_b=term_unc_b,
                                cons_case=cons_case,
                                cons_uncertain_data_dict=cons_data[initial_index+i])

                running_ind += 1

            return constraints, cons_data, int(total_constraint_num+initial_index)

        def _gen_canon_robust_problem(problem: RobustProblem,
                                      A_certain: ndarray, A_uncertain,
                                    b_certain: ndarray, b_uncertain,
                                      uncertain_params, cones,
                        variables,cons_data: dict, initial_index: int) -> tuple:
            """
            This is a helper function that generates the new problem, new constraints
            (need to add cone constraints to it), and the new slack variable.
            """
            # variables = problem.variables()
            # u = problem.uncertain_parameters()[0]
            new_objective = _gen_objective(problem)
            new_constraints, cons_data_updated, total_cons_num =\
                _gen_constraints(A_certain=A_certain,
                                    A_uncertain=A_uncertain,
                                    b_certain=b_certain,
                                    b_uncertain=b_uncertain,
                                    uncertain_params = uncertain_params,
                                    variables=variables, cones=cones,
                                    cons_data=cons_data,
                                    initial_index=initial_index)
            return new_objective, new_constraints, cons_data_updated, total_cons_num

        def _gen_dummy_problem(objective: Expression,
                        constraints: list[Constraint],
                        cons_data: dict, initial_index: int, solver = solver) \
                                                -> RobustProblem:
            """
            This internal function creates a dummy problem from a given problem and a list of
            constraints.
            """
            dummy_problem = RobustProblem(objective=objective, constraints=constraints, \
                                           verify_x_parameters=False)
            #Get A, b tensors (A separated to uncertain and certain parts).
            A_certain, A_uncertain, b_certain, b_uncertain, uncertain_params,\
                  cones,variables = _get_tensors(dummy_problem, solver=solver)

            new_objective, new_constraints, cons_data_updated, total_cons_num \
                = _gen_canon_robust_problem(dummy_problem,
                                                    A_certain, A_uncertain,
                                                    b_certain,b_uncertain, uncertain_params,
                                                    cones, variables,cons_data,
                                                    initial_index)
            return new_constraints, cons_data_updated, total_cons_num

        inverse_data = InverseData(problem)
        solver = problem._solver if problem._solver is not None else solver
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
                                    initial_index = total_cons_number, solver = solver)
            new_constraints += dummy_constraints
            constraints_by_type[id] = dummy_constraints
            # A_certain, A_uncertain, b_certain, b_uncertain, cones,variables \
            #                                         = _get_tensors(problem, solver=solver)

            # new_objective, new_constraints, cons_data = _gen_canon_robust_problem(problem,
            #                                         A_certain, A_uncertain, b_certain,b_uncertain,
            #                                         cones, variables)
        eval_exp = getattr(problem, "eval_exp", None)
        new_problem = RobustProblem(objective=problem.objective, constraints=new_constraints,
                                                cons_data=cons_data, eval_exp=eval_exp)
        new_problem.constraints_by_type = constraints_by_type
        new_problem.ordered_uncertain_no_max_constraints = \
            problem.ordered_uncertain_no_max_constraints
        new_problem.is_max_constraint = problem.is_max_constraint

        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        return standard_invert(solution=solution, inverse_data=inverse_data)
