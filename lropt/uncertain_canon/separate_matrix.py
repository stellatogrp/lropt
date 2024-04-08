import numpy as np
from numpy import ndarray
from torch import Tensor
from cvxpy import Variable, problems, SCS, Constant, Zero, NonNeg
from cvxpy.constraints.nonpos import Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.expression import Expression
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy import Parameter
from cvxpy.atoms.affine.hstack import Hstack
from scipy.sparse import csc_matrix, csr_matrix
from cvxpy.constraints.constraint import Constraint
from scipy.sparse._coo import coo_matrix

from lropt.uncertain import UncertainParameter
from lropt.uncertain_canon.atom_canonicalizers import CANON_METHODS as remove_uncertain_methods
from lropt.uncertain_canon.atom_canonicalizers.mul_canon import mul_canon_transform
from lropt.uncertain_canon.remove_constant import REMOVE_CONSTANT_METHODS as rm_const_methods
from lropt.uncertain_canon.separate_uncertainty import SEPARATION_METHODS as sep_methods
from lropt.uncertainty_sets.mro import MRO
from lropt.utils import unique_list
from lropt.uncertain_canon.utils import standard_invert
from lropt.robust_problem import RobustProblem


def tensor_reshaper(T_Ab: coo_matrix, n_var: int) -> np.ndarray:
    """
    This function reshapes T_Ab so T_Ab@param_vec gives the constraints row by row instead of 
    column by column. At the moment, it returns a dense matrix instead of a sparse one.
    """
    def _calc_source_row(target_row: int, num_constraints: int) -> int:
        """
        This is a helper function that calculates the index of the source row of T_Ab for the
        reshaped target row.
        """
        constraint_num = target_row%(num_constraints-1)
        var_num = target_row//(num_constraints-1)
        source_row = constraint_num*num_constraints+var_num
        return source_row


    T_Ab = csc_matrix(T_Ab)
    n_var_full = n_var+1 #Includes the free paramter
    num_rows = T_Ab.shape[0]
    num_constraints = num_rows//n_var_full
    T_Ab_res = csr_matrix(T_Ab.shape)
    target_row = 0 #Counter for populating the new row of T_Ab_res
    for target_row in range(num_rows):
        source_row = _calc_source_row(target_row, num_constraints)
        T_Ab_res[target_row, :] = T_Ab[source_row, :]
    return T_Ab_res

def calc_num_constraints(constraints: list[Constraint]) -> int:
    """
    This function calculates the number of constraints from a list of constraints.
    """
    num_constraints = 0
    for constraint in constraints:
        num_constraints += constraint.size
    return num_constraints

class SeparateMatrix(Reduction):
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
                def _select_target(is_uncertain: bool, param_vec_certain: list,
                                   param_vec_uncertain: list, T_Ab_certain: list,
                                   T_Ab_uncertain: list) -> tuple[list, list]:
                    """
                    This is a helper function that determines whether to add the new parameter and
                    columns to the uncertain parameters or the certain parameters.
                    """
                    if is_uncertain:
                        return param_vec_uncertain, T_Ab_uncertain
                    else:
                        return param_vec_certain, T_Ab_certain
                
                def _gen_param_value(param: Parameter | UncertainParameter, is_uncertain: bool) -> \
                                                        np.ndarray | UncertainParameter:
                    """
                    This is a helper function that returns the uncertain parameter if the input is
                    an uncertain parameter, or the parameter's value for known parameters.
                    """
                    #TODO: Originanlly was the block below, not sure if it should be it or just
                    #param. If just param, this function is redundant.
                    # return param
                    if is_uncertain:
                        return param
                    return param.value
                
                def _safe_np_hstack(vec: list) -> np.ndarray:
                    """
                    This is a helper function that hstacks the elements of vec or returns None if
                    vec is empty.
                    """
                    #Empty vector - return None
                    if not vec:
                        return None
                    #A vector of uncertain parameters needs cvxpy hstack
                    if isinstance(vec[0], UncertainParameter):
                        return Hstack(*vec)
                    return np.hstack(vec)

                def _safe_gen_vecAb(T_Ab: np.ndarray, param_vec: np.ndarray | Expression | None):
                    """
                    This function safely generates vecAb = T_Ab @ vec_param, or returns None if
                    vec_param is empty.
                    """
                    if param_vec is None or (isinstance(param_vec, ndarray) and len(param_vec)==0):
                        return None
                    return T_Ab @ param_vec

                n_var = param_prob.reduced_A.var_len
                T_Ab = param_prob.A
                T_Ab = tensor_reshaper(T_Ab, n_var)
                param_vec_certain   = []
                param_vec_uncertain = []
                T_Ab_certain   = []
                T_Ab_uncertain = []
                running_param_size = 0 #This is a running counter that keeps track of the total size
                                        #of all the parameters seen so far.
                for param in param_prob.parameters:
                    param_size = param_prob.param_id_to_size[param.id]
                    is_uncertain = isinstance(param, UncertainParameter)
                    param_vec_target, T_Ab_target = _select_target(is_uncertain, param_vec_certain,
                                                                   param_vec_uncertain,
                                                                   T_Ab_certain, T_Ab_uncertain)
                    param_val = _gen_param_value(param, is_uncertain)
                    param_vec_target.append(param_val)
                    T_Ab_target.append(T_Ab[:, running_param_size:running_param_size+param_size])
                    running_param_size += param_size
                
                #Add the parameter-free element:
                #The last element is always 1, represents the free element (not a parameter)                  
                param_vec_certain.append(1)
                T_Ab_certain.append(T_Ab[:, running_param_size:])

                #Stack all variables. Certain is never empty - always has the free element
                param_vec_uncertain = _safe_np_hstack(param_vec_uncertain)
                param_vec_certain   = _safe_np_hstack(param_vec_certain)
                T_Ab_uncertain      = _safe_np_hstack(T_Ab_uncertain)
                T_Ab_certain        = _safe_np_hstack(T_Ab_certain)
                vec_Ab_certain      = _safe_gen_vecAb(T_Ab_certain, param_vec_certain)
                vec_Ab_uncertain    = _safe_gen_vecAb(T_Ab_uncertain, param_vec_uncertain)

                return vec_Ab_certain, vec_Ab_uncertain
            
            def _finalize_expressions(vec_Ab: ndarray | Expression, is_uncertain: bool, n_var: int)\
                                                                                         -> tuple:
                """
                This is a helper function that generates A, b from vec_Ab.
                """
                if vec_Ab is None:
                    return None, None
                Ab_dim = (-1, n_var+1) #+1 for the free parameter
                Ab = vec_Ab.reshape(Ab_dim, order='C')
                # note minus sign for different conic form in A_rec
                A_rec = -Ab[:, :-1]
                b_rec = None
                if not is_uncertain:
                    b_rec = Ab[:, -1]
                return A_rec, b_rec

            param_prob = problem.get_problem_data(solver=solver)[0]["param_prob"]
            vec_Ab_certain, vec_Ab_uncertain = _gen_param_vec(param_prob)
            n_var = param_prob.reduced_A.var_len
            A_rec_certain, b_rec = _finalize_expressions(vec_Ab_certain, is_uncertain=False,
                                                         n_var=n_var)
            A_rec_uncertain, _ = _finalize_expressions(vec_Ab_uncertain, is_uncertain=True,
                                                        n_var=n_var)
            return A_rec_certain, A_rec_uncertain, b_rec

        def _gen_objective(problem: RobustProblem) -> Expression:
            #TODO: update this function to reformulate the objective
            return problem.objective
        
        def _gen_constraints(A_rec_certain: ndarray, A_rec_uncertain: Expression, b_rec: ndarray, 
                                        variables: list[Variable]) -> list[Expression]:
            """
            This is a helper function that generates a new constraint.
            """
            s = Variable(A_rec_certain.shape[0])
            variables_stacked = Hstack(*variables)
            constraints = []
            lhs_uncertain = 0 if A_rec_uncertain is None else A_rec_uncertain@variables_stacked
            lhs = A_rec_certain@variables_stacked + lhs_uncertain + s
            # lhs = (A_rec_certain+A_rec_uncertain)@Hstack(*variables) + s
            for i in range(b_rec.size):
                constraints.append(lhs[i] <= b_rec[i])
                constraints[-1].new_name = f"_gen_constraints_<=_i={i}"
                constraints.append(lhs[i] >= b_rec[i])
                constraints[-1].new_name = f"_gen_constraints_>=_i={i}"
            # constraints = [lhs <= b_rec, lhs>=b_rec]
            # constraints[0].new_name = "_gen_constraints_<=" #TODO: DEBUG ONLY
            # constraints[1].new_name = "_gen_constraints_>=" #TODO: DEBUG ONLY
            return constraints, s
        
        def _restore_param_canon(unc_canon_dict: dict) -> None:
            """
            This function restores the canonicalize function of each uncertain parameter.
            """
            for u, orig_canon in unc_canon_dict.values():
                u.canonicalize = orig_canon

        def _gen_basic_problem(problem: RobustProblem, A_rec_certain: ndarray, A_rec_uncertain: Expression, b_rec: ndarray) -> tuple:
            """
            This is a helper function that generates the new problem, new constraints (need to add cone constraints to it), and the new slack variable.
            """
            variables = problem.variables()
            new_objective = _gen_objective(problem)
            new_constraints, s = _gen_constraints(A_rec_certain, A_rec_uncertain, b_rec, variables)
            return new_objective, new_constraints, s

        inverse_data = InverseData(problem)

        #Change uncertain paramter to use its original canonicalize
        unc_canon_dict = _unc_param_to_canon(problem)
        
        #Get A, b tensors (A separated to uncertain and certain parts).
        A_rec_certain, A_rec_uncertain, b_rec = _get_tensors(problem, solver=solver)

        #Change uncertain parameter to use its new canonicalize
        _restore_param_canon(unc_canon_dict)

        #TODO: What do I return? Need help from Irina. Need to verify
        #new objective: c@x
        #new constraint: (A_certain + A_uncertain@u)@x = b
        #Create a new problem with new objective and new constraint

        #TODO: Put this block in a function
        #Create the new problem
        # variables = problem.variables()
        # new_objective = _gen_objective(problem)
        # new_constraints, s = _gen_constraints(A_rec_certain, A_rec_uncertain, b_rec, variables)
        # new_problem = RobustProblem(objective=new_objective, constraints=new_constraints)
        new_objective, new_constraints, s = _gen_basic_problem(problem, A_rec_certain, A_rec_uncertain, b_rec)

        #Add cone constraints TODO: Creata a function for this
        cones = problem.get_problem_data(solver=solver)[0]["dims"]
        if cones.zero > 0:
            new_constraints.append(Zero(s[:cones.zero]))
            new_constraints[-1].new_name = "cones.zero" #TODO: DEBUG ONLY
        if cones.nonneg > 0:
            new_constraints.append(NonNeg(s[cones.zero:cones.zero + cones.nonneg]))
            new_constraints[-1].new_name = "cones.nonneg" #TODO: DEBUG ONLY

        new_problem = RobustProblem(objective=new_objective, constraints=new_constraints)
        #TODO: Update this
        # #Map the ids
        # inverse_data.cons_id_map.update({constraint.id: canon_constr.id})

        return new_problem, inverse_data

    def invert(self, solution, inverse_data):
        #TODO: Go back to this function and make sure it is correct. If it is, need to write a new
        #function and consolidate with uncertain_canonicalization.invert
        return standard_invert(solution=solution, inverse_data=inverse_data)