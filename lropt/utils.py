from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.problems.problem import Problem
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.reduction import Reduction

from lropt.uncertain_parameter import UncertainParameter


def unique_list(duplicates_list):
    """
    Return unique list preserving the order.
    https://stackoverflow.com/a/480227
    """
    used = set()
    unique = [x for x in duplicates_list if not (x in used or used.add(x))]

    return unique

def gen_and_apply_chain(problem: Problem, reductions: list[Reduction]) -> \
                                                    tuple[Problem, list, Chain]:
    """
    This function generates a chain for a list of reductions, and applies it on the problem.

    Parameters:
        problem (Problem):
            The problem to apply the chain on.
        reductions (list[Reduction]):
            A list containing the reductions to apply on the chain.
    
    Returns:
        new_problem (Problem):
            The new problem after applying the chain.
        inverse_data (list):
            A list containing the inverse data.
        chain (Chain):
            The generated chain.
    """

    chain = Chain(problem, reductions=reductions)
    chain.accepts(problem)
    new_problem, inverse_data = chain.apply(problem)
    return chain, new_problem, inverse_data

def count_unq_uncertain_param(expr: Expression) -> int:
    unc_params = [v for v in expr.parameters() if isinstance(v, UncertainParameter)]
    return len(unique_list(unc_params))

def has_unc_param(expr: Expression) -> bool:
    if not isinstance(expr, int) and not isinstance(expr, float):
        return count_unq_uncertain_param(expr) >= 1
    else:
        return False

def has_unc_param_constraint(constr: Constraint) -> bool:
    """
    This function returns True if the constraint contains uncertain parameters and False otherwise.
    """
    for arg in constr.args:
        if has_unc_param(arg):
            return True
    return False

def cast_as_parent(obj: object) -> object:
    """
    This function creates a new object which is identical to the input with the type of its parent.
    """
    # Create a new object of the parent class
    new_obj = obj.__class__.__bases__[0].__new__(obj.__class__.__bases__[0])
    # Updaet its field to be identical to the object
    new_obj.__dict__.update(obj.__dict__)
    return new_obj