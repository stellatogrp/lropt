from cvxpy.problems.problem import Problem
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.reduction import Reduction


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

