from cvxpy.reductions.solution import Solution
from cvxpy.reductions.inverse_data import InverseData

def standard_invert(solution: Solution, inverse_data: InverseData) -> Solution:
    """
    This function maps the primal/dual variables of the solution to the ones of original problem.
    Should be used as an invert function of a Reduction object.
    """
    pvars = {vid: solution.primal_vars[vid] for vid in inverse_data.id_map
                if vid in solution.primal_vars}
    dvars = {orig_id: solution.dual_vars[vid]
                for orig_id, vid in inverse_data.cons_id_map.items()
                if vid in solution.dual_vars}

    return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)