from dataclasses import dataclass
from typing import Optional

from cvxpy import settings as s


@dataclass
class SolverStats:
    """Reports some of the miscellaneous information that is returned
    by the solver after solving but that is not captured directly by
    the Problem instance.

    Attributes
    ----------
    solver_name : str
        The name of the solver.
    solve_time : double
        The time (in seconds) it took for the solver to solve the problem.
    setup_time : double
        The time (in seconds) it took for the solver to setup the problem.
    num_iters : int
        The number of iterations the solver had to go through to find a solution.
    extra_stats : object
        Extra statistics specific to the solver; these statistics are typically
        returned directly from the solver, without modification by CVXPY.
        This object may be a dict, or a custom Python object.
    """

    solver_name: str
    solve_time: Optional[float] = None
    setup_time: Optional[float] = None
    num_iters: Optional[int] = None
    extra_stats: Optional[dict] = None

    @classmethod
    def from_dict(cls, attr: dict, solver_name: str) -> "SolverStats":
        """Construct a SolverStats object from a dictionary of attributes.

        Parameters
        ----------
        attr : dict
            A dictionary of attributes returned by the solver.
        solver_name : str
            The name of the solver.

        Returns
        -------
        SolverStats
            A SolverStats object.
        """
        return cls(
            solver_name,
            solve_time=attr.get(s.SOLVE_TIME),
            setup_time=attr.get(s.SETUP_TIME),
            num_iters=attr.get(s.NUM_ITERS),
            extra_stats=attr.get(s.EXTRA_STATS),
        )
