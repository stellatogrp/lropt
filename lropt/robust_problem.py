import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Optional

import cvxpy as cp
import torch
from cvxpy import error
from cvxpy import settings as s
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.leaf import Leaf
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solution import INF_OR_UNB_MESSAGE
from cvxtorch import TorchExpression

# from pathos.multiprocessing import ProcessPool as Pool
import lropt.train.settings as settings
from lropt.train.batch import batchify
from lropt.train.parameter import ContextParameter
from lropt.uncertain_canon.remove_uncertainty import RemoveUncertainty
from lropt.uncertain_canon.utils import CERTAIN_ID, UNCERTAIN_NO_MAX_ID
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.mro import MRO
from lropt.utils import gen_and_apply_chain

torch.manual_seed(0) #TODO: Remove all seed setters


class RobustProblem(Problem):
    """Create a Robust Optimization Problem with uncertain variables"""

    _EVAL_INPUT_CASE = Enum("_EVAL_INPUT_CASE", "MEAN EVALMEAN MAX")

    def __init__(
        self, objective, constraints,
        eval_exp=None, train_flag=True, cons_data = None, verify_x_parameters: bool = True
    ):
        self._trained = False
        self._values = None
        self._numvars = 0
        super(RobustProblem, self).__init__(objective, constraints)
        self._trained = False
        self._values = None
        self.problem_canon = None #The canonicalized robust problem (has uncertain parameters)
        self.problem_no_unc = None #The counterpart problem without uncertain parameters
        self.inverse_data_canon = None
        self.chain_canon = None
        self._init = None
        self.train_flag = train_flag
        self._solution = None
        self._status = None
        self._cons_data = cons_data

        self.num_xs = self.verify_x_parameters() if verify_x_parameters else None
        self._store_variables_parameters()
        self.eval_exp = eval_exp

    @property
    def trained(self):
        return self._trained

    @property
    def param_values(self):
        return self._values

    def uncertain_parameters(self):
        """Find uncertain (u) parameters"""
        return [v for v in self.parameters() if isinstance(v, UncertainParameter)]

    def x_parameters(self):
        """Find context (x) parameters"""
        return [v for v in self.parameters() if isinstance(v, ContextParameter)]

    def verify_x_parameters(self):
        """
        This function verifies that x and u are in the correct diemsnions.
        """

        x_parameters = self.x_parameters()
        u_parameters = self.uncertain_parameters()
        num_xs = 1
        if len(x_parameters) > 0:
            num_xs = x_parameters[0].data.shape[0]
        #Check that both y and u dimensions are okay
        for params in [x_parameters, u_parameters]:
            for param in params:
                #Fetch the current shape - different from Parameter and UncertainParameter
                if params is x_parameters:
                    curr_shape = param.data.shape[0]
                else:
                    #Skip the check if there is no data
                    if param.uncertainty_set.data is None:
                        continue
                    #Skip the check if _train==False (MRO without training)
                    train_mro = getattr(param.uncertainty_set, "_train", True)
                    if not train_mro:
                        continue
                    curr_shape = param.uncertainty_set.data.shape[0]
                if curr_shape != num_xs:
                    raise ValueError(f"shape inconsistency: expected num_ys={num_xs}, "
                                     f"but got {curr_shape}.")
        return num_xs

    def fg_to_lh(self):
        """
        Returns l and h function pointers.
        Each of them takes a single x,y,u triplet (i.e. one instance of each)
        """
        # TODO (Amit): Change this function name to a better name
        h_funcs = []
        for g in self.g:
            def hg(*args, **kwargs):
                return (torch.maximum(g(*args) - kwargs["alpha"], torch.tensor(0.0,
                                dtype=settings.DTYPE, requires_grad=self.train_flag))/kwargs["eta"])

            h_funcs.append(hg)

        self.h = h_funcs
        self.num_g = len(h_funcs)

    def unpack(self, solution) -> None:
        """Updates the problem state given a Solution.

        Updates problem.status, problem.value and value of primal and dual
        variables. If solution.status is in cvxpy.settins.ERROR, this method
        is a no-op.

        Arguments
        _________
        solution : cvxpy.Solution
            A Solution object.

        Raises
        ------
        ValueError
            If the solution object has an invalid status
        """
        if solution.status in s.SOLUTION_PRESENT:
            for v in self.variables():
                v.save_value(solution.primal_vars[v.id])
            for c in self.constraints:
                if c.id in solution.dual_vars:
                    c.save_dual_value(solution.dual_vars[c.id])
            # Eliminate confusion of problem.value versus objective.value.
            self._value = solution.opt_val

        elif solution.status in s.INF_OR_UNB:
            for v in self.variables():
                v.save_value(None)
            for constr in self.constraints:
                for dv in constr.dual_variables:
                    dv.save_value(None)
            self._value = solution.opt_val
        else:
            raise ValueError("Cannot unpack invalid solution: %s" % solution)

        self._status = solution.status
        self._solution = solution

    def unpack_results_unc(self, solution, chain, inverse_data,solvername) -> None:
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Arguments
        _________
        solution : object
            The solution returned by applying the chain to the problem
            and invoking the solver on the resulting data.
        chain : SolvingChain
            A solving chain that was used to solve the problem.
        inverse_data : list
            The inverse data returned by applying the chain to the problem.

        Raises
        ------
        cvxpy.error.SolverError
            If the solver failed
        """

        solution = chain.invert(solution, inverse_data)
        if solution.status in s.INACCURATE:
            warnings.warn(
                "Solution may be inaccurate. Try another solver, "
                "adjusting the solver settings, or solve with "
                "verbose=True for more information."
            )
        if solution.status == s.INFEASIBLE_OR_UNBOUNDED:
            warnings.warn(INF_OR_UNB_MESSAGE)
        if solution.status in s.ERROR:
            raise error.SolverError(
                    f"Solver {solvername} failed. "
                    "Try another solver, or solve with verbose=True for more "
                    "information.")
        self.unpack(solution)
        self._solver_stats = SolverStats.from_dict(self._solution.attr, solvername)

    def _validate_uncertain_parameters(self):
        """
        This function checks if there are uncertain parameters.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError if there are no uncertain parameters
        """
        if self.uncertain_parameters() is None:
            raise ValueError("The problem has no uncertain parameters")

    def _store_variables_parameters(self):
        """
        This is an internal function that generates a dictionary of all the variables and parameters
        of the problem from the objective and the constraints.
        The dictionary's keys are the indeces (in order which they are discovered), and the values
        are the variables or the parameters
        """
        def update_vars_params(expr: Expression | cp.constraints.constraint.Constraint,
                               vars_params: dict):
            """
            This function updates vars_params with all the varaibles and params found in expr.
            """
            def safe_check_in_dict(var, vars_dict):
                """
                This function checks if var is in vars_dict.
                For some reason var in vars_dict fails so we check manually
                """
                for value in vars_dict.values():
                    if (var is value):
                        return True
                return False

            vars_dict = TorchExpression(expr).variables_dictionary
            for var_param in vars_dict.vars_dict:
                if safe_check_in_dict(var_param, vars_params):
                    continue
                vars_params[len(vars_params)] = var_param

        vars_params = dict()
        update_vars_params(expr=self.objective.expr, vars_params=vars_params)
        for constraint in self.constraints:
            update_vars_params(expr=constraint, vars_params=vars_params)
        self.vars_params = vars_params

    def _gen_torch_exp(self, expr: Expression, batch_flag: bool=True):
        """
        This function generates a torch expression to be used by RobustProblem from an expression
        and a vars_dict generated by any cvxpy expression. Also returns a variable indicating
        if this toch_exp has uncertain parameters or not.
        """

        def gen_args_inds_to_pass(vars_params, vars_dict):
            """
            This is a helper function that generates a dictionary from a variable/parameter index
            in vars_params (a dictionary that contains all the problem's variables/parameters)
            to vars_dict (a dictionary that contains all the expression's variables/parameters)
            """
            args_inds_to_pass = dict()
            for global_ind, var_param in vars_params.items():
                if var_param not in vars_dict.vars_dict.keys():
                    continue
                args_inds_to_pass[global_ind] = vars_dict.vars_dict[var_param]
            return args_inds_to_pass

        def wrapped_function(torch_exp, args_inds_to_pass: dict[int, int], batch_flag: bool, *args):
            """
            This is the function that wraps the torch expression.

            Args:
                torch_exp:
                    A function (partial)
                args_inds_to_pass:
                    A dictionary from index in *args to the args that will be passed.
                    Note that len(args) > len(args_inds_to_pass) is possible.
                batch_flag (bool):
                    Batch mode on/off.
                *args
                    The arguments of torch_exp
            """

            def _safe_increase_axis(expr: Expression, arg_to_orig_axis: dict) -> None:
                """
                This is an internal function that increases expr.axis by 1 if it is not negative.
                It is needed because we add a new dimension that is reserved for batching, and when
                CVXPY atoms are created, they are unaware of that.
                The increase happens only if batch mode is recognized.
                """

                #Recursively increase the axis of the expression
                for arg in expr.args:
                    if isinstance(arg, Leaf):
                        arg_to_orig_axis[arg] = False
                        continue
                    _safe_increase_axis(arg, arg_to_orig_axis)

                if not hasattr(expr, "axis"):
                    arg_to_orig_axis[expr] = False
                    return
                original_axis = expr.axis
                arg_to_orig_axis[expr] = original_axis

                #If axis=None is equivalent to 0. This is needed to make sure numeric functions
                #do not flatten the inputs.
                if expr.axis is None:
                    expr.axis = 0

                if expr.axis>=0:
                    expr.axis += 1

            def _restore_original_axis(expr: Expression, arg_to_orig_axis: dict) -> None:
                """
                This is an internal function restores the original axis to the expression and all
                of its sub expressions.
                """
                for arg in expr.args:
                    #Recursively restore original axis of the subexpressions
                    _restore_original_axis(arg, arg_to_orig_axis)
                    #Restore the original axis of this expression
                original_axis = arg_to_orig_axis[expr]
                if original_axis is not False:
                    expr.axis = original_axis

            args_to_pass = [None]*len(args_inds_to_pass)
            for key, value in args_inds_to_pass.items():
                args_to_pass[value] = args[key]

            #To make sure batched inputs are processed correctly, we need to update expr.axis
            #(if applicable). It is important to revert it back to the original value when done,
            #hence we save original_axis.
            expr = torch_exp.args[1] #torch_exp.args[1] is the expression
            if batch_flag:
                arg_to_orig_axis = {} #Expression (arg) -> original axis dictionary
                _safe_increase_axis(expr, arg_to_orig_axis)
            res = torch_exp(*args_to_pass)
            #Revert to the original axis if applicable. Note: None is a valid axis (unlike False).
            if batch_flag:
                _restore_original_axis(expr, arg_to_orig_axis)
            return res

        # vars_dict contains a dictionary from variable/param -> index in *args (for the expression)
        # THIS BATCHIFY IS IMPORTANT, BOTH OF THEM ARE NEEDED!
        if batch_flag:
            expr = batchify(expr)
        cvxtorch_exp = TorchExpression(expr)
        torch_exp = cvxtorch_exp.torch_expression
        vars_dict = cvxtorch_exp.variables_dictionary
        # Need to rebatchify because cvxtorch may introduce new unbatched add atoms
        if batch_flag:
            torch_exp = batchify(torch_exp)

        # Create a dictionary from index -> variable/param (for the problem)
        args_inds_to_pass = gen_args_inds_to_pass(self.vars_params, vars_dict)

        return partial(wrapped_function, torch_exp, args_inds_to_pass, batch_flag)

    def _gen_all_torch_expressions(self, eval_exp: Expression | None = None):
        """
        This function generates torch expressions for the canonicalized objective and constraints.
        """
        self.f = self._gen_torch_exp(self.objective.expr)
        self.g = []
        self.g_shapes = []
        self.num_g_total = 0
        self.constraint_checkers = []
        #For each max_id, select all the constraints with this max_id:
        #   Each of these constraints is NonPos, so constraint.args[0] is an expression.
        #   Create an object (list) that has all constraint.args[0] from all these constraints.
        #   new_constraint = cp.NonPos(cp.Maximum(all of these expressions))
        #   new_constraint is fed to _gen_torch_exp, but I shouldn't modify self.constraints
        for max_id in self.constraints_by_type.keys():
            if max_id==CERTAIN_ID: #Nothing to do with certain constraints
                continue
            elif max_id==UNCERTAIN_NO_MAX_ID:
                constraints = self.constraints_by_type[max_id]
            else:
                #Create a constraint from all the constraints of this max_id
                args = [constraint.args[0] for constraint in self.constraints_by_type[max_id]]
                constraints = [cp.NonPos(cp.maximum(*args))]
            for constraint in constraints: #NOT self.constraints: these are the new constraints
                g = self._gen_torch_exp(constraint)
                self.g.append(g) #Always has uncertainty, no need to check
                if len(constraint.shape) >= 1:
                    self.g_shapes.append(constraint.shape[0])
                    self.num_g_total += constraint.shape[0]
                else:
                    self.g_shapes.append(1)
                    self.num_g_total += 1

        if self.eval_exp is None:
            self.eval_exp = self.objective.expr
        # self.eval_exp = eval_exp #This is needed for when RobustProblem() is called in a reduction
        self.eval = self._gen_torch_exp(self.eval_exp, batch_flag=False)
        # self.eval = self.f #This function should be called on the canonicalized problem, so this
                            #instance of self.eval should not be used.
        self.fg_to_lh()

    def remove_uncertainty(self,override = False, solver = None):
        """
        This function canonizes a problem and saves it to self.problem_no_unc

        Args:

        override
            If True, will override current problem_no_unc.
            If False and problem_no_unc exists, does nothing.

        Returns:

        None
        """
        def _uncertain_canonicalization(problem: RobustProblem) -> tuple:
            """
            This helper function applies FlipObjective and UncertainCanonicalization steps.

            Parameters:
                problem (RobustProblem):
                    This robust problem.

            Returns:
                chain_canon (Chain):
                    The constructed reduction chain.
                problem_canon (RobustProblem):
                    The canonicalized robust problem.
                inverse_data
            """
            reductions_canon = []
            if isinstance(problem.objective, Maximize):
                #If maximization problem, flip to minimize
                reductions_canon += [FlipObjective()]
            reductions_canon += [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            chain_canon, problem_canon, inverse_data_canon = gen_and_apply_chain(problem=problem,
                                                                        reductions=reductions_canon)
            # problem_canon.eval = problem.eval #The evaluation expression is not canonicalized
            return chain_canon, problem_canon, inverse_data_canon

        from lropt.uncertain_canon.flip_objective import FlipObjective
        from lropt.uncertain_canon.remove_uncertain_maximum import RemoveSumOfMaxOfUncertain
        from lropt.uncertain_canon.uncertain_canonicalization import UncertainCanonicalization


        if (not override) and (self.problem_canon):
            return
        self._solver = solver
        if self.uncertain_parameters():
            #Uncertain Canonicalization
            self.chain_canon, self.problem_canon, self.inverse_data_canon = \
                                        _uncertain_canonicalization(self)

            #Generating torch expressions and batchify
            self.problem_canon._gen_all_torch_expressions()
            self.num_g_total = self.problem_canon.num_g_total

            #Removing uncertainty and saving the new problem
            self.chain_no_unc, self.problem_no_unc, self.inverse_data_no_unc = \
                                        gen_and_apply_chain(self.problem_canon,
                                                            reductions=[RemoveUncertainty()])

    def solve(self,
               solver: str = None,
               warm_start: bool = True,
               verbose: bool = False,
               gp: bool = False,
               qcp: bool = False,
               requires_grad: bool = False,
               enforce_dpp: bool = False,
               ignore_dpp: bool = False,
               canon_backend: str | None = None,
               **kwargs):
        """
        This function solves the robust problem, and dualizes it first if it has
        not been dualized

        Returns: the solution to the original problem
        """
        self._solver = solver
        unc_param_lst = self.uncertain_parameters()
        if len(unc_param_lst) >= 1:
            solver_func = self._helper_solve
            if self.problem_canon is None:
                # if no data is passed, no training is needed
                if unc_param_lst[0].uncertainty_set.data is None:
                    self.remove_uncertainty(solver = solver)
                else:
                    from lropt.train.trainer import Trainer
                    # if not MRO set and not trained
                    if not isinstance(unc_param_lst[0].uncertainty_set, MRO):
                        self.trainer = Trainer(self, solver= solver)
                        _ = self.trainer.train()
                        for x in self.x_parameters():
                            x.value = x.data[0]
                    # if MRO set and training needed
                    elif unc_param_lst[0].uncertainty_set._train:
                        self.trainer = Trainer(self, solver= solver)
                        _ = self.trainer.train()
                        for x in self.x_parameters():
                            x.value = x.data[0]
                    else:
                        # if MRO set and no training needed
                        self.remove_uncertainty(solver= solver)
        else:
            solver_func = super(RobustProblem, self).solve
        solver_func(solver=solver, warm_start=warm_start, verbose=verbose, gp=gp, qcp=qcp,
                                            requires_grad=requires_grad, enforce_dpp=enforce_dpp,
                                            ignore_dpp=ignore_dpp, canon_backend=canon_backend,
                                            **kwargs)

    def _helper_solve(self,
                solver: str = None,
                warm_start: bool = True,
                verbose: bool = False,
                gp: bool = False,
                qcp: bool = False,
                requires_grad: bool = False,
                enforce_dpp: bool = False,
                ignore_dpp: bool = False,
                canon_backend: str | None = None,
                **kwargs):
        """
        This function solves the dualized robust problem

        Returns: the solution to the original problem
        """
        prob = self.problem_no_unc
        for x in prob.parameters():
            if x.value is None:
                x.value = x.data[0]
        inverse_data = self.inverse_data_canon
        uncertain_chain = self.chain_canon
        prob.solve(solver,warm_start,verbose,gp,qcp,requires_grad,enforce_dpp,ignore_dpp,canon_backend,**kwargs)
        solvername = prob.solver_stats.solver_name
        solution = prob._solution
        self.unpack_results_unc(solution, uncertain_chain, inverse_data,solvername)
        return self.value

    # def evaluate(self, x: list[torch.Tensor], u: list[torch.Tensor]) -> float:
    #     """
    #     TODO: Irina, add docstring
    #     x (context parameters) Every element is a b x w tensor, b is the batch size, w is the dimension of x.
    #     u (uncertain parameter) Every element is a b x d tensor, b is the batch size, d is the dimension of u.
    #     """

    #     #To generate eval_args:
    #     #Need to take self.variables, x (input dataset - context parameters), u (input dataset - uncertain_parameter),
    #     #And then reorder them (like in what we do in Trainer.order_args but tailored for self.problem_canon.eval)
        
        
    #     eval_input(batch_int=b,
    #                eval_func=self.problem_canon.eval,
    #                eval_args=SEE_ABOVE,
    #                init_val=0,
    #                eval_input_case=Trainer._EVAL_INPUT_CASE.MEAN,
    #                quantiles=None,
    #                serial_flag=False)



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
