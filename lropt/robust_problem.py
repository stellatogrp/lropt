import warnings
from enum import Enum

import cvxpy as cp
import torch
from cvxpy import Parameter as OrigParameter
from cvxpy import error
from cvxpy import settings as s
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solution import INF_OR_UNB_MESSAGE
from cvxtorch import TorchExpression

# from pathos.multiprocessing import ProcessPool as Pool
from lropt.solver_stats import SolverStats
from lropt.torch_expression_generator import (
    generate_torch_expressions,
    get_eval_batch_size,
    get_eval_data,
)
from lropt.train.parameter import ContextParameter
from lropt.train.settings import DefaultTrainerSettings
from lropt.train.utils import EVAL_INPUT_CASE, eval_input
from lropt.uncertain_canon.remove_uncertainty import RemoveUncertainty
from lropt.uncertain_canon.utils import CERTAIN_ID, UNCERTAIN_NO_MAX_ID
from lropt.uncertain_parameter import UncertainParameter
from lropt.uncertainty_sets.mro import MRO
from lropt.uncertainty_sets.scenario import Scenario
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

        # Constants for constraint types
        self.CERTAIN_ID = CERTAIN_ID
        self.UNCERTAIN_NO_MAX_ID = UNCERTAIN_NO_MAX_ID

        self.num_xs = self.verify_x_parameters() if verify_x_parameters else None
        self._store_variables_parameters()
        self.eval_exp = eval_exp if eval_exp else self.objective.expr

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
            generate_torch_expressions(self.problem_canon)
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
                elif isinstance(unc_param_lst[0].uncertainty_set, Scenario):
                    self.remove_uncertainty(solver = solver)
                else:
                    from lropt.train.trainer import Trainer
                    # if not MRO set and not trained
                    if not isinstance(unc_param_lst[0].uncertainty_set, MRO):
                        self.trainer = Trainer(self)
                        trainer_settings = DefaultTrainerSettings()
                        _ = self.trainer.train(trainer_settings=
                                               trainer_settings)
                        for x in self.x_parameters():
                            x.value = x.data[0]
                    # if MRO set and training needed
                    elif unc_param_lst[0].uncertainty_set._train:
                        self.trainer = Trainer(self)
                        trainer_settings = DefaultTrainerSettings()
                        _ = self.trainer.train(trainer_settings=
                                               trainer_settings)
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


    def evaluate(self) -> float:
        """
        Evaluates the out-of-sample value of the current solution and
        cvxpy/context parameters, with respect to the input data-set
        of uncertain parameters. The dataset is taken from u.eval_data_sol
        for each uncertain parameter u.
        """
        batch_size = get_eval_batch_size(self)

        #Get TorchExpression related data
        #Not actually torch expression, but partial.
        tch_exp = TorchExpression(self.eval_exp).torch_expression

        res = [None]*batch_size
        for batch_num in range(batch_size):
            eval_args = get_eval_data(self, tch_exp=tch_exp, batch_num=batch_num)
            eval_res = eval_input(batch_int=batch_size,
                       eval_func=tch_exp,
                       eval_args=eval_args,
                       init_val=0,
                       eval_input_case=EVAL_INPUT_CASE.MEAN,
                       quantiles=None,
                       serial_flag=False)
            res[batch_num] = eval_res
        return res

    def order_args(self, z_batch: list[torch.Tensor], x_batch: list[torch.Tensor],
                   u_batch: list[torch.Tensor] | torch.Tensor):
        """
        This function orders z_batch (decisions), x_batch (context), and
        u_batch (uncertainty) according to the order in vars_params.
        """
        args = []
        # self.vars_params is a dictionary, hence unsorted. Need to iterate over it in order
        ind_dict = {
            Variable: 0,
            ContextParameter: 0,
            UncertainParameter: 0,
        }
        args_dict = {
            Variable: z_batch,
            ContextParameter: x_batch,
            UncertainParameter: u_batch,
        }

        for i in range(len(self.vars_params)):
            curr_type = type(self.vars_params[i])
            if curr_type == OrigParameter:
                continue
            # This checks for list/tuple or not, to support the fact that currently
            # u_batch is not a list. Irina said in the future this might change.

            # If list or tuple: append the next element
            if isinstance(args_dict[curr_type], tuple) or isinstance(args_dict[curr_type], list):
                append_item = args_dict[curr_type][ind_dict[curr_type]]
                ind_dict[curr_type] += 1
            # If not list-like (e.g. a tensor), append it
            else:
                append_item = args_dict[curr_type]
            args.append(append_item)

        return args
