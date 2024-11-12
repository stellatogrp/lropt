from cvxpy import Constraint, Parameter
from cvxpylayers.torch import CvxpyLayer
from torch import Tensor

from lropt.train.parameter import ShapeParameter
from lropt.violation_checker.utils import (
    CONSTRAINT_STATUS,
    check_constraint,
    get_batch_size,
    set_orig_parameters,
    set_rho_mult_parameter,
    set_shape_parameters,
    set_y_parameter,
)


class ViolationChecker():
    """
    A constraint checker class.

    Args:
        trainer (Trainer):
            A trainer object.
    """
    def __init__(self, cvxpylayer: CvxpyLayer, constraints: list[Constraint]):
        self._cvxpylayer = cvxpylayer
        self._constraints = constraints

    def _set_var_values(self, var_values: list[Tensor]) -> None:
        """
        This is an internal function that assigns values to the CVXPYLayer's variables.
        """
        for i, var_value in enumerate(var_values):
            var = self._cvxpylayer.variables[i]
            val = var_value.reshape(var.shape).detach().numpy()
            var.value = val

    def _set_parameters(self, rho_mult_parameter: list[Parameter], eps_tch: Tensor,
                            orig_parameters: list[Parameter], y_orig_tch: list[Tensor],
                            y_parameter: list[Parameter], y_batch: list[Tensor],
                            shape_parameters: list[ShapeParameter], shape_torches: list[Tensor],
                            batch_number: int) -> None:
        """
        This function sets all the parameters.
        """
        set_rho_mult_parameter(rho_mult_parameter=rho_mult_parameter, eps_tch=eps_tch)
        set_orig_parameters(orig_parameters=orig_parameters, y_orig_tch=y_orig_tch,
                            batch_number=batch_number)
        set_y_parameter(y_parameter=y_parameter, y_batch=y_batch, batch_number=batch_number)
        set_shape_parameters(shape_parameters=shape_parameters, shape_torches=shape_torches,
                            batch_number=batch_number)

    def check_constraints(self, var_values: list[Tensor], rho_mult_parameter: list[Parameter],
                                    eps_tch: Tensor, orig_parameters: list[Parameter],
                                    y_orig_tch: list[Tensor], y_parameter: list[Parameter],
                                    y_batch: list[Tensor], shape_parameters: list[ShapeParameter],
                                    shape_torches: list[Tensor]) -> CONSTRAINT_STATUS:
        """
        This function checks if the constraints of self.problem_no_unc are violated or not for the
        given var_values.
        var_values should be given in the same order as self.problem_no_unc.variables
        (this is the default behavior)

        Args:
            var_values (list[Tensor]):
                A list of var_values to check (output of CVXPYLayer call).
        
        Returns:
            status (CONSTRAINT_STATTUS):
                CONSTRAINT_STATUS.INFEASIBLE if any constraint of problem_no_unc is violated,
                else CONSTRAINT_STATUS.FEASIBLE.
        """
        self._set_var_values(var_values=var_values)
        batch_size = get_batch_size(in_cp=y_parameter, in_tch=y_batch)
        #eps_tch is never batched
        set_rho_mult_parameter(rho_mult_parameter, eps_tch)
        for batch_number in range(batch_size):
            #orig_parameters, shape_parameters may be batched, so they need to be in the loop.
            #y_parameter is always batched.
            set_orig_parameters(orig_parameters, y_orig_tch, batch_number)
            set_y_parameter(y_parameter, y_batch, batch_number)
            set_shape_parameters(shape_parameters, shape_torches, batch_number)
            for constraint in self._constraints:
                if check_constraint(constraint)==CONSTRAINT_STATUS.INFEASIBLE:
                    return CONSTRAINT_STATUS.INFEASIBLE
        return CONSTRAINT_STATUS.FEASIBLE
        
