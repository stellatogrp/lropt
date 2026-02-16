from cvxpy import Constraint, Parameter
from cvxpylayers.torch import CvxpyLayer
from torch import Tensor

from cvxro.parameter import ContextParameter, ShapeParameter
from cvxro.violation_checker.utils import (
    CONSTRAINT_STATUS,
    check_constraint,
    get_batch_size,
    set_cp_parameters,
    set_rho_mult_parameter,
    set_shape_parameters,
    set_x_parameter,
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

    def _set_z_batch(self, z_batch: list[Tensor], batch_number: int) -> None:
        """
        This is an internal function that assigns values to the CVXPYLayer's variables.
        """
        for i, z_tch in enumerate(z_batch):
            var = self._cvxpylayer.variables[i]
            # var = select_batch_object([self._cvxpylayer.variables[i]], var_value, batch_number)[0]
            batch_size = get_batch_size([var], [z_tch])
            z_tch = z_tch[batch_number] if batch_size>1 else z_tch
            val = z_tch.reshape(var.shape).detach().numpy()
            var.value = val

    def _set_parameters(self, rho_mult_parameter: list[Parameter], rho_tch: Tensor,
                            cp_parameters: list[Parameter], cp_param_tch: list[Tensor],
                            x_parameters: list[ContextParameter], x_batch: list[Tensor],
                            shape_parameters: list[ShapeParameter], shape_torches: list[Tensor],
                            batch_number: int) -> None:
        """
        This function sets all the parameters.
        """
        set_rho_mult_parameter(rho_mult_parameter=rho_mult_parameter, rho_tch=rho_tch)
        set_cp_parameters(cp_parameters=cp_parameters, cp_param_tch=cp_param_tch,
                            batch_number=batch_number)
        set_x_parameter(x_parameters=x_parameters, x_batch=x_batch, batch_number=batch_number)
        set_shape_parameters(shape_parameters=shape_parameters, shape_torches=shape_torches,
                            batch_number=batch_number)

    def check_constraints(self, z_batch: list[Tensor], rho_mult_parameter: list[Parameter],
                                    rho_tch: Tensor, cp_parameters: list[Parameter],
                                    cp_param_tch: list[Tensor],
                                    x_parameters: list[ContextParameter],
                                    x_batch: list[Tensor], shape_parameters: list[ShapeParameter],
                                    shape_torches: list[Tensor]) -> CONSTRAINT_STATUS:
        """
        This function checks if the constraints of self.problem_no_unc are violated or not for the
        given z_batch.
        z_batch should be given in the same order as self.problem_no_unc.variables
        (this is the default behavior)

        Args:
            z_batch (list[Tensor]):
                A list of z_batch to check (output of CVXPYLayer call).

        Returns:
            status (CONSTRAINT_STATTUS):
                CONSTRAINT_STATUS.INFEASIBLE if any constraint of problem_no_unc is violated,
                else CONSTRAINT_STATUS.FEASIBLE.
        """
        batch_size = get_batch_size(in_cp=x_parameters, in_tch=x_batch)
        #eps_tch is never batched
        set_rho_mult_parameter(rho_mult_parameter, rho_tch)
        for batch_number in range(batch_size):
            # z_batch, cp_parameters, shape_parameters may be batched,
            # so they need to be in the loop.
            # x_parameter is always batched.
            self._set_z_batch(z_batch, batch_number)
            set_cp_parameters(cp_parameters, cp_param_tch, batch_number)
            set_x_parameter(x_parameters, x_batch, batch_number)
            set_shape_parameters(shape_parameters, shape_torches, batch_number)
            for constraint in self._constraints:
                if check_constraint(constraint)==CONSTRAINT_STATUS.INFEASIBLE:
                    return CONSTRAINT_STATUS.INFEASIBLE
        return CONSTRAINT_STATUS.FEASIBLE
