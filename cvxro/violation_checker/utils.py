from enum import Enum

import numpy as np
from cvxpy import Constraint, Parameter
from torch import Tensor

from cvxro.parameter import ShapeParameter
from cvxro.violation_checker.settings import NO_BATCH, VIOLATION_TOL

CONSTRAINT_STATUS = Enum("CONSTRAINT_STATUS", ["FEASIBLE", "INFEASIBLE"])

class InfeasibleConstraintException(Exception):
    """
    This exception is thrown when an infeasible constraint is found.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def check_constraint(constraint: Constraint) -> CONSTRAINT_STATUS:
    """
    This is a helper function that converts CVXPY's violation status (integer coded) to
    CONSTRAINT_STATUS.
    zero -> Feasible
    positive -> Infeasible.
    """
    violations = constraint.violation()
    for violation in np.nditer(violations):
        if violation > VIOLATION_TOL:
            # print(violation, VIOLATION_TOL)
            return CONSTRAINT_STATUS.INFEASIBLE
        elif violation<0:
            raise ValueError(f"Unknown violation for constraint {constraint}: Expected non-negative"
                            f"number, got {violation} instead.")
    return CONSTRAINT_STATUS.FEASIBLE

def check_empty_tensor(func: callable) -> callable:
    """
    This is a decorator that checks if the second argument is empty.
    If it is, nothing happens.
    If it's not, do the decorated function.
    """
    def wrapper(*args, **kwargrs):
        if not args[1]:
            return
        func(*args, **kwargrs)
    return wrapper


#NEVER BATCHED
@check_empty_tensor
def set_rho_mult_parameter(rho_mult_parameter: list[Parameter], rho_tch: Tensor) -> None:
    """
    This helper function assigns rho_tch to rho_mult_parameter.
    This function assumes rho_tch is a scalar.
    """
    rho_mult_parameter[0].value = rho_tch.item()

#SOMETIMES BTACHED
@check_empty_tensor
def set_cp_parameters(cp_parameters: list[Parameter], cp_param_tch: list[Tensor],
                                            batch_number: int) -> None:
    """
    This is a helper function that assigns cp_param_tch to cp_parameters.
    """
    for cp_param, cp_tch in zip(cp_parameters, cp_param_tch):
        cp_param_curr, cp_tch_curr = select_batch_object(in_cp=cp_param, in_tch=cp_tch,
                                            batch_number=batch_number)
        cp_param_curr.value = cp_tch_curr.detach().numpy()

#ALWAYS BATCHED
@check_empty_tensor
def set_x_parameter(x_parameter: list[Parameter], x_batch: list[Tensor], batch_number: int) -> None:
    """
    This is a helper function that assigns x_batch[i] to x_parameters[i].
    """
    for curr_x_parameter, curr_x_batch in zip(x_parameter, x_batch):
        curr_x_parameter.value = curr_x_batch.detach().numpy()[batch_number]

#SOMETIMES BATCHED
@check_empty_tensor
def set_shape_parameters(shape_parameters: list[ShapeParameter], shape_torches: list[Tensor],
                                            batch_number: int) -> None:
    """
    This is a helper function that assigns shape_torches to shape_parameters.
    """
    for shape_parameter, shape_torch in zip(shape_parameters, shape_torches):
        #select_batch_object expects lists.
        shape_parameters_curr, shape_torch_curr = select_batch_object(in_cp=[shape_parameter],
                                            in_tch=[shape_torch], batch_number=batch_number)
        shape_parameters_curr.value = shape_torch_curr.detach().numpy()


def get_batch_size(in_cp: list, in_tch: list[Tensor]) -> int:
    """
    This function returns the batch size, or NO_BATCH if in_tch is not batched.
    The batch dimension is always assumed to be the 0-th dimension.

    Args:
        in_cp:
            A list of CVXPY objects.
        in_tch:
            A list of torch.Tensor.

    Returns:
        The batch size (int), or NO_BATCH if the data is not batched.

    Raises:
        ValueError if the CVXPY object has more dimensions than the torch object.
    """

    shape_cp = in_cp[0].shape
    shape_tch = in_tch[0].shape
    len_cp = len(shape_cp)
    len_tch = len(shape_tch)

    if len_cp>len_tch:
        raise ValueError(f"Invalid inputs: The CVXPY object has {len_cp} dimensions, which is "
                         f"greater than the torch's {len_tch}.")

    #If the shapes have the same length, it means the torch is not batched.
    elif len_cp==len_tch:
        return NO_BATCH

    return shape_tch[0]

def select_batch_object(in_cp: list[Parameter], in_tch: list[Tensor], batch_number: int) \
                                                    -> Parameter:
    """
    This function selects the [batch_number] element of in_cp and in_tch if it is batched,
    or returns the only item it has otherwise.
    """
    if get_batch_size(in_cp=in_cp, in_tch=in_tch) is NO_BATCH: #Need "is" instead of "=="
        return in_cp[0], in_tch[0]
    return in_cp[0], in_tch[0][batch_number]
