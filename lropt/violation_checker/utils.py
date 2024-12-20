from enum import Enum

import numpy as np
from cvxpy import Constraint, Parameter
from torch import Tensor

from lropt.train.parameter import ShapeParameter
from lropt.violation_checker.settings import NO_BATCH, VIOLATION_TOL

CONSTRAINT_STATUS = Enum("CONSTRAINT_STATUS", ["FEASIBLE", "INFEASIBLE"])

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
def set_rho_mult_parameter(rho_mult_parameter: list[Parameter], eps_tch: Tensor) -> None:
    """
    This helper function assigns eps_tch to rho_mult_parameter.
    This function assumes eps_tch is a scalar.
    """
    rho_mult_parameter[0].value = eps_tch.item()

#SOMETIMES BTACHED
@check_empty_tensor
def set_orig_parameters(orig_parameters: list[Parameter], y_orig_tch: list[Tensor],
                                            batch_number: int) -> None:
    """
    This is a helper function that assigns y_orig_tch to orig_parameters.
    """
    for orig_param, orig_tch in zip(orig_parameters, y_orig_tch):
        orig_param_curr, orig_tch_curr = select_batch_object(in_cp=orig_param, in_tch=orig_tch,
                                            batch_number=batch_number)
        orig_param_curr.value = orig_tch_curr.detech().numpy()

#ALWAYS BATCHED
@check_empty_tensor
def set_y_parameter(y_parameter: list[Parameter], y_batch: list[Tensor], batch_number: int) -> None:
    """
    This is a helper function that assigns y_batch[i] to y_parameter[i].
    """
    for curr_y_parameter, curr_y_batch in zip(y_parameter, y_batch):
        curr_y_parameter.value = curr_y_batch.detach().numpy()[batch_number]

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
    return in_cp[0][batch_number], in_tch[0][batch_number]
