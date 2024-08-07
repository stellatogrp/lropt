import multiprocessing
import os
from functools import partial

import numpy as np
import torch
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from torch import Tensor


def get_n_processes(max_n=np.inf):

    try:
        # NOTE: only available on some Unix platforms
        n_cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n_cpus = multiprocessing.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc

##########################
#Batch utilitiy functions
##########################

def recursive_apply(expr: Expression, support_types: dict[Atom: Atom]) \
                                                                            -> Expression:
    """
    This is a helper function that recursively applies inner_trans on every atom of expr, if the
    atom is a key in support types.
    
    Parameters:
        expr (Expression)
            A CVXPY Expression whose atoms are recursively transformed using trans if they are
            in support_types
        apply_type (APPLY_TYPE):
            What kind of transformation to apply.
        support_types (dict):
            A dictionary from CVXPY atoms to replace to the new atoms.
    """

    #Recursively change all the args of this expression
    args = [recursive_apply(arg, support_types) for arg in expr.args]
    expr.args = args

    #Change this expression if necessary
    new_type = support_types[type(expr)] if type(expr) in support_types else None

    if not new_type:
        return expr
    
    return new_type.transform(expr)

def inner_transform(expr: Expression, batch_type: type) -> Expression:
    """
    This method returns a new expression where the atom's opertaions are replaced with
    batched operations.
    IMPORTANT: Should be used ONLY if:
    1. The atom is supposed to work on a single tensor (e.g. sllicing) and not on multiple atoms
    (e.g. matrix multiplication) TODO: not sure if this is true?.
    2. The atom has a static get_args function implemented (should be implemented anyway).

    Parameters:
        expr (Expression):
            Input expression
        batch_type (type):
            Batched atom type (e.g. BatchedIndex)

    Returns:
        An expression using the batched atom (if relevant) or the regular one otherwise.
    """

    def _should_transform(expr: Expression, parent_type: type):
        """
        This is a helper function that checks if this expression needs to be transformed
        """

        return isinstance(expr, parent_type)

    def _get_parent_type(batch_type: type):
        """
        This is an inner function that returns the parent type of batch_type, and raises an error
        if there is not exactly 1 such parent.
        """
        parent_type = batch_type.__bases__
        if len(parent_type) != 1:
            raise ValueError(f"Expected {type(batch_type)} to have 1 parent, but {len(parent_type)}"
                            f" were found.")
        return parent_type[0]

    parent_type = _get_parent_type(batch_type)
    #Change this expression if necessary
    if not _should_transform(expr, parent_type):
        return expr
    return batch_type(*batch_type.get_args(expr))

def is_batch(atom: Atom, values: list[Tensor], orig_shape_flag: bool=True) -> bool:
    """
    This is a helper function that returns True if this is batch mode.
    IMPORTANT: Should be used ONLY if:
    1. self._orig_shape was updated (if not updated, orig_shape_flag=pass False)
    2. The atom has only 1 input in values.
    """
    curr_shape = values[0].ndim
    if orig_shape_flag:
        atom_shape = getattr(atom, "_orig_shape", atom.shape)
    else:
        atom_shape = atom.shape
    return ((curr_shape - len(atom_shape)) >= 1) or np.prod(values[0].shape) > np.prod(atom_shape)

def expand_tensor(arg: Tensor, batch_size: int) -> Tensor:
    """
    This function expands a tensor on the 0-th dimension batch_size times.
    It supports sparse and dense tensors.
    This is equivalent to arg.expand(size=([batch_size] + list(arg.shape))) but works on
    sparse matrices as well.

    Parameters:
        arg (Tensor):
            A sparse or desnse tensor.
        batch_sizse (int):
            The number of times to expand the tensor.
    
    Returns:
        Expanded tensor (sparse/dense if the input is sparse/dense).
    """
    return torch.stack([arg for _ in range(batch_size)], dim=0)

def stack_tensor(arg: Tensor, x: int) -> Tensor:
    """
    This function stacks an input tensor x times.

    Parameters:
        arg (Tensor):
            Input tensor to stack.
        x (int):
            How many times to stack the tensor.
    
    Returns:
        A tensor stacked x times.
    """
    res = torch.stack([arg for _ in range(x)])
    if res.ndim == arg.ndim:
        raise RuntimeError(f"stack_tensor failed to increase the ndim of {arg}.")
    return res

def already_padded(arg: Expression, value: Tensor) -> bool:
    """
    This is a helper function that returns True if the value has already been padded.
    It is important to avoid padding the same tensor more than once.
    """
    return len(arg.shape) < len(value.shape)

def get_batch_size(arg: Expression, value: Tensor) -> int:
    """
    This function returns the batch size of the input tensor, or 1 if it is not batched.
    """
    #No batch is equivalent to batch of size 1.
    if not is_batch(arg, [value]):
        return 1

    #The 0-th dimension is always assumed to have the batched data.
    return value.shape[0]
    
def replace_partial_args(part: partial, new_args: tuple) -> partial:
    """
    This function raplces the args of a partial input with new_args.
    """
    _, _, f = part.__reduce__()
    f, _, k, n = f
    part.__setstate__((f, new_args, k, n))
    return part
