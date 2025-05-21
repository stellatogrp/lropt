import multiprocessing
import os
from enum import Enum
from functools import partial

import numpy as np
import torch
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from torch import Tensor

import lropt.train.settings as settings

EVAL_INPUT_CASE = Enum("_EVAL_INPUT_CASE", "MEAN EVALMEAN MAX CVAR")


def get_n_processes(max_n=np.inf):
    try:
        # NOTE: only available on some Unix platforms
        n_cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n_cpus = multiprocessing.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc


def eval_prob_constr_violation(
    g: list[partial], g_shapes: list[int], batch_int: int, eval_args: list[torch.Tensor]
) -> torch.Tensor:
    """
    This function evaluates the probability of constraint violation of all uncertain constraints
    over the batched set.
    Args:
        g (list[partial]):
            Constraints to evaluate.
        g_shapes (list[int]):
            Shapes of the constraints of g.
        batch_int (int):
            The number of samples in the batch, to take the mean over
        eval_args (list[torch.Tensor]):
            The arguments of the constraints
    Returns:
        The average among all evaluated J x N pairs
    """
    num_g_total = len(g)
    G = torch.zeros((num_g_total, batch_int), dtype=settings.DTYPE)
    for k, g_k in enumerate(g):
        G[sum(g_shapes[:k]) : sum(g_shapes[: (k + 1)])] = eval_input(
            batch_int,
            eval_func=g_k,
            eval_args=eval_args,
            init_val=G[sum(g_shapes[:k]) : sum(g_shapes[: (k + 1)])],
            eval_input_case=EVAL_INPUT_CASE.MAX,
            quantiles=None,
            eta_target = None
        )
    return G.mean(axis=1)


def eval_input(
    batch_int,
    eval_func,
    eval_args,
    init_val,
    eval_input_case,
    quantiles,
    eta_target,
    serial_flag=False,
    **kwargs,
):
    """
    This function takes decision variables, y's, and u's,
        evaluates them and averages them on a given function.

    Args:
        batch_int:
            The number of samples in the batch, to take the mean over
        eval_func:
            The function used for evaluation.
        eval_args:
            The arguments for eval_func
        init_val:
            The placeholder for the returned values
        eval_input_case:
            The type of evaluation performed. Can be MEAN, EVALMEAN, or MAX
        quantiles:
            The quantiles for mean values. can be None.
        serial_flag:
            Whether or not to evalute the function in serial
        kwargs:
            Additional arguments for the eval_func

    Returns:
        The average among all evaluated J x N pairs
    """

    def _serial_eval(batch_int, eval_args, init_val=None, **kwargs):
        """
        This is a helper function that calls eval_func in a serial way.
        """

        def _sample_args(eval_args, sample_ind):
            """
            This is a helper function that samples arguments to be passed to eval_func.
            """
            res = []
            for eval_arg in eval_args:
                curr_arg = eval_arg[sample_ind]
                res.append(curr_arg)
            return res

        curr_result = {}
        for j in range(batch_int):
            curr_eval_args = _sample_args(eval_args, j)
            if init_val:
                init_val[:, j] = eval_func(*curr_eval_args, **kwargs)
            else:
                curr_result[j] = eval_func(*curr_eval_args, **kwargs)
        return curr_result

    if eval_input_case != EVAL_INPUT_CASE.MAX:
        if serial_flag:
            curr_result = _serial_eval(batch_int, eval_args, **kwargs)
        else:
            curr_result = eval_func(*eval_args, **kwargs)
    if eval_input_case == EVAL_INPUT_CASE.MEAN:
        if serial_flag:
            init_val = torch.vstack([curr_result[v] for v in curr_result])
        else:
            init_val = curr_result
        init_val = torch.mean(init_val, axis=0)
    elif eval_input_case == EVAL_INPUT_CASE.EVALMEAN:
        if serial_flag:
            init_val = torch.vstack([curr_result[v] for v in curr_result])
        else:
            init_val = curr_result
        bot_q, top_q = quantiles
        init_val_lower = torch.quantile(init_val, bot_q, axis=0)
        init_val_mean = torch.mean(init_val, axis=0)
        init_val_upper = torch.quantile(init_val, top_q, axis=0)
        return (init_val_lower, init_val_mean, init_val_upper)
    elif eval_input_case == EVAL_INPUT_CASE.CVAR:
        if serial_flag:
            init_val = torch.hstack([curr_result[v] for v in curr_result])
        else:
            init_val = curr_result
        quant = min(len(init_val)-1, int((1-eta_target)*len(init_val)) + 1)
        init_sorted = torch.sort(init_val, descending=False)[0]
        quant = init_sorted[quant]
        init_ge_quant = init_val.ge(quant).float()
        cvar_loss =  init_val.mul(init_ge_quant).sum() / init_ge_quant.sum()
        return (cvar_loss, quant)
    elif eval_input_case == EVAL_INPUT_CASE.MAX:
        # We want to see if there's a violation: either 1 from previous iterations,
        # or new positive value from now
        if serial_flag:
            _ = _serial_eval(batch_int, eval_args, init_val, **kwargs)
        else:
            init_val = eval_func(*eval_args, **kwargs)
            if len(init_val.shape) > 1:
                init_val = init_val.permute(*reversed(range(init_val.ndim))) # replaces init_val.T
        init_val = (init_val > settings.TOL).float()
    return init_val


##########################
# Batch utilitiy functions
##########################


def recursive_apply(expr: Expression, support_types: dict[Atom:Atom]) -> Expression:
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

    # Recursively change all the args of this expression
    args = [recursive_apply(arg, support_types) for arg in expr.args]
    expr.args = args

    # Change this expression if necessary
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
            raise ValueError(
                f"Expected {type(batch_type)} to have 1 parent, but {len(parent_type)} were found."
            )
        return parent_type[0]

    parent_type = _get_parent_type(batch_type)
    # Change this expression if necessary
    if not _should_transform(expr, parent_type):
        return expr
    return batch_type(*batch_type.get_args(expr))


def is_batch(atom: Atom, values: list[Tensor], orig_shape_flag: bool = True) -> bool:
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
    # No batch is equivalent to batch of size 1.
    if not is_batch(arg, [value]):
        return 1

    # The 0-th dimension is always assumed to have the batched data.
    return value.shape[0]


def replace_partial_args(part: partial, new_args: tuple) -> partial:
    """
    This function raplces the args of a partial input with new_args.
    """
    _, _, f = part.__reduce__()
    f, _, k, n = f
    part.__setstate__((f, new_args, k, n))
    return part


##########################
# Optimizer utility functions
##########################


def take_step(
    opt: torch.optim.Optimizer,
    rho_tch: torch.Tensor,
    scheduler: torch.optim.lr_scheduler.StepLR | None,
    update_state = True,
    prev_states = []
) -> list:
    """
    This function performs an optimization step.

    Args:
        opt (torch.optim.Optimizer):
            The optimizer to unroll.
        rho_tch (torch.Tensor):
            rho_tch tensor.
        scheduler (torch.optim_lr_scheduler.StepLR | None):
            If passed, a StepLR scheduler.
    """
    if update_state:
        prev_states = []
        for param in opt.param_groups[0]["params"]:
            if param.grad is not None:
                #new_param = np.array(param.clone().detach().data)
                new_param = param.clone().detach()
                prev_states.append(new_param)
    opt.step()
    # opt.zero_grad()
    with torch.no_grad():
        newrho_tch = torch.clamp(rho_tch, min=0.001)
        rho_tch.copy_(newrho_tch)
    if scheduler:
        scheduler.step()
    return prev_states

def undo_step(opt: torch.optim.Optimizer,state) -> None:
    """
    This function undoes the last optimizer step.

    Args:
        opt (torchoptim.Optimizer):
            The optimizer whose step to undo.
    """
    for group in opt.param_groups:
        for ind, param in enumerate(group["params"]):
            if param.grad is not None:
                param.data = state[ind]
                # param.data.add_(param.grad, alpha=-group["lr"])


def reduce_step_size(opt: torch.optim.Optimizer,step_mult) -> None:
    """
    This function reduces the step size of an optimizer.

    Args:
        opt (torchoptim.Optimizer):
            The optimizer whose step to halve.
    """
    for group in opt.param_groups:
        group["lr"] *= step_mult


def restore_step_size(opt: torch.optim.Optimizer, num_steps: int,step_mult) -> None:
    """
    This function restores the optimizer's step size to its original value,
    i.e. multiplies it by (1/step_mult)^num_steps (where num_steps is 0-indexed).

    Args:
        opt (torch.optim.Optimizer):
            The optimizer whose step size to restore.

        num_steps (int):
            The number of times the step size was halved.
    """
    for group in opt.param_groups:
        # More stable than 2**num_steps
        for _ in range(num_steps):
            group["lr"] *= (1/step_mult)
