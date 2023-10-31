import multiprocessing
import os

import numpy as np


class UncertaintyError(Exception):
    """Error thrown if the uncertain problem has not been formulated correctly."""

    pass


def check_affine_transform(affine_transform):
    assert "b" in affine_transform
    assert "A" in affine_transform


def unique_list(duplicates_list):
    """
    Return unique list preserving the order.
    https://stackoverflow.com/a/480227
    """
    used = set()
    unique = [x for x in duplicates_list if not (x in used or used.add(x))]

    return unique


def cvxpy_to_torch(cvxpy_expr, dec_var_dict, unc_param_dict, fam_param_dict):
    return None

def get_n_processes(max_n=np.inf):

    try:
        # NOTE: only available on some Unix platforms
        n_cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n_cpus = multiprocessing.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc
