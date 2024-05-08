import multiprocessing
import os

import numpy as np


def get_n_processes(max_n=np.inf):

    try:
        # NOTE: only available on some Unix platforms
        n_cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n_cpus = multiprocessing.cpu_count()

    n_proc = max(min(max_n, n_cpus), 1)

    return n_proc
