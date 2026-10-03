"""The reported violation probability is the full failed-decision event.

Regression tests for the JMLR revision-04 audit finding X2: Avg_prob_* used to be the
certificate event alone (realized residuals, no solve mask, averaged per constraint row),
while feasibility_stats ANDs physical feasibility with the robust-solution check. The two
rates describe non-nested events and neither was P(certificate violated OR failed
decision), the event a guarantee at level eta is about. These tests pin the repaired
conventions on a stub trainer with disjoint physical, epigraph-only and failed-solve
events, and the NaN-residual convention in the shared indicator.
"""
import numpy as np
import pytest
import torch

from cvxro.torch_expression_generator import EXCLUDED_GROUP
from cvxro.train import settings as s
from cvxro.train.trainer import Trainer
from cvxro.train.utils import eval_constr_indicator

N = 10


def _t(values):
    return torch.tensor(np.asarray(values, dtype=float), dtype=s.DTYPE)


def make_stub(phys_resid, epi_resid, costs):
    """A Trainer with two constraint groups: an epigraph-only group and a physical one."""

    class _Canon:
        g = [lambda *a, **k: _t(epi_resid), lambda *a, **k: _t(phys_resid)]
        g_feas = [EXCLUDED_GROUP, None]

    class _Settings:
        target_eta = 0.10
        quantiles = (0.1, 0.9)

    tr = Trainer.__new__(Trainer)
    tr.problem_canon = _Canon()
    tr.g = _Canon.g
    tr.g_shapes = [1, 1]
    tr.settings = _Settings()
    tr.eval = lambda *a, **k: _t(costs)[a[0]] if a else _t(costs)
    return tr


@pytest.fixture
def stub():
    # sample 0: physical violation; sample 1: epigraph-only; samples 2, 3: failed solve
    phys = np.full(N, -1.0)
    phys[0] = 1.0
    epi = np.full(N, -1.0)
    epi[1] = 1.0
    return make_stub(phys, epi, np.arange(N, dtype=float))


@pytest.fixture
def solved():
    out = np.ones(N, dtype=bool)
    out[[2, 3]] = False
    return out


@pytest.mark.core
def test_prob_failure_is_the_union(stub, solved):
    eval_args = [torch.arange(N)]
    # certificate alone: samples 0 and 1
    assert float(stub.prob_failure(N, eval_args)) == pytest.approx(0.2)
    # with the solve check: samples 0, 1, 2, 3
    assert float(stub.prob_failure(N, eval_args, solved=solved)) == pytest.approx(0.4)
    indicator = stub.failure_indicator(N, eval_args, solved=solved)
    assert indicator.tolist() == [True, True, True, True] + [False] * 6


@pytest.mark.core
def test_prob_failure_counts_any_violated_row_once(stub):
    """P(any row violated), not the per-row average: a sample violating both groups is
    one failure, and two samples violating one group each are two."""
    phys = np.full(N, -1.0)
    phys[0] = 1.0
    epi = np.full(N, -1.0)
    epi[0] = 1.0  # same sample violates both rows
    both = make_stub(phys, epi, np.arange(N, dtype=float))
    assert float(both.prob_failure(N, [torch.arange(N)])) == pytest.approx(0.1)


@pytest.mark.core
def test_feasibility_stats_stays_physical_and_solved(stub, solved):
    eval_args = [torch.arange(N)]
    feas, _obj, _cvar, frac_unsolved = Trainer.feasibility_stats(
        stub, N, eval_args, quantiles=(0.1, 0.9), eta=0.10, solved=solved
    )
    # infeasible: the physical violation (0) and the failed solves (2, 3) -- not the
    # epigraph-only sample, which stays a certificate matter
    assert feas == pytest.approx(0.7)
    assert frac_unsolved == pytest.approx(0.2)
    # the full event dominates both reported rates
    full = float(stub.prob_failure(N, eval_args, solved=solved))
    assert full >= 1 - feas - 1e-12 and full >= float(stub.prob_failure(N, eval_args)) - 1e-12


@pytest.mark.core
def test_nonfinite_residuals_are_violations():
    """NaN > tol is False: a NaN or inf residual must count as a violation, never as
    a satisfied constraint."""
    resid = np.full(N, -1.0)
    resid[4] = float("nan")
    resid[5] = float("inf")
    g = [lambda *a, **k: _t(resid)]
    indicator = eval_constr_indicator(g=g, g_shapes=[1], batch_int=N, eval_args=[])
    assert indicator[0].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
