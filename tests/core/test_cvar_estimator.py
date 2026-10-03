"""The empirical CVaR estimator: ties, fractional tail mass, continuity, gradients.

Regression tests for the JMLR revision-04 audit finding X1: the previous convention took
the sorted index min(n-1, int((1-eta)n)+1) and averaged ALL costs at or above that
threshold, which shifted the quantile by one rank and, under ties, swept in excess tied
mass (up to -44% on saved newsvendor figure costs, method-dependent). The estimator is
also the training objective via DefaultSimulator.stage_cost, so these cases pin down the
exact definition: CVaR_eta = min_t t + mean((c - t)_+) / eta.
"""
import numpy as np
import pytest
import torch

from cvxro.train.utils import EVAL_INPUT_CASE, empirical_cvar, eval_input

DTYPE = torch.float64


def _t(values):
    return torch.tensor(np.asarray(values, dtype=float), dtype=DTYPE)


def epigraph_cvar(costs, eta):
    """Brute-force min_t t + mean((c-t)_+)/(eta): a minimizer always sits on a data point."""
    c = np.asarray(costs, dtype=float)
    return min(t + np.maximum(c - t, 0).sum() / (eta * len(c)) for t in np.unique(c))


@pytest.mark.core
@pytest.mark.parametrize(
    "costs, eta, expected",
    [
        ([0.0] * 99 + [10.0], 0.10, 1.0),        # tied mass: old convention returned 0.1
        (list(range(100)), 0.10, 94.5),           # index shift: old convention returned 95
        (list(range(30)), 0.10, 28.0),            # a training batch: old returned 28.5
        ([5.0], 0.10, 5.0),                       # eta*n < 1: the maximum
        ([1.0, 2.0, 3.0, 4.0], 1.0, 2.5),         # eta = 1: the mean
        ([0.0] * 27 + [1.0, 1.0, 3.0], 0.10, 5.0 / 3.0),  # ties at the quantile
    ],
)
def test_known_values(costs, eta, expected):
    cvar, _ = empirical_cvar(_t(costs), eta)
    assert float(cvar) == pytest.approx(expected, rel=1e-12)


@pytest.mark.core
def test_matches_epigraph_definition(rng):
    for n in (17, 30, 100, 1000):
        costs = rng.standard_normal(n)
        for eta in (0.03, 0.10, 0.25, 0.50):
            cvar, _ = empirical_cvar(_t(costs), eta)
            assert float(cvar) == pytest.approx(epigraph_cvar(costs, eta), rel=1e-10)


@pytest.mark.core
def test_matches_epigraph_definition_with_atoms(rng):
    """Discrete costs tie heavily; the estimator must still equal the epigraph value."""
    for n in (30, 100, 400):
        costs = rng.integers(0, 4, size=n).astype(float)
        for eta in (0.05, 0.10, 0.30):
            cvar, _ = empirical_cvar(_t(costs), eta)
            assert float(cvar) == pytest.approx(epigraph_cvar(costs, eta), rel=1e-10)


@pytest.mark.core
def test_continuous_in_the_data():
    """27 zeros + [1-eps, 1, 3]: the old convention jumped from 2 to 5/3 at eps = 0."""
    values = []
    for eps in (1e-2, 1e-6, 1e-12, 0.0):
        cvar, _ = empirical_cvar(_t([0.0] * 27 + [1.0 - eps, 1.0, 3.0]), 0.10)
        values.append(float(cvar))
    assert values[-1] == pytest.approx(5.0 / 3.0, rel=1e-12)
    assert max(values) - min(values) < 1e-2


@pytest.mark.core
def test_fractional_tail_mass():
    # n = 10, eta = 0.25: r = 2.5, the two worst plus half the third.
    costs = _t([float(v) for v in range(10)])
    cvar, _ = empirical_cvar(costs, 0.25)
    assert float(cvar) == pytest.approx((9 + 8 + 0.5 * 7) / 2.5, rel=1e-12)


@pytest.mark.core
def test_var_is_higher_quantile_and_epigraph_minimizer(rng):
    costs = rng.standard_normal(200)
    for eta in (0.05, 0.10, 0.17):
        cvar, var = empirical_cvar(_t(costs), eta)
        # t = VaR attains the epigraph minimum
        attained = var + np.maximum(costs - float(var), 0).sum() / (eta * len(costs))
        assert float(attained) == pytest.approx(float(cvar), rel=1e-10)


@pytest.mark.core
def test_monotone_nonincreasing_in_eta(rng):
    costs = _t(rng.standard_normal(123))
    values = [float(empirical_cvar(costs, eta)[0]) for eta in np.linspace(0.02, 1.0, 40)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


@pytest.mark.core
def test_gradient_is_tail_average_weights(rng):
    """d CVaR / d c_i = 1/(eta n) on the strict tail, fractional on the boundary sample."""
    costs = torch.tensor(np.arange(10, dtype=float), dtype=DTYPE, requires_grad=True)
    cvar, _ = empirical_cvar(costs, 0.25)  # r = 2.5: worst two full, third at half weight
    cvar.backward()
    grad = costs.grad.numpy()
    assert grad[9] == pytest.approx(1 / 2.5)
    assert grad[8] == pytest.approx(1 / 2.5)
    assert grad[7] == pytest.approx(0.5 / 2.5)
    assert np.all(grad[:7] == 0)
    assert grad.sum() == pytest.approx(1.0)


@pytest.mark.core
def test_eval_input_cvar_branch_and_mask():
    costs = _t([0.0] * 99 + [10.0])
    out = eval_input(
        batch_int=100, eval_func=lambda: costs, eval_args=[], init_val=None,
        eval_input_case=EVAL_INPUT_CASE.CVAR, quantiles=None, eta_target=0.10,
    )
    assert float(out[0]) == pytest.approx(1.0)
    # the mask selects samples, so the quantile index is over the selected count
    mask = torch.zeros(100, dtype=torch.bool)
    mask[90:] = True  # nine zeros and the 10
    out = eval_input(
        batch_int=100, eval_func=lambda: costs, eval_args=[], init_val=None,
        eval_input_case=EVAL_INPUT_CASE.CVAR, quantiles=None, eta_target=0.10, mask=mask,
    )
    assert float(out[0]) == pytest.approx(10.0)  # r = 1: the single worst selected cost
    # an empty selection is NaN, never a fallback to all costs
    out = eval_input(
        batch_int=100, eval_func=lambda: costs, eval_args=[], init_val=None,
        eval_input_case=EVAL_INPUT_CASE.CVAR, quantiles=None, eta_target=0.10,
        mask=torch.zeros(100, dtype=torch.bool),
    )
    assert torch.isnan(out[0]) and torch.isnan(out[1])
