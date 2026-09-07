"""Warm-start behaviour of the nonlinear-head predictors.

`NonlinearHeadPredictor._zero_and_set_bias` must give a head that (a) outputs a
CONSTANT at initialization -- the nonlinear analogue of a plain nn.Linear's
zero-weight/constant-bias warm start -- while (b) keeping its hidden features
INPUT-DEPENDENT, so the head can actually depart from that constant.

(b) is the regression guarded here. An earlier version zeroed every nn.Linear in
the head. For a head whose first stage is itself a bare nn.Linear
(MLPCovPredictor's Linear -> ReLU -> Linear) that made the hidden activations the
input-independent constant relu(b1), so the second layer could only ever fit a
constant until the first layer moved -- and the first layer's gradient is exactly
zero while the second layer's weight is zero. SigmoidCovPredictor and
AdditiveCovPredictor were never affected (their first stage is not a bare
nn.Linear), and this test pins that asymmetry down so the fix cannot regress for
one head while silently changing another.
"""

import pytest
import torch

from cvxro.train.predictors.additive import AdditiveCovPredictor
from cvxro.train.predictors.mlp import MLPCovPredictor
from cvxro.train.predictors.nonlinear_base import NonlinearHeadPredictor
from cvxro.train.predictors.sigmoid import SigmoidCovPredictor

PREDICTORS = [MLPCovPredictor, SigmoidCovPredictor, AdditiveCovPredictor]
NUM_IN, OUT_DIM, BATCH = 5, 12, 64


def _warm_started_head(cls, bias_value=0.25):
    torch.manual_seed(0)
    module = cls()._build_module(NUM_IN, OUT_DIM)
    bias = torch.full((OUT_DIM,), bias_value, dtype=torch.double)
    NonlinearHeadPredictor._zero_and_set_bias(module, bias)
    return module


@pytest.mark.parametrize("cls", PREDICTORS, ids=lambda c: c.__name__)
def test_warm_start_output_is_constant(cls):
    """The head must output exactly the requested constant, for every input."""
    module = _warm_started_head(cls)
    out = module(torch.randn(BATCH, NUM_IN, dtype=torch.double))
    assert torch.allclose(out, torch.full_like(out, 0.25))


@pytest.mark.parametrize("cls", PREDICTORS, ids=lambda c: c.__name__)
def test_hidden_features_stay_input_dependent(cls):
    """The regression: hidden features must vary with the input.

    Without this, a constant-output warm start is also a constant-FUNCTION trap.
    """
    module = _warm_started_head(cls)
    feats = module[0](torch.randn(BATCH, NUM_IN, dtype=torch.double))
    assert feats.std(dim=0).max().item() > 1e-6


@pytest.mark.parametrize("cls", PREDICTORS, ids=lambda c: c.__name__)
def test_head_learns_a_context_dependent_map(cls):
    """After a short fit the head must reproduce most of a context-dependent
    target's variation -- i.e. it genuinely escaped the constant warm start."""
    module = _warm_started_head(cls)
    x = torch.rand(256, NUM_IN, dtype=torch.double) * 2 - 1
    target = torch.sigmoid(3.0 * x[:, :1]).repeat(1, OUT_DIM)
    opt = torch.optim.Adam(module.parameters(), lr=0.01)
    for _ in range(300):
        loss = ((module(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    captured = module(x).std(dim=0).mean() / target.std(dim=0).mean()
    assert captured > 0.5, f"{cls.__name__} captured only {captured:.1%} of target variation"
