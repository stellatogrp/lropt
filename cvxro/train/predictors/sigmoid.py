import torch

import cvxro.train.settings as settings
from cvxro.train.predictors.nonlinear_base import NonlinearHeadPredictor

torch.set_default_dtype(settings.DTYPE)


class ScaledSigmoid(torch.nn.Module):
    """Elementwise sigmoid(steepness * x + shift), with steepness and shift
    learned per input dimension. Matches the regime-weight nonlinearity
    w(x) = sigmoid(steepness * x + ...) used by
    experiments/portfolio/data_continuous_nonlinear.py's DGP, so a linear
    combination of these features (the Linear layer that follows, in
    SigmoidCovPredictor) can represent that DGP's covariance/mean target
    without the predictor being handed the true steepness -- it is fit like
    any other parameter, not assumed.
    """

    def __init__(self, num_in):
        super().__init__()
        self.log_steepness = torch.nn.Parameter(torch.zeros(num_in, dtype=settings.DTYPE))
        self.shift = torch.nn.Parameter(torch.zeros(num_in, dtype=settings.DTYPE))

    def forward(self, x):
        steepness = torch.exp(self.log_steepness)
        return torch.sigmoid(steepness * x + self.shift)


class SigmoidCovPredictor(NonlinearHeadPredictor):
    """Predictor whose mean AND covariance heads are both fed elementwise
    sigmoid features of the raw context (via ScaledSigmoid) before a final
    linear layer, instead of the raw context (plain linear head), a fixed
    polynomial basis (quad_cov), or a generic dense MLP (MLPCovPredictor).
    """

    def _build_module(self, num_in, out_dim):
        return torch.nn.Sequential(ScaledSigmoid(num_in), torch.nn.Linear(num_in, out_dim))
