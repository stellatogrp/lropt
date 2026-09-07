import torch

import cvxro.train.settings as settings
from cvxro.train.predictors.nonlinear_base import NonlinearHeadPredictor

torch.set_default_dtype(settings.DTYPE)


class PerFeatureNet(torch.nn.Module):
    """Independent small MLP per input dimension: y_j = f_j(x_j), each f_j
    fit from data instead of assumed. Generalizes across whatever elementwise
    nonlinearity (or none) governs each context coordinate -- linear,
    quadratic, sigmoid, or otherwise -- without committing to any one family
    the way quad_cov (polynomial) or SigmoidCovPredictor (sigmoid) do.

    Deliberately never mixes dimensions: this matches data_continuous_linear.py
    (Sigma/mu built from W @ w(x) for a purely per-coordinate w(x), no cross
    terms), but NOT data_continuous_nonlinear.py, whose w(x) also depends on
    each coordinate's interaction with every other regime
    (interaction_scale * x_j * sum_{l!=j} x_l). PerFeatureNet is structurally
    blind to that cross term by construction -- no choice of per-dimension
    f_j can recover a function of x_j*x_l -- which is the deliberate point:
    it isolates "does the predictor need to represent cross-regime
    interaction at all" from "does it need the right elementwise
    nonlinearity" (SigmoidCovPredictor's question). MLPCovPredictor can
    represent the cross term (nothing restricts it to per-dimension
    functions), but has to discover it from data unassisted, unlike a
    predictor built with the interaction structure in mind.
    """

    def __init__(self, num_in, hidden=8):
        super().__init__()
        self.nets = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(1, hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden, 1),
            )
            for _ in range(num_in)
        ])

    def forward(self, x):
        return torch.cat([net(x[:, j:j + 1]) for j, net in enumerate(self.nets)], dim=1)


class AdditiveCovPredictor(NonlinearHeadPredictor):
    """Predictor whose mean AND covariance heads are both fed per-dimension
    learned features from PerFeatureNet before a final linear layer, instead
    of the raw context (plain linear head), a fixed polynomial basis
    (quad_cov), a generic dense MLP (MLPCovPredictor), or a hard-coded
    sigmoid (SigmoidCovPredictor). Each head gets its own independent
    PerFeatureNet instance (mean and covariance are fit separately).

    Sits between SigmoidCovPredictor and MLPCovPredictor on the
    structure-vs-generality spectrum: it does not assume the true
    per-coordinate nonlinearity is a sigmoid (or any other specific family),
    so it can fit data_continuous_linear.py's affine coupling or
    data_continuous_nonlinear.py's per-coordinate sigmoid term equally well
    in principle, for both mean and covariance -- but its "no
    cross-coordinate interaction" restriction means it CANNOT represent
    data_continuous_nonlinear.py's cross-regime interaction term (see
    PerFeatureNet's docstring), for either head, unlike MLPCovPredictor's
    fully general dense networks, which can represent that interaction but
    must discover it from data unassisted.
    """

    def __init__(self, per_feature_hidden=8, **kwargs):
        super().__init__(**kwargs)
        self.per_feature_hidden = per_feature_hidden

    def _build_module(self, num_in, out_dim):
        return torch.nn.Sequential(
            PerFeatureNet(num_in, hidden=self.per_feature_hidden),
            torch.nn.Linear(num_in, out_dim),
        )
