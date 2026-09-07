import torch

from cvxro.train.predictors.nonlinear_base import NonlinearHeadPredictor


class MLPCovPredictor(NonlinearHeadPredictor):
    """Predictor whose mean AND covariance heads are both small dense ReLU
    networks on raw context (Linear -> ReLU -> Linear), instead of the plain
    linear mean / linear-or-featured covariance used elsewhere. No
    elementwise-only restriction (unlike AdditiveCovPredictor's
    PerFeatureNet) and no assumed nonlinearity family (unlike
    SigmoidCovPredictor or quad_cov's polynomial basis) 
    Both heads share the constructor's `mlp_hidden` width but are otherwise
    fit independently (separate parameters, separate pretraining targets --
    linear_mean against realized u_train, linear_cov against a kNN-local
    covariance target). All other behavior (forward/param_groups/pretrain
    scheduling/freeze_cov) is inherited from NonlinearHeadPredictor /
    LinearPredictor unchanged.
    """

    def __init__(self, mlp_hidden=32, **kwargs):
        super().__init__(**kwargs)
        self.mlp_hidden = mlp_hidden

    def _build_module(self, num_in, out_dim):
        return torch.nn.Sequential(
            torch.nn.Linear(num_in, self.mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_hidden, out_dim),
        )
