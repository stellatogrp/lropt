import torch

from cvxro.train.predictors.nonlinear_base import NonlinearHeadPredictor


class MLPCovPredictor(NonlinearHeadPredictor):
    """Predictor whose mean AND covariance heads are both small dense ReLU
    networks on raw context (Linear -> ReLU -> Linear), instead of the plain
    linear mean / linear-or-featured covariance used elsewhere. No
    elementwise-only restriction (unlike AdditiveCovPredictor's
    PerFeatureNet) and no assumed nonlinearity family (unlike
    SigmoidCovPredictor or quad_cov's polynomial basis) 
    `mlp_layers` sets the number of hidden layers (default 1, the original
    Linear -> ReLU -> Linear).
    Both heads share the constructor's `mlp_hidden` width but are otherwise
    fit independently (separate parameters, separate pretraining targets --
    linear_mean against realized u_train, linear_cov against a kNN-local
    covariance target). All other behavior (forward/param_groups/pretrain
    scheduling/freeze_cov) is inherited from NonlinearHeadPredictor /
    LinearPredictor unchanged.
    """

    def __init__(self, mlp_hidden=32, mlp_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.mlp_hidden = mlp_hidden
        if int(mlp_layers) < 1:
            raise ValueError(f"mlp_layers must be >= 1, got {mlp_layers}")
        self.mlp_layers = int(mlp_layers)

    def _build_module(self, num_in, out_dim):
        layers = [torch.nn.Linear(num_in, self.mlp_hidden), torch.nn.ReLU()]
        for _ in range(self.mlp_layers - 1):
            layers += [torch.nn.Linear(self.mlp_hidden, self.mlp_hidden), torch.nn.ReLU()]
        layers += [torch.nn.Linear(self.mlp_hidden, out_dim)]
        return torch.nn.Sequential(*layers)
