import numpy as np
import torch

from cvxro.train.predictors.linear import LinearPredictor


def _inverse_softplus(y):
    """x such that softplus(x) == y, for y > 0."""
    return float(np.log(np.expm1(y)))


class ProfilePredictor(LinearPredictor):
    r"""Learn only the conservatism profile,
    :math:`A(x) = \hat\Sigma(x)^{1/2}\mathop{\bf diag}(w(x))`.

    The decision sees a shape matrix only through the margins
    :math:`\rho\|A(x)^{\mathsf T}p_l\|_2`, and since
    :math:`\|A^{\mathsf T}p\|^2 = p^{\mathsf T}(AA^{\mathsf T})p`, only through the quadratic form
    :math:`AA^{\mathsf T}` at the :math:`L` constraint normals. Theorems 1--3 of
    `experiments/inventory/WHY_LRO_BEATS_PRETRAIN.md` say the only thing that
    costs anything is the *dispersion* of the conservatism profile
    :math:`\kappa_l = \|A^{\mathsf T}p_l\| / \|\Sigma^{1/2}p_l\|`; a constant profile is
    absorbed by :math:`\rho`.

    `LinearPredictor(predict_cov=True)` searches for that profile indirectly, through
    an affine map with :math:`n^2 p` parameters fit to an :math:`n(n+1)/2`-dimensional
    covariance target. This class searches for it directly:

    * the shape is anchored to the plug-in estimate :math:`\hat\Sigma(x)^{1/2}`, which
      already carries whatever covariance structure the data has, and
    * the only trainable object is the :math:`n`-vector :math:`w(x)`, which reweights the
      margins per coordinate direction.

    **No oracle is used.** :math:`\hat\Sigma` is the kNN plug-in and the residual
    correction uses this predictor's own fitted ``linear_mean`` (leave-one-out on the kNN
    branch), so the class is deployable as-is. Note what that costs: at
    :math:`w \equiv 1` the profile is identically one *relative to* :math:`\hat\Sigma`,
    not relative to the true :math:`\Sigma`. The true initial profile is
    :math:`\|\hat\Sigma^{1/2\mathsf T}p_l\| / \|\Sigma^{1/2}p_l\|`, i.e. the plug-in's own
    residual distortion, measured at :math:`\Psi_B \approx 1.26` on the inventory
    instance -- better than the affine fit's :math:`1.42`, but not isotropic. Training
    starts at the *corrected plug-in*, not at the ideal mean-variance set.

    What it does buy: the trainable dimension drops from :math:`n^2 p` to :math:`np` (or
    to :math:`n`, with ``profile_contextual=False``), all of it on the axis Theorems 1--3
    identify -- rather than searching for the profile indirectly through a covariance
    fit. And :math:`\mathop{\bf diag}(w)` reweights coordinates, so it cannot repair
    off-diagonal error in :math:`\hat\Sigma`; on this instance that error is 46% of the
    plug-in's mass and is visible to the decision.

    The base estimate comes from the kNN path (`knn_cov=True`, `knn_scale=1.0`), with
    `knn_residual_cov=True` so it is the mean-leakage-corrected estimator. Pass
    ``knn_scale`` below 1 to blend the affine `linear_cov` head back in.
    """

    def __init__(self, profile_contextual=True, profile_init=1.0, **kwargs):
        kwargs.setdefault("knn_cov", True)
        kwargs.setdefault("knn_scale", 1.0)
        kwargs.setdefault("knn_residual_cov", True)
        # linear_cov is multiplied by (1 - knn_scale), so at the default it contributes
        # nothing; freezing it keeps it out of the optimizer as well.
        kwargs.setdefault("freeze_cov", True)
        super().__init__(**kwargs)
        self.profile_contextual = profile_contextual
        self.profile_init = float(profile_init)
        self.profile = None

    def initialize(self, a_tch, b_tch, trainer):
        super().initialize(a_tch, b_tch, trainer)
        num_in, _, b_out = trainer.initialize_predictor_dims()
        in_dim = num_in if self.profile_contextual else 1
        self.profile = torch.nn.Linear(in_dim, b_out, dtype=torch.double)
        # Start at w == profile_init exactly: zero weights, bias at the softplus
        # pre-image. So the initial set is the plug-in one and kappa is whatever the
        # plug-in's is -- identically 1 when the plug-in is the true Sigma^{1/2}.
        torch.nn.init.zeros_(self.profile.weight)
        torch.nn.init.constant_(self.profile.bias, _inverse_softplus(self.profile_init))

    def _profile_weights(self, x):
        """The per-coordinate multipliers w(x), strictly positive."""
        features = x if self.profile_contextual else torch.ones(
            x.shape[0], 1, dtype=x.dtype)
        return torch.nn.functional.softplus(self.profile(features))

    def forward(self, x, a_shape, b_shape, train_flag):
        a_tch, b_tch, radius = super().forward(x, a_shape, b_shape, train_flag)
        if self.profile is None:
            return a_tch, b_tch, radius
        w = self._profile_weights(x)
        if not train_flag:
            w = w.detach().clone()
        # Right-multiplying by diag(w) scales A's COLUMNS, which is what reweights the
        # margins per coordinate direction: (A diag(w))^T p = diag(w) A^T p.
        a_tch = a_tch * w.unsqueeze(1)
        return a_tch, b_tch, radius

    def param_groups(self, base_lr, extra_params=None):
        groups = super().param_groups(base_lr, extra_params=extra_params)
        if self.profile is not None:
            groups.append({"params": list(self.profile.parameters()), "lr": base_lr})
        return groups
