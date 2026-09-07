import torch

import cvxro.train.settings as settings
from cvxro.train.predictors.linear import LinearPredictor

torch.set_default_dtype(settings.DTYPE)


class NonlinearHeadPredictor(LinearPredictor):
    """Shared base for predictor classes whose MEAN head is nonlinear, not
    just the covariance head.

    Subclasses implement `_build_module(num_in, out_dim)` to supply ONE
    architecture used for BOTH linear_mean and linear_cov (a separate
    instance per head, each fit independently) -- a dense ReLU MLP, a
    sigmoid-featured linear head, or a per-coordinate additive net.
    """

    def __init__(self, **kwargs):
        # A nonlinear head consumes the raw context; quad_cov's polynomial feature map
        # is an alternative to nonlinearity, not a companion to it. Forcing it off here
        # (rather than in each subclass) is also what lets LinearPredictor.forward,
        # which these classes inherit unchanged, apply _cov_features unconditionally.
        kwargs["quad_cov"] = False
        super().__init__(**kwargs)

    def _build_module(self, num_in, out_dim):
        raise NotImplementedError

    @staticmethod
    def _zero_and_set_bias(module, bias_value):
        """Warm-start `module` to output the constant `bias_value` regardless
        of input -- the nonlinear-head analogue of a plain nn.Linear's
        zero-weight/bias warm start -- by zeroing only its FINAL nn.Linear.

        Only the last layer is zeroed. Zeroing every nn.Linear also gives a
        constant output, but for a head whose first stage is itself a bare
        nn.Linear (MLPCovPredictor's Linear -> ReLU -> Linear) it makes the
        hidden activations the input-independent constant relu(b1), so
        d(loss)/d(first layer weight) is IDENTICALLY ZERO until the second
        layer leaves zero. Such a head cannot depart from a constant warm
        start in any reasonable number of pretraining epochs -- measured
        directly: zero feature variation and zero first-stage gradient at
        init, and a fitted b(x) flatter across outputs than a plain linear
        head's. Leaving the earlier layers at their random init keeps the
        features input-dependent, so gradients flow on the first step, while
        the zeroed final layer still makes the head's OUTPUT exactly the
        constant bias_value.
        """
        if isinstance(module, torch.nn.Linear):
            module.weight.data.fill_(0.0)
            module.bias.data = bias_value
        else:
            linears = [layer for layer in module if isinstance(layer, torch.nn.Linear)]
            if not linears:
                raise ValueError(
                    "A nonlinear head must contain at least one nn.Linear layer "
                    "to carry the constant warm start."
                )
            linears[-1].weight.data.fill_(0.0)
            module[-1].bias.data = bias_value

    def customize(self, a_tch, b_tch, init_bias, init_weight, random_init):
        """Warm-start both heads to output the initial (context-independent)
        mean/shape constants a_tch/b_tch, mirroring LinearPredictor's
        zero-weight/constant-bias warm start but for arbitrary nn.Module
        heads instead of assuming nn.Linear."""
        if random_init:
            return
        if init_weight is not None or init_bias is not None:
            raise NotImplementedError(
                "init_weight/init_bias are not supported for nonlinear "
                "linear_mean heads -- there is no single weight matrix to "
                "overwrite the way there is for a plain nn.Linear."
            )
        with torch.no_grad():
            if self.cov_rank:
                low_rank_np = self._low_rank_factor(a_tch.detach().numpy())
                torch_a = torch.tensor(low_rank_np, dtype=torch.double).flatten()
            else:
                torch_a = a_tch.flatten()
        self._zero_and_set_bias(self.linear_mean, b_tch)
        self._zero_and_set_bias(self.linear_cov, torch_a)

    def pretrain_mlp(self, module, input_tensors, target):
        """Fit `module` (linear_mean or linear_cov) to `target` via a short
        Adam/MSE loop -- the nonlinear-head analogue of gen_weights()'s
        closed-form lstsq fit, which has no equivalent for a nonlinear map."""
        N = input_tensors.shape[0]
        target_flat = target.reshape(N, -1)
        lr = self.lr if self.lr else 0.01
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        for _ in range(self.epochs):
            pred = module(input_tensors).view(N, -1)
            loss = criterion(pred, target_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _fit_cov_module(self, trainer, input_tensors, output_tensors):
        """Nonlinear heads have no closed form, so the covariance head is fit by
        pretrain_mlp's Adam/MSE loop instead of lstsq."""
        self.pretrain_mlp(self.linear_cov, input_tensors, output_tensors)

    def initialize(self, a_tch, b_tch, trainer):
        num_in, a_out, b_out = trainer.initialize_predictor_dims()
        a_shape_full = trainer.unc_set._a.shape
        a_out_eff = a_shape_full[0] * self.cov_rank if self.cov_rank else a_out
        self.linear_mean = self._build_module(num_in, b_out)
        self.linear_cov = self._build_module(num_in, a_out_eff)
        self.customize(a_tch, b_tch, trainer.settings.init_bias, trainer.settings.init_weight,
                        trainer.settings.random_init)
        if self.knn_cov or self.predict_cov or self.knn_mean:
            self.knn_fit(trainer)
        if self.predict:
            input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
            self.pretrain_mlp(self.linear_mean, input_tensors, trainer.u_train_tch)
        # After the linear_mean fit above (unlike LinearPredictor.initialize,
        # knn_fit here runs *before* the mean is fit), so the residuals
        # knn_predict takes covariances of use the fitted mean.
        self.fit_knn_residuals()
        if self.predict_cov:
            self._fit_cov_head(trainer)
        self.train()
        if self.pretrain:
            self.pretrain_func(trainer)
            # pretrain_func's SGD genuinely moves a nonlinear mean head (unlike
            # LinearPredictor's, where lstsq already sat at the optimum), so the
            # residual-based shape matrix has to be rebound to it.
            self._rebind_cov_to_pretrained_mean(trainer)
        if self.freeze_cov:
            for p in self.linear_cov.parameters():
                p.requires_grad_(False)
