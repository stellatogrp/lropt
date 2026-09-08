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

    def __init__(self, cov_fit_lr=None, cov_fit_epochs=None, **kwargs):
        # A nonlinear head consumes the raw context; quad_cov's polynomial feature map
        # is an alternative to nonlinearity, not a companion to it. Forcing it off here
        # (rather than in each subclass) is also what lets LinearPredictor.forward,
        # which these classes inherit unchanged, apply _cov_features unconditionally.
        kwargs["quad_cov"] = False
        super().__init__(**kwargs)
        # Separate fit hyperparameters for the COVARIANCE head. The two heads regress
        # onto very different targets -- linear_mean onto realized u, linear_cov onto
        # sqrtm of a kNN-local covariance -- and measurement says they want different
        # settings. On the dcopf instance (n=300, nonlinear DGP), sweeping lr x epochs
        # per head and taking the best of each:
        #
        #   target      best lr   plateaus by   R^2 (mlp, 2 hidden layers)
        #   mu(x)          0.05      ~500 ep    0.995 -> 0.998
        #   A(x)           0.01      ~2000 ep   0.882 (500) -> 0.952 (2000)
        #
        # A single shared value is therefore wrong in BOTH directions: the mean wastes
        # epochs, and the covariance head at the mean's rate collapses -- the same MLP
        # scores 0.045 on A(x) at lr=0.05 and 0.891 at lr=0.01, which reads as a broken
        # architecture rather than a mistuned one. Default None keeps the shared
        # lr/epochs, so existing callers are unchanged.
        self.cov_fit_lr = self.lr if cov_fit_lr is None else float(cov_fit_lr)
        self.cov_fit_epochs = (self.epochs if cov_fit_epochs is None
                               else int(cov_fit_epochs))

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

    def pretrain_mlp(self, module, input_tensors, target, lr=None, epochs=None):
        """Fit `module` (linear_mean or linear_cov) to `target` via a short
        Adam/MSE loop -- the nonlinear-head analogue of gen_weights()'s
        closed-form lstsq fit, which has no equivalent for a nonlinear map.

        `lr` / `epochs` override the shared defaults so the covariance head can be
        fit differently from the mean head -- see __init__."""
        N = input_tensors.shape[0]
        target_flat = target.reshape(N, -1)
        lr = (self.lr if lr is None else lr) or 0.01
        n_epochs = self.epochs if epochs is None else epochs
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        for _ in range(n_epochs):
            pred = module(input_tensors).view(N, -1)
            loss = criterion(pred, target_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _fit_cov_module(self, trainer, input_tensors, output_tensors):
        """Nonlinear heads have no closed form, so the covariance head is fit by
        pretrain_mlp's Adam/MSE loop instead of lstsq -- at its OWN lr/epochs."""
        self.pretrain_mlp(self.linear_cov, input_tensors, output_tensors,
                          lr=self.cov_fit_lr, epochs=self.cov_fit_epochs)

    def pretrain_func(self, trainer):
        """Fit the nonlinear mean head. THE single place it is fitted.

        Overrides LinearPredictor.pretrain_func, which exists for a different reason:
        there the mean is solved exactly by gen_weights' lstsq and pretrain_func's SGD
        loop is a provable no-op on top of it, so it is gated behind `pretrain`. A
        nonlinear head has no closed form, so this loop IS the fit rather than a
        refinement of one, and it therefore runs whenever `predict_mean` is set --
        unconditionally on `pretrain`, which for these heads no longer selects a second
        training pass. (Gating it on `pretrain` would leave every `pretrain=False`
        caller -- CMV, for one -- with a mean head frozen at its constant warm start,
        which is strictly worse than the linear fit it replaced.)

        Uses Adam via pretrain_mlp, matching how the covariance head is fit, rather
        than the base class's SGD.
        """
        assert (len(trainer.x_train_tch) != 0) and (len(trainer.u_train_tch) != 0)
        input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
        self.pretrain_mlp(self.linear_mean, input_tensors, trainer.u_train_tch)

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
        # The mean is fit ONCE, here, by pretrain_func -- and before the residuals and
        # the covariance head, so that neither has to be rebound afterwards. This
        # replaces an earlier arrangement in which initialize() called pretrain_mlp
        # directly and `pretrain` then ran pretrain_func as a SECOND, differently
        # optimized (SGD, not Adam) pass over the same head against the same MSE
        # objective, forcing _rebind_cov_to_pretrained_mean to redo the residuals and
        # refit the covariance head. See pretrain_func below.
        if self.predict:
            self.pretrain_func(trainer)
        # After the linear_mean fit above (unlike LinearPredictor.initialize,
        # knn_fit here runs *before* the mean is fit), so the residuals
        # knn_predict takes covariances of use the fitted mean.
        self.fit_knn_residuals()
        if self.predict_cov:
            self._fit_cov_head(trainer)
        self.train()
        if self.freeze_cov:
            for p in self.linear_cov.parameters():
                p.requires_grad_(False)
