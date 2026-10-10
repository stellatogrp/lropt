import numpy as np
import scipy as sc
import torch
from sklearn.neighbors import NearestNeighbors

import cvxro.train.settings as settings

torch.set_default_dtype(settings.DTYPE)

class LinearPredictor(torch.nn.Module):

    def __init__(self,predict_mean = False, predict_cov= False,
                 pretrain = False,epochs = 100,lr = 0,
                 knn_cov = False, knn_mean = False,
                 n_neighbors = 10, knn_scale = 1, knn_residual_cov = False,
                 knn_anchor_residual = False,
                 quad_cov = False,
                 freeze_cov = False, freeze_mean = False,
                 cov_ridge = 0.0, cov_lr_mult = 1.0,
                 cov_rank = None, cov_rank_floor = 1e-2,
                 knn_cov_ridge = 0.0):
        super(LinearPredictor, self).__init__()
        self.predict = predict_mean
        self.pretrain = pretrain
        self.predict_cov = predict_cov
        self.epochs = epochs
        self.lr = lr
        self.n_neighbors = n_neighbors
        self.knn_cov = knn_cov
        self.knn_mean = knn_mean
        self.u_train_vals = None
        self.knn_scale = knn_scale
        # knn_residual_cov: build the local kNN covariance from the center
        # model's residuals u_j - mu_hat(x_j) instead of the raw neighbor
        # values u_j. 
        self.knn_residual_cov = knn_residual_cov
        # knn_anchor_residual: replace the CONSTANT term of the knn_scale blend
        #   A(x) = (1-knn_scale)*A_global + knn_scale*A_knn(x)
        # with sqrtm(cov(residuals)) instead of whatever init_A was passed in.
        # Messoudi et al. (2022) Eq. (4) specifies BOTH terms as covariances of the
        # observed errors; init_A is conventionally the MARGINAL covariance of u,
        # which is far larger whenever the mean head explains most of u's variance,
        # so the blend silently inflates A. Only meaningful together with
        # knn_residual_cov (which supplies the residuals) and knn_scale < 1.
        # Defaults to False -- runs are unchanged unless it is passed.
        self.knn_anchor_residual = knn_anchor_residual
        # knn_cov_ridge: add knn_cov_ridge * (tr C / n) * I to each local kNN covariance C
        # before its square root -- a ridge relative to the mean variance. A purely local
        # covariance from k <= n neighbours has rank <= k-1 < n, so its set is flat in the
        # remaining directions (and sqrtm returns a complex matrix). 0 = off, unchanged.
        self.knn_cov_ridge = float(knn_cov_ridge)
        # Residuals and the context matrix knn_fit indexed with, cached by
        # fit_knn_residuals() once linear_mean has been fit.
        self.resid_train_vals = None
        self.knn_x_train = None
        # quad_cov: feed linear_cov degree-2 polynomial features of the raw
        # context instead of the raw context itself
        self.quad_cov = quad_cov
        # freeze_cov: after the initial lstsq/kNN-target fit in initialize(),
        # hold linear_cov's parameters fixed for the rest of training
        self.freeze_cov = freeze_cov
        # freeze_mean: the mirror image -- hold linear_mean fixed after its initial fit,
        # so the AL loop trains the SHAPE alone. With freeze_cov it isolates which half
        # of the uncertainty set decision-focused training actually moves.
        self.freeze_mean = freeze_mean
        # cov_ridge: L2 penalty for gen_weights_cov's lstsq fit. 
        self.cov_ridge = cov_ridge
        # cov_lr_mult: multiplier on the AL loop's base lr applied only to
        # linear_cov's parameters (see param_groups()). Lets covariance keep
        # training under the true objective (unlike freeze_cov) while being
        # less exposed to a noisy/weak gradient signal than linear_mean.
        self.cov_lr_mult = cov_lr_mult
        # cov_rank: instead of a full n x n shape matrix, linear_cov only
        # ever outputs an n x cov_rank factor L (zero-padded to n x n in
        # forward()), so A@A.T = L@L.T is exactly rank <= cov_rank. Cuts the
        # covariance predictor's output dimension from n^2 to n*cov_rank --
        # exact when the true covariance genuinely has low rank (e.g. a
        # k-factor model), and a much smaller parameter count regardless.
        self.cov_rank = cov_rank
        # cov_rank_floor: small idiosyncratic-variance floor added to A's
        # diagonal when cov_rank is set. Exact zero-padding makes A exactly
        # singular (rank <= cov_rank < n), but the downstream differentiable
        # evaluation needs A itself invertible, not just A@A.T -- the same
        # role D plays in a real factor covariance model Sigma = B@B.T + D.
        self.cov_rank_floor = cov_rank_floor

    def param_groups(self, base_lr, extra_params=None):
        """Per-submodule optimizer parameter groups: linear_cov gets its own
        (typically smaller) learning rate via cov_lr_mult, instead of sharing
        linear_mean's. Trainer.py uses this when present (duck-typed) instead
        of flattening every parameter into one group with a single lr."""
        groups = [{"params": list(self.linear_mean.parameters()), "lr": base_lr}]
        groups.append({"params": list(self.linear_cov.parameters()),
                       "lr": base_lr * self.cov_lr_mult})
        if extra_params is not None:
            groups.append({"params": list(extra_params), "lr": base_lr})
        return groups

    def initialize(self,a_tch,b_tch,trainer):
        """Initialize the parameters"""
        num_in, a_out,b_out = trainer.initialize_predictor_dims()
        a_shape_full = trainer.unc_set._a.shape
        a_out_eff = a_shape_full[0] * self.cov_rank if self.cov_rank else a_out
        self.linear_mean = torch.nn.Linear(num_in,b_out)
        self.linear_cov = torch.nn.Linear(self._cov_in_dim(num_in),a_out_eff)
        self.customize(a_tch,b_tch,trainer.settings.init_bias,trainer.settings.init_weight,trainer.settings.random_init)
        if self.predict:
            input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
            self.gen_weights(input_tensors,trainer.u_train_tch)
        if self.knn_cov or self.predict_cov or self.knn_mean:
            self.knn_fit(trainer)
        # After gen_weights above, so the residuals use the fitted mean.
        self.fit_knn_residuals()
        if self.predict_cov:
            self._fit_cov_head(trainer)
        self.train()
        if self.pretrain:
            self.pretrain_func(trainer)
            self._rebind_cov_to_pretrained_mean(trainer)
        self._set_residual_anchor()
        self._apply_freezes()

    def _apply_freezes(self):
        """Hold either head fixed after its initial fit. Called at the END of
        initialize(), so a frozen head keeps its FITTED parameters -- not the constant
        warm start it began from."""
        if self.freeze_cov:
            for prm in self.linear_cov.parameters():
                prm.requires_grad_(False)
        if self.freeze_mean:
            for prm in self.linear_mean.parameters():
                prm.requires_grad_(False)

    def _low_rank_factor(self, cov_sqrt):
        """Given a (n,n) or (N,n,n) 'square root of covariance' matrix batch
        (as produced by knn_predict, or the sqrtm(cov) used to init a_tch),
        return the top cov_rank eigenfactor L (shape (n,cov_rank) or
        (N,n,cov_rank)) such that L@L.T approximates the true covariance
        M@M.T -- the target linear_cov fits/warm-starts against when it only
        ever outputs n*cov_rank values instead of n*n."""
        r = self.cov_rank
        single = (cov_sqrt.ndim == 2)
        batch = cov_sqrt[None] if single else cov_sqrt
        N, n, _ = batch.shape
        L = np.zeros((N, n, r))
        for i in range(N):
            sigma = batch[i] @ batch[i].T
            w, v = np.linalg.eigh(sigma)
            idx = np.argsort(w)[::-1][:r]
            w_r = np.clip(w[idx], 0, None)
            L[i] = v[:, idx] * np.sqrt(w_r)[None, :]
        return L[0] if single else L

    def _cov_in_dim(self, num_in):
        """Input dimension linear_cov expects, given quad_cov's feature map."""
        if not self.quad_cov:
            return num_in
        return num_in + num_in * (num_in + 1) // 2

    def _cov_features(self, x):
        """Degree-2 polynomial features of the raw context (torch), used to
        feed linear_cov when quad_cov is enabled. Any target that is an exact
        quadratic form in the raw context (e.g. Sigma(F) = c*F@F.T) lies in
        the linear span of these features, without the model ever being
        handed the target itself (unlike knn_cov)."""
        if not self.quad_cov:
            return x
        d = x.shape[1]
        iu, ju = torch.triu_indices(d, d)
        quad = x[:, iu] * x[:, ju]
        return torch.cat([x, quad], dim=1)

    def _cov_features_np(self, x):
        """Numpy counterpart of _cov_features, for the lstsq fit."""
        if not self.quad_cov:
            return x
        d = x.shape[1]
        iu, ju = np.triu_indices(d)
        quad = x[:, iu] * x[:, ju]
        return np.hstack([x, quad])

    def customize(self,a_tch,b_tch,init_bias,init_weight,random_init):
        """Set the weights of the predictor using mean-variance or given info
        Args:
        a_totsize:
            index where the bias begins
        a_tch, b_tch
            initialized a and b
        init_weight, init_bias
            initial weight and bias to use for the linear predictor
        random_init
            whether or not the predictor weights are initialized randomly
        """
        self.linear_mean.bias.data = b_tch
        if not random_init:
            with torch.no_grad():
                # torch_b = b_tch
                if self.cov_rank:
                    low_rank_np = self._low_rank_factor(a_tch.detach().numpy())
                    torch_a = torch.tensor(low_rank_np, dtype=torch.double).flatten()
                else:
                    torch_a = a_tch.flatten()
                # torch_concat = torch.hstack([torch_a, torch_b])
            self.linear_mean.weight.data.fill_(0.000)
            self.linear_cov.weight.data.fill_(0.000)
            self.linear_cov.bias.data = torch_a
            self.linear_mean.bias.data = b_tch
            if init_weight is not None:
                self.linear_mean.weight.data = init_weight.clone().detach().requires_grad_(True)
            if init_bias is not None:
                self.linear_mean.bias.data = torch.tensor(
                    init_bias, dtype=torch.double, requires_grad=True
                )

    def gen_weights(self,input,output):
        """Set the weights of the predictor using lstsq
        Args:
        input
            context data to train on
        output
            uncertain parameter data to train on
        """
        input = input.detach().numpy()
        output = output.detach().numpy()
        # num_in = input.shape[1]
        m = output.shape[1]
        N = input.shape[0]
        stacked_context = np.hstack([input,np.ones((N,1))])
        mults = [np.linalg.lstsq(stacked_context,output[:,0])[0]]
        for i in range(1,m):
            new_mults = np.linalg.lstsq(stacked_context,output[:,i])[0]
            mults.append(new_mults)
        mults_mean = np.vstack(mults)
        mults_mean_weight = mults_mean[:,:-1]
        mults_mean_bias = mults_mean[:,-1]
        # self.init_bias = np.hstack([a_tch.detach().numpy().flatten(),mults_mean_bias])
        # self.init_weight = np.vstack([np.zeros((m*m,num_in)),mults_mean_weight])
        self.linear_mean.weight.data = torch.tensor(
            mults_mean_weight, dtype=torch.double, requires_grad=True
        )
        self.linear_mean.bias.data = torch.tensor(
            mults_mean_bias, dtype=torch.double, requires_grad=True
        )

    def gen_weights_cov(self,input,output):
        """Set the weights of the predictor using lstsq
        Args:
        input
            context data to train on
        output
            uncertain parameter data to train on
        """
        N = input.shape[0]
        input = self._cov_features_np(input.detach().numpy())
        output = output.detach().numpy().reshape(N,-1)
        d = input.shape[1]
        stacked_context = np.hstack([input,np.ones((N,1))])
        if self.cov_ridge:
            # Ridge (L2-penalized) fit, solved for all output columns at
            # once. Needed once quad_cov's feature count (O(n_in^2)) starts
            # to approach the pretrain sample count, where the plain lstsq
            # below is ill-conditioned/overfit. The bias/intercept column is
            # left unpenalized, as is standard for ridge regression.
            reg = self.cov_ridge * np.eye(d + 1)
            reg[-1, -1] = 0.0
            mults_mean = np.linalg.solve(
                stacked_context.T @ stacked_context + reg,
                stacked_context.T @ output,
            ).T
        else:
            m = output.shape[1]
            mults = [np.linalg.lstsq(stacked_context,output[:,0])[0]]
            for i in range(1,m):
                new_mults = np.linalg.lstsq(stacked_context,output[:,i])[0]
                mults.append(new_mults)
            mults_mean = np.vstack(mults)
        mults_mean_weight = mults_mean[:,:-1]
        mults_mean_bias = mults_mean[:,-1]
        self.linear_cov.weight.data = torch.tensor(
            mults_mean_weight, dtype=torch.double, requires_grad=True
        )
        self.linear_cov.bias.data = torch.tensor(
            mults_mean_bias, dtype=torch.double, requires_grad=True)

    def forward(self, x,a_shape,b_shape,train_flag):
        """create a_tch and b_tch using the predictor"""
        cov_in = self._cov_features(x)
        out_a = self.linear_cov(cov_in)
        out_b = self.linear_mean(x)
        # raw_a = out[:, : a_shape[0] * a_shape[1]]
        # raw_b = out[:, a_shape[0] * a_shape[1] :]
        if self.cov_rank:
            low_rank = out_a.view(out_b.shape[0], a_shape[0], self.cov_rank)
            if a_shape[1] > self.cov_rank:
                pad = torch.zeros(out_b.shape[0], a_shape[0], a_shape[1] - self.cov_rank,
                                  dtype=low_rank.dtype)
                a_tch = torch.cat([low_rank, pad], dim=2)
            else:
                a_tch = low_rank
            if self.cov_rank_floor and a_shape[0] == a_shape[1]:
                eye = torch.eye(a_shape[0], dtype=a_tch.dtype).unsqueeze(0)
                a_tch = a_tch + (self.cov_rank_floor ** 0.5) * eye
        else:
            a_tch = out_a.view(out_b.shape[0], a_shape[0], a_shape[1])
        b_tch = out_b.view(out_b.shape[0], b_shape[0])
        if self.knn_cov:
            new_a_tch = self.knn_predict(x)
            a_tch = (1-self.knn_scale)*a_tch + self.knn_scale*new_a_tch
        if self.knn_mean:
            new_b_tch = self.knn_predict_mean(x)
            b_tch = (1-self.knn_scale)*b_tch + self.knn_scale*new_b_tch
        if not train_flag:
            a_tch = a_tch.detach().clone()
            b_tch = b_tch.detach().clone()
        return a_tch, b_tch, 1

    def knn_fit(self,trainer):
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        x = trainer.create_input_tensors(trainer.x_train_tch)
        knn.fit(x.detach())
        self.knn = knn
        self.u_train_vals = trainer.u_train_set
        # Kept so fit_knn_residuals() can evaluate linear_mean on exactly
        # the rows u_train_vals/the kNN index are aligned to.
        self.knn_x_train = x.detach().clone()
        self.resid_train_vals = None

    def _knn_mean_train_loo(self):
        """Leave-one-out kNN mean prediction mu_hat(x_j) for every training
        row, i.e. what knn_predict_mean() would return for x_j if x_j were
        not in the index. Returns (means, k_eff).

        Self-exclusion matters: querying the index with a row it was fit on
        returns that row as its own nearest neighbor, so a naive mean would
        contain u_j itself and the residual u_j - mu_hat(x_j) would be
        shrunk toward zero by construction. k_eff is the number of neighbors
        actually averaged, which _train_center() uses to undo the residual's
        remaining variance inflation."""
        u_vals = np.asarray(self.u_train_vals)
        n_rows = u_vals.shape[0]
        n_query = min(self.n_neighbors + 1, n_rows)
        neighbors = self.knn.kneighbors(
            self.knn_x_train, n_neighbors=n_query, return_distance=False)
        is_self = neighbors == np.arange(n_rows)[:, None]
        # Drop x_j's own row; if it is absent (duplicate contexts can push it
        # out of the returned set) drop the farthest neighbor instead, so
        # every row averages the same k_eff values.
        drop = np.where(is_self.any(axis=1), is_self.argmax(axis=1), n_query - 1)
        keep = np.ones(neighbors.shape, dtype=bool)
        keep[np.arange(n_rows), drop] = False
        loo_neighbors = neighbors[keep].reshape(n_rows, n_query - 1)
        return u_vals[loo_neighbors].mean(axis=1), loo_neighbors.shape[1]

    def _train_center(self):
        """The center forward() would place on the training rows -- the same
        (1-knn_scale)*linear_mean + knn_scale*kNN-mean blend, evaluated at
        x_j, except that the kNN branch is leave-one-out. Residuals are taken
        against this so the shape matrix describes spread about the center
        the set actually uses."""
        u_vals = np.asarray(self.u_train_vals)
        with torch.no_grad():
            center = self.linear_mean(self.knn_x_train).numpy().reshape(u_vals.shape)
        resid_scale = 1.0
        if self.knn_mean:
            knn_center, k_eff = self._knn_mean_train_loo()
            center = (1 - self.knn_scale) * center + self.knn_scale * knn_center
            # The LOO mean is itself an average of k_eff noisy draws, so under
            # homoscedastic noise Var(u_j - center_j) = sigma^2 (1 + s^2/k_eff)
            # rather than sigma^2. Undo that so A estimates the noise scale
            # itself. This is a single scalar shared by every x, so the rho
            # grid search would absorb it anyway -- it is here to keep A
            # interpretable on its own, and to keep the correction from
            # drifting with n_neighbors.
            resid_scale = 1.0 / np.sqrt(1.0 + (self.knn_scale ** 2) / k_eff)
        return center, resid_scale

    def _fit_cov_head(self, trainer):
        """Fit linear_cov against the local kNN covariance targets. Split out of
        initialize() so it can be re-run once the mean has moved."""
        input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
        output_tensors = self.knn_predict(input_tensors)
        if self.cov_rank:
            low_rank_np = self._low_rank_factor(output_tensors.detach().numpy())
            output_tensors = torch.tensor(low_rank_np, dtype=torch.double)
        self._fit_cov_module(trainer, input_tensors, output_tensors)

    def _fit_cov_module(self, trainer, input_tensors, output_tensors):
        """How this predictor fits its covariance head: closed-form lstsq here,
        overridden with a gradient loop by the nonlinear heads."""
        self.gen_weights_cov(input_tensors, output_tensors)

    def _rebind_cov_to_pretrained_mean(self, trainer):
        """Re-derive the residuals -- and the shape matrix fit from them -- against
        the mean pretrain_func just produced.

        The residuals are u_j - mu_hat(x_j), so anything that moves mu_hat
        invalidates them, and pretrain_func runs AFTER the initial fit. Only
        meaningful when knn_residual_cov is set: with raw-neighbor covariances the
        targets do not depend on the mean at all, and re-running the fit would be
        wasted work (and, for a gradient-fit head, would perturb it)."""
        if not self.knn_residual_cov:
            return
        self.fit_knn_residuals()
        if self.predict_cov:
            self._fit_cov_head(trainer)

    def _set_residual_anchor(self):
        """Re-point the blend's constant term at the residual covariance.

        `customize` parked init_A in linear_cov.bias with a zeroed weight, so for a
        knn_cov arm that never fits its covariance head (predict_cov=False) the bias IS
        the A_global of the knn_scale blend. This swaps it for sqrtm(cov(residuals)) --
        the same objects the LOCAL term is built from -- so lambda is a pure locality
        dial rather than also a scale change. A no-op unless knn_anchor_residual is set
        and the residuals were actually computed.
        """
        if not self.knn_anchor_residual or self.resid_train_vals is None:
            return
        # ONLY meaningful as the constant term of the knn_scale blend. Two guards, and
        # both matter:
        #   predict_cov -- _fit_cov_head has already fitted linear_cov (weight AND bias)
        #     against its targets, and this runs after it, so writing the bias here would
        #     CLOBBER that fit and silently break the arm.
        #   knn_cov / knn_scale -- with no kNN branch, or knn_scale == 1, the constant
        #     term carries zero weight in forward(), so there is nothing to correct.
        if self.predict_cov or not self.knn_cov or self.knn_scale >= 1:
            return
        resid = np.asarray(self.resid_train_vals, float)
        n = resid.shape[1]
        anchor = sc.linalg.sqrtm(np.cov(resid.T) + 1e-5 * np.eye(n)).real
        # linear_cov is an nn.Linear here but an nn.Sequential on the MLP heads, whose
        # warm start (NonlinearHeadPredictor._zero_and_set_bias) parks init_A in the FINAL
        # Linear's bias with a zeroed weight -- so that bias is the constant term there.
        final = self.linear_cov
        if not isinstance(final, torch.nn.Linear):
            final = [m for m in self.linear_cov.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            if self.cov_rank:
                anchor = self._low_rank_factor(anchor)
            final.bias.data = torch.tensor(
                anchor.flatten(), dtype=final.bias.dtype)

    def fit_knn_residuals(self):
        """Cache the training residuals u_j - mu_hat(x_j) that knn_predict()
        takes local covariances of when knn_residual_cov is set. Call after
        the center has been fit (gen_weights/pretrain_mlp, knn_fit), so the
        residuals reflect the fitted center rather than the constant it was
        initialized to. A no-op unless knn_residual_cov is set."""
        if not self.knn_residual_cov or self.knn_x_train is None:
            return
        center, resid_scale = self._train_center()
        self.resid_train_vals = (np.asarray(self.u_train_vals) - center) * resid_scale

    def _knn_cov_vals(self):
        """The array knn_predict() takes local covariances of: residuals if
        knn_residual_cov is set, else the raw uncertain-parameter values."""
        if not self.knn_residual_cov:
            return self.u_train_vals
        if self.resid_train_vals is None:
            # Predictor built outside initialize()'s ordering -- bind the
            # residuals to whatever linear_mean currently is.
            self.fit_knn_residuals()
        return self.resid_train_vals

    def knn_predict(self,x):
        cov_vals = self._knn_cov_vals()
        neighbors = self.knn.kneighbors(x.detach(), return_distance=False)
        atchs = []
        for i in range(x.shape[0]):
            cov = np.cov(cov_vals[neighbors[i]].T)
            if self.knn_cov_ridge:
                cov = cov + self.knn_cov_ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
            atchs.append(sc.linalg.sqrtm(cov))
        atchs = np.stack(atchs)
        a_tch = torch.tensor(atchs, dtype=torch.double, requires_grad=True)
        return a_tch

    def knn_predict_mean(self,x):
        neighbors = self.knn.kneighbors(x.detach(), return_distance=False)
        btchs = []
        for i in range(x.shape[0]):
            btchs.append(np.mean(self.u_train_vals[neighbors[i]], axis=0))
        btchs = np.stack(btchs)
        b_tch = torch.tensor(btchs, dtype=torch.double, requires_grad=True)
        return b_tch

    def pretrain_func(self,trainer):
        # call it pre-training
        if self.pretrain:
            assert (len(trainer.x_train_tch) != 0) and (len(trainer.u_train_tch) != 0)
            pred_optimizer = torch.optim.SGD(
                self.linear_mean.parameters(),
                lr = self.lr)
            criterion = torch.nn.MSELoss()
            epochs=self.epochs
            for epoch in range(epochs):
                # project=False: the regression target is the head's RAW output. Clamping
                # it to the support here would zero the gradient at exactly the points
                # whose prediction is out of support, which are the ones this loop exists
                # to pull back. The projection belongs where the SET is built.
                _,yhat,_=trainer.create_predictor_tensors(trainer.x_train_tch,
                                                          project=False)
                loss=criterion(yhat,trainer.u_train_tch)
                pred_optimizer.zero_grad()
                loss.backward()
                pred_optimizer.step()
