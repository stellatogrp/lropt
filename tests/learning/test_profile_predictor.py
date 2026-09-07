import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
import torch

import cvxro


def _problem(data, ctx_data, n_samples, dim):
    u = cvxro.UncertainParameter(dim, uncertainty_set=cvxro.Ellipsoidal(p=2, data=data))
    ctx = cvxro.ContextParameter(ctx_data.shape[1], data=ctx_data)
    z = cp.Variable(dim, name="z")
    t = cp.Variable(name="t")
    return cvxro.RobustProblem(
        cp.Minimize(t),
        [cvxro.max_of_uncertain([-z @ u - t, cp.sum(z) - 1 - u[0]], epigraph_index=0) <= 0,
         z >= 0, z <= 3, ctx >= -1e3],
        eval_exp=-z @ u,
    )


def _fixture(n_samples=300, dim=3, seed=0):
    rng = np.random.RandomState(seed)
    ctx_data = rng.uniform(-1, 1, (n_samples, 2))
    data = np.column_stack([
        1.0 + 2.0 * ctx_data[:, 0] + rng.normal(0, 0.4, n_samples),
        2.0 + 1.0 * ctx_data[:, 1] + rng.normal(0, 0.4, n_samples),
        1.5 + rng.normal(0, 0.4, n_samples),
    ])
    third = n_samples // 3
    indices = {"train": np.arange(third),
               "validate": np.arange(third, 2 * third),
               "test": np.arange(2 * third, n_samples)}
    settings = cvxro.TrainerSettings()
    settings.data = data
    settings.indices_dict = indices
    settings.seed = 5
    settings.contextual = True
    settings.initialize_predictor = True
    settings.init_rho = 1.0
    settings.init_A = np.eye(dim)
    settings.init_b = np.zeros(dim)
    settings.num_iter = 0
    settings.target_eta = 0.1
    return _problem(data, ctx_data, n_samples, dim), settings, data, ctx_data


def _shape_at_train_contexts(trainer, result):
    with torch.no_grad():
        x = trainer.create_input_tensors(trainer.x_train_tch)
        a, _, _ = result._predictor.forward(
            x, trainer.unc_set._a.shape, trainer.unc_set._b.shape, False)
    return a.detach().numpy()


class TestProfilePredictor(unittest.TestCase):
    def test_starts_at_the_plug_in(self):
        """With w == 1 the shape must be the plug-in estimate, bit for bit. This is what
        makes 'training starts at the ideal mean-variance set' a real claim."""
        shapes = {}
        for name, predictor in (
            ("plug_in", cvxro.LinearPredictor(
                predict_mean=True, knn_cov=True, knn_scale=1.0,
                knn_residual_cov=True, n_neighbors=20, freeze_cov=True)),
            ("profile", cvxro.ProfilePredictor(predict_mean=True, n_neighbors=20)),
        ):
            problem, settings, _, _ = _fixture()
            settings.predictor = predictor
            trainer = cvxro.Trainer(problem)
            shapes[name] = _shape_at_train_contexts(trainer, trainer.train(settings=settings))
        npt.assert_array_equal(shapes["profile"], shapes["plug_in"])

    def test_profile_init_scales_the_shape_exactly(self):
        """profile_init = c must scale the plug-in shape by exactly c (Corollary 2 says
        a scalar is free after rho tuning, so this is the knob's calibration check)."""
        shapes = {}
        for c in (1.0, 2.5):
            problem, settings, _, _ = _fixture()
            settings.predictor = cvxro.ProfilePredictor(
                predict_mean=True, n_neighbors=20, profile_init=c)
            trainer = cvxro.Trainer(problem)
            shapes[c] = _shape_at_train_contexts(trainer, trainer.train(settings=settings))
        npt.assert_allclose(shapes[2.5], 2.5 * shapes[1.0], rtol=1e-9)

    def test_column_scaling_is_what_reweights_margins(self):
        """(A diag(w))^T p == diag(w) A^T p -- the identity the class relies on."""
        rng = np.random.RandomState(0)
        a = rng.normal(size=(4, 3, 3))
        w = np.abs(rng.normal(size=(4, 3))) + 0.1
        p = rng.normal(size=3)
        scaled = a * w[:, None, :]
        for i in range(4):
            npt.assert_allclose(scaled[i].T @ p, np.diag(w[i]) @ (a[i].T @ p), rtol=1e-12)

    def test_trainable_shape_dimension_is_the_profile(self):
        """The shape's trainable dimension must be n*p (contextual) or n (global), not
        the n^2*p of an affine shape map."""
        dim = 3
        for contextual, expected in ((True, None), (False, dim)):
            problem, settings, _, ctx = _fixture(dim=dim)
            settings.predictor = cvxro.ProfilePredictor(
                predict_mean=True, n_neighbors=20, profile_contextual=contextual)
            trainer = cvxro.Trainer(problem)
            result = trainer.train(settings=settings)
            profile = result._predictor.profile
            n_weight = profile.weight.numel()
            self.assertEqual(profile.bias.numel(), dim)
            if expected is None:
                self.assertEqual(n_weight, dim * trainer.create_input_tensors(
                    trainer.x_train_tch).shape[1])
            else:
                self.assertEqual(n_weight, dim)

    def test_linear_cov_is_frozen_by_default(self):
        """linear_cov contributes (1 - knn_scale) = 0, so it must not take gradients."""
        problem, settings, _, _ = _fixture()
        settings.predictor = cvxro.ProfilePredictor(predict_mean=True, n_neighbors=20)
        trainer = cvxro.Trainer(problem)
        result = trainer.train(settings=settings)
        self.assertFalse(any(p.requires_grad
                             for p in result._predictor.linear_cov.parameters()))
        self.assertTrue(all(p.requires_grad
                            for p in result._predictor.profile.parameters()))

    def test_training_moves_the_profile_off_one(self):
        """The point of the class: the profile must actually be learned, and stay
        positive (softplus), so the set never degenerates."""
        problem, settings, _, _ = _fixture()
        settings.predictor = cvxro.ProfilePredictor(predict_mean=True, n_neighbors=20)
        settings.num_iter = 15
        settings.lr = 0.01
        trainer = cvxro.Trainer(problem)
        result = trainer.train(settings=settings)
        with torch.no_grad():
            x = trainer.create_input_tensors(trainer.x_train_tch)
            w = result._predictor._profile_weights(x).detach().numpy()
        self.assertTrue(np.all(w > 0), "softplus must keep the profile positive")
        self.assertGreater(np.abs(w - 1.0).max(), 1e-6, "profile never moved off 1")


if __name__ == "__main__":
    unittest.main()
