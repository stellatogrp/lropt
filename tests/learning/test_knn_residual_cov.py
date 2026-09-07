"""Local kNN covariance over mean-model residuals (`knn_residual_cov`).

A contextual set is `{A(x)z + mu_hat(x)}`, so its shape matrix should describe
the spread of u about `mu_hat(x)`.

`knn_residual_cov=True` switches `knn_predict` onto the residuals
`u_j - mu_hat(x_j)`. It defaults to False, and these tests pin down that the
default is bit-identical to the historical raw-neighbor path.
"""

import numpy as np
import scipy as sc
import torch

from cvxro.train.predictors.linear import LinearPredictor

N, K, DIM, N_NEIGHBORS, QUERY = 400, 2, 3, 40, 5
NOISE_CHOL = np.array([[0.30, 0.10, 0.00],
                       [0.00, 0.25, 0.05],
                       [0.00, 0.00, 0.20]])


def _data(mean_scale):
    """Context-dependent mean plus homoscedastic noise. mean_scale=0 makes the
    conditional mean constant, so raw and residual covariances must coincide."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, K))
    u = mean_scale * (x @ rng.normal(size=(K, DIM))) + rng.normal(size=(N, DIM)) @ NOISE_CHOL
    return x, u


class _StubTrainer:
    """Only the two attributes knn_fit() reaches for."""

    def __init__(self, x, u):
        self.u_train_set = u
        self.x_train_tch = torch.tensor(x)

    def create_input_tensors(self, x_train_tch):
        return self.x_train_tch


def _fitted(x, u, knn_residual_cov, knn_mean=False):
    """A predictor with its center fit and the kNN index built, i.e.
    LinearPredictor.initialize()'s ordering without a full Trainer.

    knn_mean=False is the CMV shape (center = fitted OLS mean); knn_mean=True
    is the pure-kNN shape (center = local neighborhood mean, no mean model)."""
    predictor = LinearPredictor(predict_mean=not knn_mean, knn_cov=True,
                                knn_mean=knn_mean, n_neighbors=N_NEIGHBORS,
                                knn_residual_cov=knn_residual_cov)
    predictor.linear_mean = torch.nn.Linear(K, DIM).double()
    predictor.linear_cov = torch.nn.Linear(K, DIM * DIM).double()
    trainer = _StubTrainer(x, u)
    if not knn_mean:
        predictor.gen_weights(trainer.create_input_tensors(None), torch.tensor(u))
    predictor.knn_fit(trainer)
    predictor.fit_knn_residuals()
    return predictor


def _sqrtm_local_cov(vals, neighbors):
    return np.stack([sc.linalg.sqrtm(np.cov(vals[row].T)) for row in neighbors])


def _generalized_variance(a_batch):
    return np.mean([np.linalg.det(a @ a.T) for a in a_batch])


def test_default_matches_raw_neighbor_covariance():
    """Flag off reproduces the historical path exactly, not merely closely."""
    x, u = _data(mean_scale=2.0)
    predictor = _fitted(x, u, knn_residual_cov=False)
    query = torch.tensor(x[:QUERY])
    neighbors = predictor.knn.kneighbors(query.detach(), return_distance=False)
    expected = _sqrtm_local_cov(u, neighbors)
    assert np.array_equal(predictor.knn_predict(query).detach().numpy(), expected)


def test_residual_path_uses_mean_model_residuals():
    x, u = _data(mean_scale=2.0)
    predictor = _fitted(x, u, knn_residual_cov=True)
    query = torch.tensor(x[:QUERY])
    neighbors = predictor.knn.kneighbors(query.detach(), return_distance=False)
    weight = predictor.linear_mean.weight.detach().numpy()
    bias = predictor.linear_mean.bias.detach().numpy()
    expected = _sqrtm_local_cov(u - (x @ weight.T + bias), neighbors)
    assert np.allclose(predictor.knn_predict(query).detach().numpy(), expected)


def test_residual_path_removes_mean_drift_inflation():
    """With a strongly context-dependent mean, the raw-neighbor covariance is
    inflated by the mean drift; the residual one tracks the true noise."""
    x, u = _data(mean_scale=2.0)
    query = torch.tensor(x[:QUERY])
    raw = _fitted(x, u, False).knn_predict(query).detach().numpy()
    residual = _fitted(x, u, True).knn_predict(query).detach().numpy()
    truth = np.linalg.det(NOISE_CHOL.T @ NOISE_CHOL)
    assert _generalized_variance(raw) > 10 * _generalized_variance(residual)
    assert 0.5 * truth < _generalized_variance(residual) < 2 * truth


def test_constant_mean_makes_both_paths_agree():
    """The non-contextual case: with no true context dependence the residual
    is (up to the slope OLS spuriously fits on finite-sample noise) a global
    shift, which a covariance is invariant to -- so there is nothing for the
    residual path to remove. Agreement here is statistical, not exact: the
    spurious slope leaves a sub-percent gap, versus the >10x gap
    test_residual_path_removes_mean_drift_inflation sees with a real mean."""
    x, u = _data(mean_scale=0.0)
    query = torch.tensor(x[:QUERY])
    raw = _fitted(x, u, False).knn_predict(query).detach().numpy()
    residual = _fitted(x, u, True).knn_predict(query).detach().numpy()
    assert np.allclose(raw, residual, atol=0.01 * np.abs(raw).max())
    assert abs(_generalized_variance(raw) / _generalized_variance(residual) - 1) < 0.05


def test_residuals_bind_lazily_when_fit_hook_is_skipped():
    """_knn_cov_vals() must still produce residuals for a predictor assembled
    outside initialize()'s ordering (no explicit fit_knn_residuals call)."""
    x, u = _data(mean_scale=2.0)
    predictor = _fitted(x, u, knn_residual_cov=True)
    predictor.resid_train_vals = None
    query = torch.tensor(x[:QUERY])
    neighbors = predictor.knn.kneighbors(query.detach(), return_distance=False)
    weight = predictor.linear_mean.weight.detach().numpy()
    bias = predictor.linear_mean.bias.detach().numpy()
    expected = _sqrtm_local_cov(u - (x @ weight.T + bias), neighbors)
    assert np.allclose(predictor.knn_predict(query).detach().numpy(), expected)


def test_pure_knn_center_is_also_inflated_by_mean_drift():
    """The pure-kNN shape (knn_mean=True, no mean model) is affected too.

    It is tempting to think it is already consistent: np.cov subtracts the
    neighborhood mean, which is exactly the center knn_mean places. But that
    removes only the OFFSET of the local mean; the drift of mu(x) ACROSS the
    k neighbors survives, and it is that drift the law of total covariance
    charges to Cov(mu(x)|x in N). The residual path removes it by using each
    neighbor's own leave-one-out fitted value.
    """
    x, u = _data(mean_scale=2.0)
    query = torch.tensor(x[:QUERY])
    raw = _fitted(x, u, False, knn_mean=True).knn_predict(query).detach().numpy()
    residual = _fitted(x, u, True, knn_mean=True).knn_predict(query).detach().numpy()
    truth = np.linalg.det(NOISE_CHOL.T @ NOISE_CHOL)
    assert _generalized_variance(raw) > 5 * _generalized_variance(residual)
    assert _generalized_variance(raw) > 5 * truth          # raw is far off the truth
    assert 0.3 * truth < _generalized_variance(residual) < 3 * truth


def test_leave_one_out_excludes_each_row_from_its_own_center():
    """A self-inclusive kNN mean would contain u_j with weight 1/k and shrink
    the residual toward zero by construction, biasing A downward."""
    x, u = _data(mean_scale=2.0)
    predictor = _fitted(x, u, True, knn_mean=True)
    loo_means, k_eff = predictor._knn_mean_train_loo()
    assert k_eff == N_NEIGHBORS
    neighbors = predictor.knn.kneighbors(
        predictor.knn_x_train, n_neighbors=N_NEIGHBORS + 1, return_distance=False)
    for j in (0, 7, N - 1):
        used = [i for i in neighbors[j] if i != j][:N_NEIGHBORS]
        assert np.allclose(loo_means[j], u[used].mean(axis=0))


def test_knn_scale_blend_is_mirrored_in_the_residual_center():
    """With knn_scale < 1 forward() centers on a blend of the two branches, so
    the residuals must be taken against that same blend, not either branch."""
    x, u = _data(mean_scale=2.0)
    predictor = LinearPredictor(predict_mean=True, knn_cov=True, knn_mean=True,
                                n_neighbors=N_NEIGHBORS, knn_scale=0.4,
                                knn_residual_cov=True)
    predictor.linear_mean = torch.nn.Linear(K, DIM).double()
    predictor.linear_cov = torch.nn.Linear(K, DIM * DIM).double()
    trainer = _StubTrainer(x, u)
    predictor.gen_weights(trainer.create_input_tensors(None), torch.tensor(u))
    predictor.knn_fit(trainer)
    predictor.fit_knn_residuals()

    weight = predictor.linear_mean.weight.detach().numpy()
    bias = predictor.linear_mean.bias.detach().numpy()
    linear_center = x @ weight.T + bias
    knn_center, k_eff = predictor._knn_mean_train_loo()
    expected_center = 0.6 * linear_center + 0.4 * knn_center
    expected_scale = 1.0 / np.sqrt(1.0 + 0.4 ** 2 / k_eff)
    assert np.allclose(predictor.resid_train_vals,
                       (u - expected_center) * expected_scale)


def test_rebind_updates_residuals_when_the_mean_moves():
    """pretrain_func runs after the initial residual fit, so anything it does to
    the mean must be reflected back into the residuals (and the shape fit from
    them). _rebind_cov_to_pretrained_mean is what initialize() calls to do it."""
    x, u = _data(mean_scale=2.0)
    predictor = _fitted(x, u, knn_residual_cov=True)
    before = predictor.resid_train_vals.copy()

    with torch.no_grad():                      # stand in for pretrain_func moving it
        predictor.linear_mean.bias += 0.5
    predictor._rebind_cov_to_pretrained_mean(_StubTrainer(x, u))

    assert not np.allclose(before, predictor.resid_train_vals)
    assert np.allclose(predictor.resid_train_vals, before - 0.5)


def test_rebind_is_inert_when_the_residual_path_is_off():
    """With raw-neighbor covariances the targets do not depend on the mean, so
    rebinding must not refit anything -- for a gradient-fit head that would
    perturb an already-fit covariance."""
    x, u = _data(mean_scale=2.0)
    predictor = _fitted(x, u, knn_residual_cov=False)
    with torch.no_grad():
        predictor.linear_mean.bias += 0.5
    predictor._rebind_cov_to_pretrained_mean(_StubTrainer(x, u))
    assert predictor.resid_train_vals is None
    assert predictor._knn_cov_vals() is predictor.u_train_vals
