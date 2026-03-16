"""Tests for augmented Lagrangian improvement strategies.

Tests the two dual-update strategies (classic, pid)
and related settings.

All tests use minimal problem sizes (n=2, N=20) and few iterations
to keep memory low and runtime fast.
"""

import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
import scipy as sc

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal


def _make_portfolio_problem(n=2, N=20, seed=42):
    """Create a minimal portfolio problem with a chance constraint.

    Based on test_portfolio_intro pattern which is known to work.
    Returns (prob, trainer, data) so tests can configure settings and train.
    """
    np.random.seed(seed)
    sig = np.array([[0.5, -0.3], [-0.3, 0.4]])[:n, :n]
    mu_vec = np.random.uniform(0.2, 0.5, n)
    data = np.random.multivariate_normal(mu_vec, sig, N)

    # ContextParameter must be referenced in the problem
    dist = np.ones(n) * 3.0
    y_data = np.random.dirichlet(dist, N)
    y = ContextParameter(n, data=y_data)
    u = UncertainParameter(n, uncertainty_set=Ellipsoidal(p=2, data=data))

    x = cp.Variable(n)
    t_var = cp.Variable()
    # y is used in objective so it's discovered as a parameter
    objective = cp.Minimize(t_var + 0.2 * cp.norm(x - y, 1))
    constraints = [-x @ u <= t_var, cp.sum(x) == 1, x >= 0]
    eval_exp = -x @ u + 0.2 * cp.norm(x - y, 1)
    prob = RobustProblem(objective, constraints, eval_exp=eval_exp)

    trainer = Trainer(prob)
    return prob, trainer, data


def _base_settings(data, n=2, N=20):
    """Return TrainerSettings for a fast, minimal training run."""
    from sklearn.model_selection import train_test_split

    test_p = 0.1
    train, _ = train_test_split(
        data, test_size=max(int(N * test_p), 1), random_state=5
    )
    init_A = sc.linalg.sqrtm(np.cov(train.T) + 1e-4 * np.eye(n))
    init_b = np.mean(train, axis=0)

    settings = TrainerSettings()
    settings.lr = 0.0001
    settings.num_iter = 6
    settings.optimizer = "SGD"
    settings.momentum = 0.8
    settings.seed = 5
    settings.init_A = init_A
    settings.init_b = init_b
    settings.init_lam = 0.5
    settings.init_mu = 0.01
    settings.mu_multiplier = 1.001
    settings.init_alpha = 0.0
    settings.kappa = -0.001
    settings.test_percentage = test_p
    settings.validate_percentage = 0.01
    settings.parallel = False
    settings.random_init = False
    settings.num_random_init = 1
    settings.position = False
    settings.eta = 0.05
    settings.aug_lag_update_interval = 3  # trigger AL update twice in 6 iters
    settings.save_history = False
    settings.test_frequency = 100  # skip test eval
    settings.validate_frequency = 100  # skip validation eval
    return settings


class TestALSettings(unittest.TestCase):
    """Test that AL settings fields exist and are configurable."""

    def test_default_values(self):
        s = TrainerSettings()
        self.assertEqual(s.dual_update_strategy, "classic")
        self.assertAlmostEqual(s.pid_Kp, 5.0)
        self.assertAlmostEqual(s.pid_Ki, 1.0)
        self.assertAlmostEqual(s.pid_nu, 0.99)
        self.assertTrue(s.reset_prev_cost_on_al_update)

    def test_set_strategy(self):
        s = TrainerSettings()
        for strategy in ("classic", "pid"):
            s.dual_update_strategy = strategy
            self.assertEqual(s.dual_update_strategy, strategy)

    def test_slots_reject_unknown(self):
        s = TrainerSettings()
        with self.assertRaises(AttributeError):
            s.nonexistent_field = 1


class TestClassicStrategy(unittest.TestCase):
    """Test that classic strategy still works (backward compatibility)."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_classic_trains(self):
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "classic"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(len(result.df), 0)

    def test_classic_mu_is_scalar(self):
        """Classic strategy should keep mu as a scalar float."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "classic"
        result = self.trainer.train(settings=settings)
        mu_val = result.df["mu"].iloc[-1]
        self.assertIsInstance(mu_val, float)


class TestPIDStrategy(unittest.TestCase):
    """Test νPI controller (arXiv:2406.04558) dual-update strategy."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_pid_trains(self):
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pid"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(len(result.df), 0)

    def test_pid_lambda_nonneg(self):
        """PID strategy should keep lambda >= 0."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pid"
        result = self.trainer.train(settings=settings)
        for lam_arr in result.df["lam_list"]:
            self.assertTrue(np.all(lam_arr >= -1e-10))

    def test_pid_custom_gains(self):
        """Training should work with custom Ki, Kp, nu."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pid"
        settings.pid_Kp = 2.0
        settings.pid_Ki = 0.5
        settings.pid_nu = 0.9
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_pid_no_damping(self):
        """Kp=0 should give integral-only update (no damping)."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pid"
        settings.pid_Kp = 0.0
        settings.pid_Ki = 1.0
        settings.pid_nu = 0.0
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestResetPrevCost(unittest.TestCase):
    """Test the reset_prev_cost_on_al_update setting."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_no_reset_trains(self):
        """Training should complete with prev_fin_cost reset disabled."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.reset_prev_cost_on_al_update = False
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)

    def test_no_reset_with_pid(self):
        """PID + no reset should complete."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pid"
        settings.reset_prev_cost_on_al_update = False
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestStrategyCrossSettings(unittest.TestCase):
    """Test combinations of strategies with other settings."""

    def setUp(self):
        self.n = 2
        self.N = 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def test_all_strategies_same_seed_deterministic(self):
        """Each strategy should produce deterministic results with same seed."""
        for strategy in ("classic", "pid"):
            results = []
            for _ in range(2):
                _, trainer, data = _make_portfolio_problem(self.n, self.N)
                settings = _base_settings(data, self.n, self.N)
                settings.dual_update_strategy = strategy
                result = trainer.train(settings=settings)
                results.append(result.df["Lagrangian_val"].iloc[-1])
            npt.assert_allclose(
                results[0], results[1],
                err_msg=f"Strategy '{strategy}' not deterministic",
            )

    def test_pid_adam_optimizer(self):
        """PID strategy should work with Adam optimizer."""
        settings = _base_settings(self.data, self.n, self.N)
        settings.dual_update_strategy = "pid"
        settings.optimizer = "Adam"
        result = self.trainer.train(settings=settings)
        self.assertIsNotNone(result.df)


class TestRhoCalibration(unittest.TestCase):
    """Test post-training rho calibration."""

    def test_tune_rho_setting_defaults(self):
        """Verify new settings exist with correct defaults."""
        s = TrainerSettings()
        self.assertFalse(s.tune_rho)
        self.assertEqual(s.tune_rho_n_grid, 30)
        self.assertEqual(s.tune_rho_range, (0.01, 3.0))

    def test_tune_rho_disabled_no_change(self):
        """tune_rho=False should not alter training behavior."""
        _, trainer, data = _make_portfolio_problem()
        settings = _base_settings(data)
        settings.tune_rho = False
        result = trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(result.rho, 0)

    def test_tune_rho_enabled(self):
        """tune_rho=True should run and return a valid rho."""
        _, trainer, data = _make_portfolio_problem()
        settings = _base_settings(data)
        settings.tune_rho = True
        settings.tune_rho_n_grid = 5
        settings.tune_rho_range = (0.5, 2.0)
        result = trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(result.rho, 0)

    def test_tune_rho_with_pid(self):
        """Rho calibration should work with PID strategy."""
        _, trainer, data = _make_portfolio_problem()
        settings = _base_settings(data)
        settings.dual_update_strategy = "pid"
        settings.tune_rho = True
        settings.tune_rho_n_grid = 5
        settings.tune_rho_range = (0.5, 2.0)
        result = trainer.train(settings=settings)
        self.assertIsNotNone(result.df)
        self.assertGreater(result.rho, 0)

    def test_tune_rho_preserves_predictor(self):
        """Predictor should survive rho calibration."""
        _, trainer, data = _make_portfolio_problem()
        settings = _base_settings(data)
        settings.tune_rho = True
        settings.tune_rho_n_grid = 3
        settings.tune_rho_range = (0.8, 1.2)
        result = trainer.train(settings=settings)
        self.assertIsNotNone(result.predictor)


if __name__ == "__main__":
    unittest.main()
