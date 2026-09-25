import unittest

import cvxpy as cp
import numpy as np
import torch

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.train.predictors.linear import LinearPredictor
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal
from cvxro.violation_checker.utils import CONSTRAINT_STATUS


class TestTrainMaxUnsolved(unittest.TestCase):
    """A training batch with a few unsolvable samples trains on the rest.

    dcopf (2026-09-18): 2-13% of contexts have no feasible robust dispatch, the layer
    returns a junk solution for them, and the whole 30-sample batch was classed
    INFEASIBLE, undone and retaken at 0.8x lr fifty times per step -- so the predictor
    never moved. `TrainerSettings.train_max_unsolved` lets a batch drop up to that
    fraction of unsolvable samples and proceed as FEASIBLE. Default 0.0 keeps the old
    behaviour.
    """

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.n = 3
        n_train, n_validate, n_test = 40, 10, 10
        total = n_train + n_validate + n_test
        self.context = np.random.normal(0.0, 1.0, (total, self.n))
        self.data = self.context + np.random.normal(0.0, 0.3, (total, self.n))
        self.indices = {
            "train": np.arange(n_train),
            "validate": np.arange(n_train, n_train + n_validate),
            "test": np.arange(n_train + n_validate, total),
        }

    def _trainer(self, train_max_unsolved):
        n = self.n
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(p=2, data=self.data))
        ctx = ContextParameter(n, data=self.context)
        x = cp.Variable(n)
        t = cp.Variable()
        prob = RobustProblem(
            cp.Minimize(t),
            [-(u + ctx) @ x <= t, cp.sum(x) == 1, x >= 0],
            eval_exp=-(u + ctx) @ x,
        )
        settings = TrainerSettings()
        settings.data = self.data
        settings.indices_dict = self.indices
        settings.num_iter = 0
        settings.contextual = True
        settings.initialize_predictor = True
        settings.predictor = LinearPredictor(
            predict_mean=True, pretrain=False, lr=0.001, epochs=10, knn_cov=True,
            n_neighbors=5,
        )
        settings.init_rho = 1.0
        settings.seed = 5
        settings.batch_percentage = 0.25   # 10 of 40
        settings.train_max_unsolved = train_max_unsolved
        trainer = Trainer(prob)
        result = trainer.train(settings=settings)
        return trainer, result

    def _loss_with_two_unsolvable(self, trainer, result):
        """Run one training-path loss_and_constraints with the checker forced to report
        samples 0 and 3 unsolvable (a real check on this toy problem never fails)."""
        checker = trainer.violation_checker
        calls = {}

        def fake_check(*args, per_sample=False, **kwargs):
            batch = kwargs["x_batch"][0].shape[0]
            calls["per_sample"] = per_sample
            calls["batch"] = batch
            feasible = np.ones(batch, dtype=bool)
            feasible[[0, 3]] = False
            checker.last_sample_feasible = feasible if per_sample else None
            return CONSTRAINT_STATUS.INFEASIBLE

        checker.check_constraints = fake_check
        trainer._validate_flag = False
        trainer._test_flag = False
        out = trainer.loss_and_constraints(
            rho_tch=torch.tensor(1.0, dtype=torch.double),
            alpha=torch.tensor(0.0, dtype=torch.double),
            a_tch=torch.tensor(result.A, dtype=torch.double),
            b_tch=torch.tensor(result.b, dtype=torch.double),
            seed=1, batch_size=trainer.settings.batch_size,
        )
        return out, calls

    def test_default_zero_keeps_step_infeasible(self):
        trainer, result = self._trainer(train_max_unsolved=0.0)
        out, calls = self._loss_with_two_unsolvable(trainer, result)
        self.assertIs(out[4], CONSTRAINT_STATUS.INFEASIBLE)
        self.assertFalse(calls["per_sample"], "training path must keep the early exit")
        self.assertEqual(trainer._train_n_dropped, 0)

    def test_within_budget_drops_samples_and_proceeds(self):
        trainer, result = self._trainer(train_max_unsolved=0.25)
        out, calls = self._loss_with_two_unsolvable(trainer, result)
        self.assertTrue(calls["per_sample"])
        self.assertIs(out[4], CONSTRAINT_STATUS.FEASIBLE)
        self.assertEqual(trainer._train_n_dropped, 2)
        self.assertEqual(trainer.settings.kwargs_simulator["batch_int"], calls["batch"] - 2)
        u_0 = out[7]
        self.assertEqual(u_0.shape[0], calls["batch"] - 2)
        self.assertTrue(torch.isfinite(out[0]).all())

    def test_over_budget_stays_infeasible(self):
        trainer, result = self._trainer(train_max_unsolved=0.1)   # 2/10 > 0.1
        out, _ = self._loss_with_two_unsolvable(trainer, result)
        self.assertIs(out[4], CONSTRAINT_STATUS.INFEASIBLE)
        self.assertEqual(trainer._train_n_dropped, 0)

    def test_drop_unsolved_samples_selects_batched_only(self):
        trainer, _ = self._trainer(train_max_unsolved=0.5)
        keep = torch.tensor([True, False, True, True])
        # one batched output per layer variable, as cvxpylayers returns them
        z_t = [torch.zeros((4,) + tuple(v.shape)) for v in trainer.cvxpylayer.variables]
        x_t = [torch.arange(12.0).reshape(4, self.n)]
        u_0 = torch.arange(12.0).reshape(4, self.n)
        a_tch = torch.zeros(4, self.n, self.n)
        b_tch = torch.zeros(4, self.n)
        z2, x2, u2, a2, b2, r2 = trainer._drop_unsolved_samples(
            keep, z_t, x_t, u_0, a_tch, b_tch, None)
        self.assertEqual([z.shape[0] for z in z2], [3] * len(z_t))
        self.assertEqual(x2[0].shape[0], 3)
        self.assertEqual(u2.shape[0], 3)
        self.assertEqual(a2.shape, (3, self.n, self.n))
        self.assertEqual(b2.shape, (3, self.n))
        self.assertIsNone(r2)
        self.assertTrue(torch.equal(u2[1], u_0[2]))


if __name__ == "__main__":
    unittest.main()
