import unittest

import cvxpy as cp
import numpy as np
import torch

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.train.predictors.linear import LinearPredictor
from cvxro.train.trainer import SplitTensors
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal
from cvxro.violation_checker.utils import CONSTRAINT_STATUS
from cvxro.violation_checker.violation_checker import ViolationChecker


class TestGridContextualFeasibility(unittest.TestCase):
    """grid()'s per-sample feasibility check must use each split's own A/b.

    Regression guard: in the contextual case gen_new_z expands A/b to the batch of
    the split it is called for, and the validate block rebuilds them for its own
    unique contexts. _grid_solved_mask was reading whichever tensor happened to be
    left in a_tch_init, so the TEST check ran against A/b indexed by the *unique
    validate* contexts -- an unrelated pool. With test larger than validate (the
    experiments all run 0.5 / 0.2) that raised IndexError out of
    select_batch_object; had it been smaller it would have silently paired each
    test sample with an unrelated context's ellipsoid.
    """

    def setUp(self):
        np.random.seed(0)
        self.n = 3
        # test is deliberately larger than validate, mirroring the experiment configs
        # and making a stale validate-sized A/b tensor too short to index.
        self.n_train, self.n_validate, self.n_test = 60, 20, 50
        total = self.n_train + self.n_validate + self.n_test
        self.context = np.random.normal(0.0, 1.0, (total, self.n))
        self.data = self.context + np.random.normal(0.0, 0.3, (total, self.n))
        self.indices = {
            "train": np.arange(self.n_train),
            "validate": np.arange(self.n_train, self.n_train + self.n_validate),
            "test": np.arange(self.n_train + self.n_validate, total),
        }

    def _grid(self):
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
        settings.track_unsolved = True
        settings.init_rho = 1.0
        settings.seed = 5
        trainer = Trainer(prob)
        # The experiments always fit the predictor with train() before handing the
        # settings to grid(); grid() alone never sets trainer.x_endind.
        result = trainer.train(settings=settings)
        settings.init_A, settings.init_b = result.A, result.b
        settings.predictor = result._predictor
        return trainer.grid(rholst=np.array([0.5, 1.0]), settings=settings).df

    def test_contextual_grid_with_track_unsolved_runs(self):
        """The whole point: this used to die with IndexError before reaching the df."""
        df = self._grid()
        for column in ("Frac_unsolved_test", "Frac_unsolved_validate",
                       "Frac_feasible_test", "Frac_feasible_validate"):
            values = df[column].to_numpy(dtype=float)
            self.assertFalse(np.isnan(values).any(), f"{column} was never populated")
            self.assertTrue(((values >= 0.0) & (values <= 1.0)).all(),
                            f"{column} is not a fraction: {values}")


class TestViolationCheckerBatchGuard(unittest.TestCase):
    """A batched argument that disagrees with the context batch must raise clearly.

    Without the guard a short tensor raises a bare IndexError from deep inside
    select_batch_object, and a long one is worse still: the loop completes and each
    sample is silently checked against another sample's row.
    """

    N_CONTEXT = 10
    DIM = 3

    def _call(self, shape_batch):
        """Run check_constraints with shape_torches batched to shape_batch.

        The single constraint is on the shape parameter itself, so the check needs no
        variables and exercises nothing but the batching.
        """
        ctx = ContextParameter(self.DIM, data=np.zeros((self.N_CONTEXT, self.DIM)))
        shape_param = cp.Parameter((self.DIM, self.DIM))
        checker = ViolationChecker(None, [shape_param >= -1e6])
        return checker.check_constraints(
            z_batch=[],
            rho_mult_parameter=[cp.Parameter(nonneg=True)],
            rho_tch=torch.tensor(1.0),
            cp_parameters=[],
            cp_param_tch=[],
            x_parameters=[ctx],
            x_batch=[torch.zeros(self.N_CONTEXT, self.DIM)],
            shape_parameters=[shape_param],
            shape_torches=[torch.zeros(shape_batch, self.DIM, self.DIM)],
        )

    def test_short_shape_torch_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self._call(shape_batch=self.N_CONTEXT - 6)
        self.assertIn("Batch size mismatch", str(caught.exception))

    def test_long_shape_torch_raises_value_error(self):
        """The silent-corruption direction: long enough to index, still wrong."""
        with self.assertRaises(ValueError) as caught:
            self._call(shape_batch=self.N_CONTEXT + 15)
        self.assertIn("Batch size mismatch", str(caught.exception))

    def test_matching_batch_is_accepted(self):
        self.assertEqual(self._call(shape_batch=self.N_CONTEXT),
                         CONSTRAINT_STATUS.FEASIBLE)


class TestSplitTensorsBinding(unittest.TestCase):
    """SplitTensors must refuse a pair of tensors that describe different splits.

    This is the structural half of the fix: grid() no longer holds loose A/b tensors
    that could be crossed, and the one place a split is assembled validates itself.
    """

    DIM = 3

    def _make(self, x_len, ab_len, contextual=True):
        return SplitTensors(
            x_batch=[torch.zeros(x_len, self.DIM)],
            z=[torch.zeros(x_len, self.DIM)],
            a_tch=torch.zeros(ab_len, self.DIM, self.DIM),
            b_tch=torch.zeros(ab_len, self.DIM),
            contextual=contextual,
        )

    def test_crossed_split_is_rejected(self):
        """A validate-sized A/b paired with a test-sized x: the original bug."""
        with self.assertRaises(ValueError) as caught:
            self._make(x_len=50, ab_len=20)
        self.assertIn("a_tch has batch dimension 20", str(caught.exception))

    def test_oversized_ab_is_rejected(self):
        """The direction that would have been silent rather than an IndexError."""
        with self.assertRaises(ValueError):
            self._make(x_len=20, ab_len=50)

    def test_matching_split_is_accepted(self):
        self.assertEqual(self._make(x_len=20, ab_len=20).a_tch.shape[0], 20)

    def test_non_contextual_ab_need_not_be_batched(self):
        """Without a context, A and b are shared across samples: no batch to agree on."""
        split = SplitTensors(
            x_batch=[torch.zeros(20, self.DIM)],
            z=[torch.zeros(20, self.DIM)],
            a_tch=torch.zeros(self.DIM, self.DIM),
            b_tch=torch.zeros(self.DIM),
            contextual=False,
        )
        self.assertEqual(split.a_tch.shape, (self.DIM, self.DIM))


if __name__ == "__main__":
    unittest.main()
