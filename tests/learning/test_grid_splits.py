import unittest

import cvxpy as cp
import numpy as np

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal


class TestGridSplitProvenance(unittest.TestCase):
    """Trainer.grid()'s Validate_* columns must come from the validate split.

    Regression guard: update_train_stats carries the validate-split statistics in
    grid(), and generate_test_row surfaces them as the Validate_* columns. The
    worst-case slot was once passed the TEST quantile, which silently put test data in
    Validate_worst -- and any radius selected by minimising that column was then
    selected on the test set.
    """

    def setUp(self):
        np.random.seed(0)
        self.n = 4
        # train/validate are drawn tightly around 1.0; test is shifted and 5x wider, so
        # no correct implementation can report the same statistic for both splits.
        self.data = np.vstack([
            np.random.normal(1.0, 0.10, (400, self.n)),
            np.random.normal(1.0, 0.10, (100, self.n)),
            np.random.normal(3.0, 0.50, (100, self.n)),
        ])
        self.indices = {
            "train": np.arange(400),
            "validate": np.arange(400, 500),
            "test": np.arange(500, 600),
        }

    def _grid(self):
        n = self.n
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(p=2, data=self.data))
        # grid() requires at least one ContextParameter; a constant one behind a
        # vacuous bound keeps the problem itself non-contextual.
        ctx = ContextParameter(1, data=np.zeros((len(self.data), 1)))
        x = cp.Variable(n)
        t = cp.Variable()
        prob = RobustProblem(
            cp.Minimize(t),
            [-u @ x <= t, cp.sum(x) == 1, x >= 0, ctx >= -1e3],
            eval_exp=-u @ x,
        )
        settings = TrainerSettings()
        settings.data = self.data
        settings.indices_dict = self.indices
        settings.num_iter = 0
        settings.contextual = False
        settings.init_rho = 1.0
        settings.seed = 5
        return Trainer(prob).grid(rholst=np.array([0.5, 1.0, 2.0]), settings=settings).df

    def test_validate_columns_are_not_test_columns(self):
        df = self._grid()
        pairs = [("Validate_worst", "Test_worst"),
                 ("Validate_cvar", "Test_cvar"),
                 ("Validate_val", "Test_val")]
        for validate_col, test_col in pairs:
            validate = df[validate_col].to_numpy(dtype=float)
            test = df[test_col].to_numpy(dtype=float)
            self.assertFalse(
                np.allclose(validate, test, rtol=0, atol=0),
                f"{validate_col} is bit-identical to {test_col}; it is reporting the "
                f"test split rather than the validate split",
            )

    def test_validate_columns_agree_with_each_other(self):
        """The validate quantile, CVaR and mean must all describe the same split."""
        df = self._grid()
        worst = df["Validate_worst"].to_numpy(dtype=float)
        cvar = df["Validate_cvar"].to_numpy(dtype=float)
        mean = df["Validate_val"].to_numpy(dtype=float)
        test_worst = df["Test_worst"].to_numpy(dtype=float)
        # The validate split is an order of magnitude closer to its own CVaR/mean than
        # to anything computed on the (shifted, wider) test split.
        self.assertTrue(np.all(np.abs(worst - cvar) < np.abs(worst - test_worst)))
        self.assertTrue(np.all(np.abs(worst - mean) < np.abs(worst - test_worst)))


if __name__ == "__main__":
    unittest.main()
