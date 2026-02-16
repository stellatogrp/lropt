import unittest

import cvxpy as cp
import numpy as np
import numpy.random as npr
import numpy.testing as npt

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.train.predictors.linear import LinearPredictor
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal

ATOL = 1e-5
RTOL = 1e-5
TOLERANCE_DEFAULT = 1e-5


class TestEvaluateLearned(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.n = 4
        self.N = 20
        norms = npr.multivariate_normal(np.zeros(self.n), np.eye(self.n), self.N)
        self.data = np.exp(norms)

    def test_evaluate_and_violation_after_learning(self):
        # Data and problem setup
        n = self.n
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), self.N)

        # Context parameter (not strictly necessary for this test but mirrors learning tests)
        y = ContextParameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        x = cp.Variable(n)
        a = np.ones(n)
        c = 5.0

        # Objective and constraints
        objective = cp.Maximize(a @ x)
        constraints = [x @ (u + y) <= c, cp.norm(x) <= 2 * c]

        # Make the evaluation expression depend on the uncertain parameter
        eval_exp = -u @ x

        prob = RobustProblem(objective, constraints, eval_exp=eval_exp)

        # Train the uncertainty-set briefly (fast smoke test)
        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.lr = 0.001
        settings.num_iter = 1
        settings.optimizer = "SGD"
        settings.parallel = False
        trainer.train(settings=settings)

        # Solve the robust problem to get the decision
        prob.solve()
        x_robust = x.value

        # Create out-of-sample evaluation data and assign
        u_data = np.array([
            [0.1, 0.2, 0.3, 0.1],
            [0.1, 0.5, -0.3, 0.1],
            [0.4, 0.2, 0.3, -0.1],
        ])
        # create a fresh evaluation context dataset of matching size
        y_eval = npr.multivariate_normal(np.zeros(n), np.eye(n), u_data.shape[0])
        u.eval_data = u_data
        y.eval_data = y_eval

        # For evaluate_sol: expected is mean of -u_i @ x_robust
        expected_eval = np.mean([-u_data[i] @ x_robust for i in range(u_data.shape[0])])
        eval_value = np.mean(prob.evaluate_sol())
        npt.assert_allclose(eval_value, expected_eval, rtol=RTOL, atol=ATOL)

        # For violation_indicator_sol: only the first constraint contains uncertainty
        # Compute actual per-sample violation for the uncertain constraint
        y.value = np.zeros(n)
        first_violations = np.array([
            (x_robust @ (u_data[i] + y.value) - c) >= TOLERANCE_DEFAULT for
              i in range(u_data.shape[0])
        ], dtype=float)
        expected_viol_prob = np.mean(first_violations)

        # violation_indicator_sol returns a (num_g_total, batch) array; take mean over batch
        eval_viols = np.mean(prob.violation_indicator_sol(), axis=1)
        # There may be only one uncertain constraint; compare first entry
        npt.assert_allclose(eval_viols[0], expected_viol_prob, rtol=RTOL, atol=ATOL)

    def test_contextual_linear_predictor(self):
        """Train using a linear contextual predictor and verify evaluation paths
        use per-context re-solve to compute expected values and compare to
        `evaluate_sol()` and `violation_indicator_sol()` which should invoke the
        predictor to update set parameters per-context.
        """
        n = self.n
        y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), self.N)
        # small dataset for uncertainty

        y = ContextParameter(n, data=y_data)
        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

        x = cp.Variable(n)
        objective = cp.Maximize(np.ones(n) @ x)
        c = 5.0
        constraints = [x @ (u + y) <= c, cp.norm(x) <= 2 * c]
        eval_exp = -u @ x
        prob = RobustProblem(objective, constraints, eval_exp=eval_exp)

        trainer = Trainer(prob)
        settings = TrainerSettings()
        settings.contextual = True
        settings.predictor = LinearPredictor(predict_mean=True,knn_cov = True)
        settings.num_iter = 3
        settings.parallel = False
        trainer.train(settings=settings)

        # Manual per-context solve to compute expected evaluation and violations
        u_data = np.array([
            [0.1, 0.2, 0.3, 0.1],
            [0.1, 0.5, -0.3, 0.1],
            [0.4, 0.2, 0.3, -0.1],
        ])

        # fresh evaluation contexts separate from the training contexts
        y_eval = npr.multivariate_normal(np.zeros(n), np.eye(n), u_data.shape[0])

        manual_evals = []
        manual_violations = []
        for i in range(u_data.shape[0]):
            # use the fresh evaluation contexts when recomputing per-context solves
            y.value = y_eval[i]
            prob.solve()
            x_val = x.value
            manual_evals.append(-u_data[i] @ x_val)
            # single uncertain constraint is first one
            manual_violations.append(((x_val @ (u_data[i] + y.value) - c)
                                       >= TOLERANCE_DEFAULT).astype(float))

        expected_eval = np.mean(manual_evals)
        expected_viol = np.mean(manual_violations)

        # assign fresh evaluation data for batched evaluation which should use the predictor
        u.eval_data = u_data
        y.eval_data = y_eval

        eval_value = prob.evaluate()
        npt.assert_allclose(np.mean(eval_value), expected_eval, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(eval_value, manual_evals, rtol=RTOL, atol=ATOL)

        eval_viols =prob.violation_indicator()
        npt.assert_allclose( np.mean(eval_viols, axis=1)[0], expected_viol, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(eval_viols[0], manual_violations, rtol=RTOL, atol=ATOL)
