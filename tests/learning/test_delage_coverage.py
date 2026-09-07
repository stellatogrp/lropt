import unittest
from unittest import mock

import cvxpy as cp
import numpy as np

from cvxro import Trainer, TrainerSettings
from cvxro.parameter import ContextParameter
from cvxro.robust_problem import RobustProblem
from cvxro.train.predictors.linear import LinearPredictor
from cvxro.train.simulator import DefaultSimulator
from cvxro.uncertain_parameter import UncertainParameter
from cvxro.uncertainty_sets.ellipsoidal import Ellipsoidal

SHIFT = 1000.0


class TestDelageCoverageContexts(unittest.TestCase):
    """dist_loss must see the contexts the uncertainty set was fit from.

    dist_loss regresses the in-set indicator of u_0 onto the contexts to target a
    conditional coverage of ~0.9, so its x has to be the contexts that produced
    a_tch/b_tch/radius. In loss_and_constraints the state is advanced by simulate()
    roughly thirty lines before a_tch is refit at the bottom of the loop, and the
    delage branch sits in between -- it used to read the already-advanced x_t and so
    paired each set with the state that followed it. Both tensors have the same shape
    and batch size, so nothing but this test can catch it.

    The default simulator returns the state unchanged, which is why the defect is
    invisible in ordinary runs; the patch below gives it a state that actually moves.
    """

    def test_dist_loss_gets_solve_time_contexts(self):
        np.random.seed(0)
        n, N = 3, 40
        context = np.random.normal(0.0, 1.0, (N, n))
        data = context + np.random.normal(0.0, 0.3, (N, n))

        u = UncertainParameter(n, uncertainty_set=Ellipsoidal(p=2, data=data))
        ctx = ContextParameter(n, data=context)
        x = cp.Variable(n)
        t = cp.Variable()
        prob = RobustProblem(cp.Minimize(t),
                             [-(u + ctx) @ x <= t, cp.sum(x) == 1, x >= 0],
                             eval_exp=-(u + ctx) @ x)

        settings = TrainerSettings()
        settings.data = data
        settings.num_iter = 1
        settings.contextual = True
        settings.initialize_predictor = True
        settings.predictor = LinearPredictor(predict_mean=True, pretrain=False,
                                             lr=0.001, epochs=5, knn_cov=True,
                                             n_neighbors=5)
        settings.constraint_cvar = False
        settings.delage_coverage = True
        settings.init_rho = 1.0
        settings.seed = 5
        settings.parallel = False
        settings.num_random_init = 1

        # A large shift makes the two candidate states unmistakable: the real contexts
        # are standard normal, so anything near SHIFT is the post-simulate state.
        seen = []
        real_dist_loss = Trainer.dist_loss

        def spy(self, cho, mean, radius, xx, yy):
            seen.append(xx.detach().clone())
            return real_dist_loss(self, cho, mean, radius, xx, yy)

        def shifted(self, x, u, **kwargs):
            return [xx + SHIFT for xx in x]

        with mock.patch.object(DefaultSimulator, "simulate", shifted), \
             mock.patch.object(Trainer, "dist_loss", spy):
            Trainer(prob).train(settings=settings)

        self.assertTrue(seen, "dist_loss was never reached")
        for captured in seen:
            self.assertLess(
                captured.abs().max().item(), SHIFT / 2,
                "dist_loss received the post-simulate state, not the contexts the "
                "uncertainty set was fit from",
            )


if __name__ == "__main__":
    unittest.main()
