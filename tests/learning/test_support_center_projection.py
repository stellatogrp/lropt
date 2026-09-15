"""A contextual set with a support must never be EMPTY.

`Ellipsoidal(..., lb=...)` intersects the ellipsoid `{A(x)z + b(x) : ||z|| <= rho}` with
the support `u >= lb`. If the predicted center `b(x)` lies outside that support, the
intersection can be empty -- and an empty uncertainty set makes the robust constraint
quantify over nothing, so it bounds nothing. The solver then returns an enormous decision
and reports `optimal`, and no violation metric catches it, because the decision really is
feasible for a constraint that says nothing.

Measured on the newsvendor with `lb=0` before the fix: 9% of contexts had a center with a
negative coordinate, up to 4% of sets were empty, and those contexts came back with
`|x| ~ 1e4` and costs three orders of magnitude off, against `-1.3` for the same arm with
no support.

`TrainerSettings.project_center_to_support` (default True) clamps the center into the box
support at `Trainer.create_predictor_tensors`, which is the single point both the training
and the evaluation paths take their A(x), b(x) from. With `b(x)` inside the support, `b(x)`
itself lies in both the ellipsoid and the support, so the set is nonempty by construction.

The first test also pins down the thing that was wrongly suspected first: the counterpart
ITSELF is exact with a support, including through `max_of_uncertain`.
"""

import unittest

import cvxpy as cp
import numpy as np
import torch

import cvxro

N, DIM = 60, 2
RHO = 2.0


def _worst_case(w, A, b, rho, lb=None):
    """max w'u over {u = Az+b, ||z|| <= rho} (∩ {u >= lb}), by direct solve."""
    z = cp.Variable(DIM)
    u = A @ z + b
    cons = [cp.norm(z, 2) <= rho] + ([u >= lb] if lb is not None else [])
    prob = cp.Problem(cp.Maximize(w @ u), cons)
    prob.solve(solver="CLARABEL")
    return float(prob.value)


def _counterpart_value(pieces, A, b, rho, lb=None):
    """Smallest t with `max_i w_i'u <= t` for all u in the set, via cvxro."""
    kwargs = {} if lb is None else {"lb": float(lb)}
    u = cvxro.UncertainParameter(
        DIM, uncertainty_set=cvxro.Ellipsoidal(rho=rho, a=A, b=b, **kwargs))
    t = cp.Variable()
    rows = [w @ u for w in pieces]
    cons = [rows[0] <= t] if len(rows) == 1 else [cvxro.max_of_uncertain(rows) <= t]
    cvxro.RobustProblem(cp.Minimize(t), cons).solve()
    return float(t.value)


class ConstantCenterPredictor(torch.nn.Module):
    """A predictor that returns a fixed A and b for every context.

    Deliberately not fitted: the point is to hand the trainer a center that is OUTSIDE
    the support and see what the set does, which a fitted head would only reach for some
    contexts and not others.
    """

    def __init__(self, a, b):
        super().__init__()
        self.a = torch.tensor(np.asarray(a, float), dtype=torch.double)
        self.b = torch.tensor(np.asarray(b, float), dtype=torch.double)
        self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.double))

    def forward(self, x, a_shape, b_shape, train_flag):
        n = x.shape[0]
        return (self.a.expand(n, *self.a.shape).clone(),
                self.b.expand(n, *self.b.shape).clone(), 1)

    def initialize(self, a_tch, b_tch, trainer):
        return

    def param_groups(self, base_lr, extra_params=None):
        return [{"params": [self.dummy], "lr": base_lr}]


def _newsvendor(lb, center, seed=0):
    """A two-product newsvendor whose set is centered at `center` for every context."""
    rng = np.random.default_rng(seed)
    data = np.maximum(0.0, rng.normal(3.0, 1.0, size=(N, DIM)))
    k_data = np.maximum(0.5, rng.normal(4.0, 1.0, size=(N, DIM)))
    p_data = k_data + np.abs(rng.normal(0.0, 1.0, size=(N, DIM)))

    kwargs = {} if lb is None else {"lb": float(lb)}
    u = cvxro.UncertainParameter(DIM, uncertainty_set=cvxro.Ellipsoidal(
        data=data, **kwargs))
    x, t, p_x = cp.Variable(DIM), cp.Variable(), cp.Variable(DIM)
    k = cvxro.ContextParameter(DIM, data=k_data)
    p = cvxro.ContextParameter(DIM, data=p_data)
    cons = [cvxro.max_of_uncertain(
        [-p[0]*x[0] - p[1]*x[1], -p[0]*x[0] - p_x[1]*u[1],
         -p_x[0]*u[0] - p[1]*x[1], -p_x[0]*u[0] - p_x[1]*u[1]]) + k@x <= t,
        p_x == p, x >= 0]
    ev = k@x + cp.maximum(-p[0]*x[0] - p[1]*x[1], -p[0]*x[0] - p[1]*u[1],
                          -p[0]*u[0] - p[1]*x[1], -p[0]*u[0] - p[1]*u[1])
    prob = cvxro.RobustProblem(cp.Minimize(t + 1e-4*cp.sum_squares(x)), cons, eval_exp=ev)

    settings = cvxro.TrainerSettings()
    settings.seed, settings.contextual, settings.num_iter = 5, True, 0
    settings.initialize_predictor = False
    settings.init_A, settings.init_b = np.eye(DIM), np.mean(data, axis=0)
    settings.data = data
    settings.indices_dict = {"train": np.arange(0, 30), "validate": np.arange(30, 45),
                             "test": np.arange(45, N)}
    settings.predictor = ConstantCenterPredictor(np.eye(DIM), center)
    return prob, cvxro.Trainer(prob), settings, x


class TestSupportCounterpart(unittest.TestCase):
    """The counterpart with a polyhedral support is exact -- this was suspected first."""

    def test_matches_direct_worst_case(self):
        A = np.array([[1.0, 0.3], [0.3, 1.0]])
        b = np.array([1.0, 1.5])
        for w in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([-2.0, -3.0])):
            for lb in (None, 0.0):
                self.assertAlmostEqual(_counterpart_value([w], A, b, RHO, lb),
                                       _worst_case(w, A, b, RHO, lb), places=4)

    def test_matches_through_max_of_uncertain(self):
        A = np.array([[1.0, 0.3], [0.3, 1.0]])
        b = np.array([1.0, 1.5])
        pieces = [np.array([-1.0, -1.0]), np.array([-2.0, 0.0]), np.array([0.0, -3.0])]
        for lb in (None, 0.0):
            exact = max(_worst_case(w, A, b, RHO, lb) for w in pieces)
            self.assertAlmostEqual(_counterpart_value(pieces, A, b, RHO, lb), exact,
                                   places=4)


class TestCenterProjection(unittest.TestCase):

    def _solve(self, trainer, settings, x_var, rho=1.0):
        trainer.train(settings=settings)
        x_tch = trainer.x_test_tch
        with torch.no_grad():
            a_tch, b_tch, _ = trainer.create_predictor_tensors(x_tch)
            z = trainer.cvxpylayer(torch.tensor(float(rho), dtype=torch.double),
                                   *trainer.cp_param_tch, *x_tch, a_tch, b_tch)
        idx = [v.id for v in trainer.cvxpylayer.variables].index(x_var.id)
        return np.asarray(z[idx].detach()), b_tch.numpy()

    def test_center_outside_support_is_projected(self):
        prob, trainer, settings, x_var = _newsvendor(lb=0.0, center=[-2.0, -2.0])
        _x, b = self._solve(trainer, settings, x_var)
        self.assertGreaterEqual(b.min(), -1e-9,
                                "center was left outside the support u >= 0")

    def test_projection_keeps_decisions_bounded(self):
        """The center is 2 units below the support and the radius is 1, so without the
        projection the set misses `u >= 0` entirely and the problem is unbounded."""
        prob, trainer, settings, x_var = _newsvendor(lb=0.0, center=[-2.0, -2.0])
        x_on, _ = self._solve(trainer, settings, x_var)

        prob, trainer, settings, x_var = _newsvendor(lb=0.0, center=[-2.0, -2.0])
        settings.project_center_to_support = False
        x_off, _ = self._solve(trainer, settings, x_var)

        self.assertLess(np.abs(x_on).max(), 1e2)
        self.assertGreater(np.abs(x_off).max(), 1e2,
                           "expected the unprojected, empty-set problem to blow up")

    def test_center_inside_support_is_untouched(self):
        """A center already in the support must pass through bit-identically, so the
        projection cannot change any run that was not already broken."""
        center = [3.0, 4.0]
        prob, trainer, settings, x_var = _newsvendor(lb=0.0, center=center)
        _x, b_on = self._solve(trainer, settings, x_var)

        prob, trainer, settings, x_var = _newsvendor(lb=0.0, center=center)
        settings.project_center_to_support = False
        _x, b_off = self._solve(trainer, settings, x_var)
        np.testing.assert_array_equal(b_on, b_off)

    def test_no_support_means_no_projection(self):
        """With no lb/ub there is nothing to project onto, and a negative center is a
        legitimate set -- so it must survive."""
        prob, trainer, settings, x_var = _newsvendor(lb=None, center=[-2.0, -2.0])
        _x, b = self._solve(trainer, settings, x_var)
        np.testing.assert_allclose(b, -2.0)


if __name__ == "__main__":
    unittest.main()
