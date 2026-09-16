"""A set with a support must never be EMPTY -- for any support, on any training path.

`Ellipsoidal(..., lb=/ub=/sum_eq=/c=,d=)` intersects the norm ball `{A z + b : ||z|| <=
rho}` with the polyhedron `{u : C u <= d}`. If the center `b` lies outside that polyhedron
and the radius is too small to reach back in, the intersection is empty -- and an empty
uncertainty set makes the robust constraint quantify over nothing, so it bounds nothing.
The solver then returns an enormous decision and reports `optimal`, and no violation metric
catches it, because the decision really is feasible for a constraint that says nothing.

Measured on the newsvendor with `lb=0`: 9% of contexts had a center with a negative
coordinate, up to 4% of sets were empty, and those contexts came back with `|x| ~ 1e4` and
costs three orders of magnitude off, against `-1.3` for the same arm with no support.

THE FIX is one condition: keep the center inside the support, so that the center itself
lies in both the ball and the support and the intersection is nonempty by construction.
`TrainerSettings.project_center_to_support` (default True) enforces it at the two places a
center is produced:

  * contextual runs -- `Trainer.create_predictor_tensors`, the single point both the
    training and the evaluation paths take A(x), b(x) from;
  * non-contextual runs -- after each optimizer step, making the update a projected
    gradient step, since there `b` is a trained leaf tensor the loop can walk out of the
    support even when it starts inside.

Both handle ANY support: a box by clamping, anything else (explicit `c`/`d`, `sum_eq`, or
a mix) by Dykstra's alternating projections.

The first test class pins down what was wrongly suspected first: the counterpart ITSELF is
exact with a support, for every support kind, including through `max_of_uncertain`. The bug
was never in the reformulation.
"""

import unittest
import warnings

import cvxpy as cp
import numpy as np
import torch

import cvxro
from cvxro.uncertainty_sets.support import (
    box_bounds,
    project_into_support,
    support_matrices,
)

N, DIM = 60, 2
RHO = 2.0
A_FIXED = np.array([[1.0, 0.3], [0.3, 1.0]])

# Each support, as the kwargs a set takes and as the explicit rows it must be equivalent
# to. Keeping both sides here is what lets the counterpart be checked against a direct
# maximization over the same region.
SUPPORTS = {
    "lb": (dict(lb=0.0), -np.eye(DIM), np.zeros(DIM)),
    "lb+ub": (dict(lb=0.0, ub=4.0),
              np.vstack([-np.eye(DIM), np.eye(DIM)]),
              np.concatenate([np.zeros(DIM), 4 * np.ones(DIM)])),
    "vector bounds": (dict(lb=np.array([0.0, 1.0]), ub=np.array([4.0, 5.0])),
                      np.vstack([-np.eye(DIM), np.eye(DIM)]),
                      np.array([0.0, -1.0, 4.0, 5.0])),
    "polyhedral": (dict(c=np.array([[1.0, 1.0], [-1.0, 0.0]]), d=np.array([5.0, 1.0])),
                   np.array([[1.0, 1.0], [-1.0, 0.0]]), np.array([5.0, 1.0])),
    "sum_eq": (dict(sum_eq=3.0), np.array([[1.0, 1.0], [-1.0, -1.0]]),
               np.array([3.0, -3.0])),
}


def _worst_case(w, b, rows, rhs):
    """max w'u over the ball intersected with the support, by direct maximization."""
    z = cp.Variable(DIM)
    u = A_FIXED @ z + b
    prob = cp.Problem(cp.Maximize(w @ u), [cp.norm(z, 2) <= RHO, rows @ u <= rhs])
    prob.solve(solver="CLARABEL")
    return float(prob.value)


def _counterpart(pieces, b, kwargs):
    """Smallest t with `max_i w_i'u <= t` over the set, through cvxro's reformulation."""
    u = cvxro.UncertainParameter(
        DIM, uncertainty_set=cvxro.Ellipsoidal(rho=RHO, a=A_FIXED, b=b, **kwargs))
    t = cp.Variable()
    rows = [w @ u for w in pieces]
    cons = [rows[0] <= t] if len(rows) == 1 else [cvxro.max_of_uncertain(rows) <= t]
    cvxro.RobustProblem(cp.Minimize(t), cons).solve()
    return float(t.value)


class TestSupportCounterpart(unittest.TestCase):
    """The counterpart is exact with a support -- this is what was suspected and cleared."""

    def test_matches_direct_worst_case(self):
        b = np.array([1.0, 1.5])
        for name, (kwargs, rows, rhs) in SUPPORTS.items():
            for w in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([-2.0, -3.0])):
                with self.subTest(support=name, w=tuple(w)):
                    self.assertAlmostEqual(_counterpart([w], b, kwargs),
                                           _worst_case(w, b, rows, rhs), places=4)

    def test_matches_through_max_of_uncertain(self):
        b = np.array([1.0, 1.5])
        pieces = [np.array([-1.0, -1.0]), np.array([-2.0, 0.0]), np.array([0.0, -3.0])]
        for name, (kwargs, rows, rhs) in SUPPORTS.items():
            with self.subTest(support=name):
                exact = max(_worst_case(w, b, rows, rhs) for w in pieces)
                self.assertAlmostEqual(_counterpart(pieces, b, kwargs), exact, places=4)


class TestProjection(unittest.TestCase):
    """The projection itself, against cvxpy's answer to the same problem."""

    POLYHEDRA = {
        "box": (-np.eye(2), np.zeros(2)),
        "box both sides": (np.vstack([-np.eye(2), np.eye(2)]),
                           np.array([0.0, 0.0, 4.0, 4.0])),
        "triangle": (np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
                     np.array([3.0, 1.0, 1.0])),
        "rotated": (np.array([[1.0, -2.0], [2.0, 1.0], [-1.0, -1.0]]),
                    np.array([1.0, 4.0, 0.5])),
        "equality": (np.array([[1.0, 1.0], [-1.0, -1.0]]), np.array([3.0, -3.0])),
    }
    POINTS = [np.array([-3.0, -3.0]), np.array([1.5, 2.0]), np.array([9.0, -4.0]),
              np.array([0.0, 0.0]), np.array([5.0, 5.0])]

    def _cvxpy_projection(self, b, rows, rhs):
        u = cp.Variable(len(b))
        cp.Problem(cp.Minimize(cp.sum_squares(u - b)), [rows @ u <= rhs]).solve(
            solver="CLARABEL")
        return np.asarray(u.value)

    def test_matches_cvxpy(self):
        for name, (rows, rhs) in self.POLYHEDRA.items():
            for b in self.POINTS:
                with self.subTest(polyhedron=name, b=tuple(b)):
                    got, residual = project_into_support(torch.tensor(b), rows, rhs)
                    self.assertLessEqual(residual, 1e-6)
                    np.testing.assert_allclose(
                        got.numpy(), self._cvxpy_projection(b, rows, rhs), atol=1e-4)

    def test_feasible_point_is_unchanged(self):
        """A center already inside must pass through untouched, or the projection would be
        perturbing fits it has no business touching."""
        rows, rhs = self.POLYHEDRA["triangle"]
        inside = torch.tensor([0.0, 0.5], dtype=torch.double)
        got, _ = project_into_support(inside, rows, rhs)
        np.testing.assert_allclose(got.numpy(), inside.numpy(), atol=1e-12)

    def test_batched_and_differentiable(self):
        rows, rhs = self.POLYHEDRA["triangle"]
        b = torch.tensor(np.vstack(self.POINTS), requires_grad=True)
        got, residual = project_into_support(b, rows, rhs)
        got.sum().backward()
        self.assertEqual(tuple(got.shape), (len(self.POINTS), DIM))
        self.assertLessEqual(residual, 1e-6)
        self.assertTrue(bool(torch.isfinite(b.grad).all()))

    def test_box_detected_and_general_not(self):
        self.assertIsNotNone(box_bounds(*self.POLYHEDRA["box both sides"], DIM))
        self.assertIsNone(box_bounds(*self.POLYHEDRA["triangle"], DIM))

    def test_empty_box_is_reported(self):
        """Contradictory bounds have no projection; the caller must be able to tell."""
        rows = np.vstack([np.eye(2), -np.eye(2)])
        rhs = np.array([0.0, 0.0, -5.0, -5.0])          # u <= 0 and u >= 5
        _got, residual = project_into_support(torch.tensor([1.0, 1.0]), rows, rhs)
        self.assertFalse(np.isfinite(residual) and residual <= 1e-6)


class ConstantCenterPredictor(torch.nn.Module):
    """A predictor returning a fixed A and b for every context.

    Deliberately not fitted: the point is to hand the trainer a center OUTSIDE the support
    and see what the set does, which a fitted head would only reach at some contexts.
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


def _problem(support, contextual, center, num_iter=0, seed=0):
    """A small two-product newsvendor, contextual or not, centered outside the support."""
    rng = np.random.default_rng(seed)
    data = np.maximum(0.0, rng.normal(3.0, 1.0, size=(N, DIM)))
    k_data = np.maximum(0.5, rng.normal(4.0, 1.0, size=(N, DIM)))
    p_data = k_data + np.abs(rng.normal(0.0, 1.0, size=(N, DIM)))

    u = cvxro.UncertainParameter(DIM, uncertainty_set=cvxro.Ellipsoidal(
        data=data, **support))
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
    settings.seed, settings.contextual, settings.num_iter = 5, contextual, num_iter
    settings.lr, settings.init_rho = 0.05, 1.0
    settings.initialize_predictor = False
    settings.init_A = np.eye(DIM)
    settings.init_b = np.asarray(center, dtype=float)
    settings.data = data
    settings.indices_dict = {"train": np.arange(0, 30), "validate": np.arange(30, 45),
                             "test": np.arange(45, N)}
    if contextual:
        settings.predictor = ConstantCenterPredictor(np.eye(DIM), center)
    return prob, cvxro.Trainer(prob), settings, x


def _max_violation(centers, trainer):
    """How far outside the support the centers are (0 when all inside)."""
    rows, rhs = support_matrices(trainer.unc_set)
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    return float(np.max(np.maximum(centers @ rows.T - rhs, 0.0)))


def _centers_after_training(trainer, settings, contextual):
    result = trainer.train(settings=settings)
    if contextual:
        with torch.no_grad():
            _a, b_tch, _r = trainer.create_predictor_tensors(trainer.x_test_tch)
        return b_tch.numpy()
    return np.asarray(result.b, dtype=float)


class TestCenterStaysInSupport(unittest.TestCase):
    """The guarantee, on every combination of support kind and training path."""

    CASES = {
        "box": dict(lb=0.0),
        "polyhedral": dict(c=np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
                           d=np.array([6.0, 0.0, 0.0])),
        "sum_eq": dict(sum_eq=4.0),
    }

    def test_center_is_inside_after_training(self):
        for name, support in self.CASES.items():
            for contextual in (False, True):
                with self.subTest(support=name, contextual=contextual):
                    prob, trainer, settings, _x = _problem(
                        support, contextual, center=[-3.0, -3.0], num_iter=4)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        centers = _centers_after_training(trainer, settings, contextual)
                    self.assertLessEqual(_max_violation(centers, trainer), 1e-6)

    def test_projection_off_leaves_the_center_outside(self):
        """The guard is load-bearing: with it off, the center stays out of the support.

        Without this the tests above could pass for reasons unrelated to the projection --
        a center that happened never to leave the support would satisfy them too.
        """
        for name in ("box", "polyhedral"):
            with self.subTest(support=name):
                prob, trainer, settings, _x = _problem(
                    self.CASES[name], contextual=True, center=[-3.0, -3.0], num_iter=2)
                settings.project_center_to_support = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    centers = _centers_after_training(trainer, settings, True)
                self.assertGreater(_max_violation(centers, trainer), 1e-3)

    def test_no_support_leaves_the_center_alone(self):
        """A negative center is a legitimate set when nothing declares a support."""
        prob, trainer, settings, _x = _problem({}, contextual=True, center=[-3.0, -3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            centers = _centers_after_training(trainer, settings, True)
        np.testing.assert_allclose(centers, -3.0, atol=1e-9)

    def test_center_already_inside_is_untouched(self):
        """Bit-identical pass-through, so no run that was already correct can change."""
        prob, trainer, settings, _x = _problem(
            dict(lb=0.0), contextual=True, center=[3.0, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            on = _centers_after_training(trainer, settings, True)

        prob, trainer, settings, _x = _problem(
            dict(lb=0.0), contextual=True, center=[3.0, 4.0])
        settings.project_center_to_support = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            off = _centers_after_training(trainer, settings, True)
        np.testing.assert_array_equal(on, off)


class TestEmptySupportRaises(unittest.TestCase):

    def test_contradictory_bounds_raise(self):
        """`lb` above `ub` has no feasible center at all. That is a modelling error, and
        it should say so rather than surface later as an unbounded solve."""
        prob, trainer, settings, _x = _problem(
            dict(lb=5.0, ub=0.0), contextual=True, center=[1.0, 1.0])
        with self.assertRaises(ValueError) as ctx:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.train(settings=settings)
        self.assertIn("support is empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
