"""Tests for settings.line_search_mode = "backtrack" (Trainer._backtracking_line_search).

The legacy loop's descent test never rejected a step, and making it live blew up
training (lropt_power/multistage HANDOFF §5r). These tests pin the properties the fix
exists for: a non-improving step is re-taken smaller FROM THE SAME SNAPSHOT, the lr
comes back to exactly lr0, Adam's state advances once per accepted step, and an
all-infeasible search leaves the parameters and optimizer state where they were.

The unit tests drive the method on a stub trainer (a quadratic in one parameter), so
they need no cvxpylayer; the last class runs the real trainer on the tiny portfolio
problem from test_al_strategies.
"""

import copy
import types
import unittest

import numpy as np
import torch

from cvxro import Trainer, TrainerSettings
from cvxro.violation_checker.utils import CONSTRAINT_STATUS

from .test_al_strategies import _base_settings, _make_portfolio_problem

FEAS, INFEAS = CONSTRAINT_STATUS.FEASIBLE, CONSTRAINT_STATUS.INFEASIBLE


def _stub(theta, loss_fn, status_fn=lambda: FEAS, max_iter=10, mult=0.5, threshold=1.0,
          on_exhaust="keep_last"):
    """A stand-in for Trainer with just what _backtracking_line_search reads."""
    stub = types.SimpleNamespace()
    stub.settings = types.SimpleNamespace(
        line_search_mult=mult, line_search_threshold=threshold, line_search=True,
        line_search_on_exhaust=on_exhaust)
    stub._max_iter_line_search = max_iter
    stub._project_step = None
    stub.n_evals = 0
    stub.seeds = []

    def evaluate(a_tch, b_tch, rho_tch, alpha, lam, mu, seed_num):
        stub.n_evals += 1
        stub.seeds.append(seed_num)
        fin = loss_fn(theta)
        z = torch.zeros(1, dtype=torch.float64)
        return status_fn(), fin, z, z, z, z, z, z

    stub._evaluate_fin_cost = evaluate
    return stub


def _search(stub, opt, prev_fin_cost, step_num=1, scheduler=None, resample_seed_num=None):
    rho = torch.tensor(1.0, dtype=torch.float64)
    return Trainer._backtracking_line_search(
        stub, step_num=step_num, a_tch=None, b_tch=None, rho_tch=rho, alpha=None,
        lam=None, mu=None, opt=opt, scheduler=scheduler, seed_num=0,
        prev_fin_cost=prev_fin_cost, resample_seed_num=resample_seed_num)


def _primed(lr, theta0=3.0):
    """A parameter, an Adam optimizer that has taken one step, and a gradient in p.grad
    -- the state the train loop hands over after an accepted step."""
    theta = torch.nn.Parameter(torch.tensor([theta0], dtype=torch.float64))
    opt = torch.optim.Adam([theta], lr=lr)
    (theta ** 2).sum().backward()
    opt.step()
    opt.zero_grad()
    loss = (theta ** 2).sum()
    loss.backward()
    return theta, opt, loss.detach()


class TestBacktrackingMechanics(unittest.TestCase):

    def test_overshoot_is_retaken_smaller_from_the_snapshot(self):
        # Adam's steps are ~lr in size, so lr 10 overshoots x^2 from x ~ 3 badly. Prime
        # at a small lr: priming at 10 would already have jumped past the minimum.
        theta, opt, prev = _primed(lr=0.1)
        opt.param_groups[0]["lr"] = 10.0
        ref_theta = copy.deepcopy(theta.detach())
        ref_state = copy.deepcopy(opt.state_dict())
        step0 = int(opt.state[theta]["step"])
        stub = _stub(theta, lambda t: (t ** 2).sum())

        out = _search(stub, opt, prev)
        prev_new, tries, outcome = out[8], out[9], out[10]

        self.assertEqual(outcome, "accepted")
        self.assertGreater(tries, 1)
        self.assertLessEqual(float(prev_new), float(prev))
        self.assertEqual(opt.param_groups[0]["lr"], 10.0)
        self.assertEqual(int(opt.state[theta]["step"]), step0 + 1)

        # The accepted point is exactly ONE Adam step from the snapshot at lr*mult**(k-1).
        replica = torch.nn.Parameter(ref_theta.clone())
        ropt = torch.optim.Adam([replica], lr=10.0)
        ropt.load_state_dict(ref_state)
        ropt.param_groups[0]["lr"] = 10.0 * 0.5 ** (tries - 1)
        replica.grad = 2 * ref_theta.clone()  # the gradient _primed left in p.grad
        ropt.step()
        torch.testing.assert_close(theta.detach(), replica.detach())

    def test_no_descent_accepts_smallest_step_and_lr_is_not_inflated(self):
        # The legacy path multiplied lr by (1/mult)**(max_iter-1) here and exploded.
        theta, opt, _ = _primed(lr=0.1)
        stub = _stub(theta, lambda t: (t ** 2).sum(), max_iter=6)
        out = _search(stub, opt, prev_fin_cost=torch.tensor(-1.0, dtype=torch.float64))
        self.assertEqual(out[10], "no_descent")
        self.assertEqual(out[9], 6)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)

    def test_all_infeasible_reverts_parameters_and_optimizer_state(self):
        theta, opt, prev = _primed(lr=0.1)
        theta_before = theta.detach().clone()
        step_before = int(opt.state[theta]["step"])
        exp_avg_before = opt.state[theta]["exp_avg"].clone()
        stub = _stub(theta, lambda t: (t ** 2).sum(), status_fn=lambda: INFEAS, max_iter=5)

        out = _search(stub, opt, prev)

        self.assertIs(out[0], INFEAS)
        self.assertEqual(out[10], "infeasible")
        self.assertEqual(float(out[8]), float(prev))
        torch.testing.assert_close(theta.detach(), theta_before)
        self.assertEqual(int(opt.state[theta]["step"]), step_before)
        torch.testing.assert_close(opt.state[theta]["exp_avg"], exp_avg_before)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)
        self.assertEqual(stub.n_evals, 6)  # 5 tries + the re-evaluation at the snapshot

    def test_scheduler_ticks_once_per_accepted_step_only(self):
        theta, opt, prev = _primed(lr=10.0)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        out = _search(_stub(theta, lambda t: (t ** 2).sum()), opt, prev, scheduler=sched)
        self.assertEqual(out[10], "accepted")
        self.assertEqual(opt.param_groups[0]["lr"], 5.0)

        theta, opt, prev = _primed(lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        _search(_stub(theta, lambda t: (t ** 2).sum(), status_fn=lambda: INFEAS,
                      max_iter=3), opt, prev, scheduler=sched)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)

    def test_no_descent_revert_restores_snapshot_and_skips_scheduler(self):
        theta, opt, _ = _primed(lr=0.1)
        theta_before = theta.detach().clone()
        step_before = int(opt.state[theta]["step"])
        exp_avg_before = opt.state[theta]["exp_avg"].clone()
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        stub = _stub(theta, lambda t: (t ** 2).sum(), max_iter=4, on_exhaust="revert")

        out = _search(stub, opt, torch.tensor(-1.0, dtype=torch.float64), scheduler=sched)

        self.assertIs(out[0], FEAS)
        self.assertEqual(out[10], "no_descent")
        torch.testing.assert_close(theta.detach(), theta_before)
        self.assertEqual(int(opt.state[theta]["step"]), step_before)
        torch.testing.assert_close(opt.state[theta]["exp_avg"], exp_avg_before)
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)  # no scheduler tick
        self.assertEqual(stub.n_evals, 5)  # 4 tries + the re-evaluation at the snapshot
        # The descent reference is now the snapshot's own cost.
        self.assertAlmostEqual(float(out[8]), float((theta_before ** 2).sum()))
        self.assertIsNotNone(theta.grad)

    def test_resample_reevaluates_on_the_new_batch(self):
        for status, outcome in ((FEAS, "no_descent"), (INFEAS, "infeasible")):
            theta, opt, _ = _primed(lr=0.1)
            theta_before = theta.detach().clone()
            stub = _stub(theta, lambda t: (t ** 2).sum(), status_fn=lambda s=status: s,
                         max_iter=3, on_exhaust="revert")
            out = _search(stub, opt, torch.tensor(-1.0, dtype=torch.float64),
                          resample_seed_num=77)
            self.assertEqual(out[10], outcome)
            self.assertEqual(stub.seeds, [0, 0, 0, 77])
            # Reference taken on the new batch in BOTH cases (infeasible too).
            self.assertAlmostEqual(float(out[8]), float((theta_before ** 2).sum()))

    def test_resample_unused_on_acceptance(self):
        theta, opt, prev = _primed(lr=0.1)
        stub = _stub(theta, lambda t: (t ** 2).sum(), on_exhaust="revert")
        out = _search(stub, opt, prev, resample_seed_num=77)
        self.assertEqual(out[10], "accepted")
        self.assertNotIn(77, stub.seeds)

    def test_first_step_takes_no_step(self):
        theta, opt, _ = _primed(lr=0.1)
        before = theta.detach().clone()
        out = _search(_stub(theta, lambda t: (t ** 2).sum()), opt, np.inf, step_num=0)
        self.assertEqual(out[10], "first_step")
        torch.testing.assert_close(theta.detach(), before)
        self.assertTrue(np.isfinite(float(out[8])))


class TestBacktrackingTrainer(unittest.TestCase):

    def setUp(self):
        self.n, self.N = 2, 20
        _, self.trainer, self.data = _make_portfolio_problem(self.n, self.N)

    def _settings(self, mode):
        settings = _base_settings(self.data, self.n, self.N)
        settings.line_search_mode = mode
        settings.max_iter_line_search = 8
        settings.line_search_mult = 0.5
        return settings

    def test_default_is_legacy(self):
        self.assertEqual(TrainerSettings().line_search_mode, "legacy")

    def test_backtrack_trains_and_descends_within_a_batch(self):
        settings = self._settings("backtrack")
        settings.lr = 0.05  # large enough that some steps must be backtracked
        df = self.trainer.train(settings=settings).df
        self.assertIn("Line_search_tries", df.columns)
        self.assertTrue(np.all(np.isfinite(df["Lagrangian_val"].astype(float))))
        # Within one AL interval (same batch, same multipliers) accepted steps descend.
        interval = settings.aug_lag_update_interval
        lag = df["Lagrangian_val"].astype(float).to_numpy()
        outcome = df["Line_search_outcome"].to_numpy()
        for k in range(1, len(df)):
            # prev_fin_cost is reset after every step with step % interval == 0.
            if (k - 1) % interval != 0 and outcome[k] == "accepted":
                self.assertLessEqual(lag[k], lag[k - 1] + 1e-12)

    def test_revert_resample_moves_the_batch_on_rejection(self):
        settings = self._settings("backtrack")
        settings.lr = 0.05
        settings.line_search_on_exhaust = "revert"
        settings.line_search_resample = True
        df = self.trainer.train(settings=settings).df
        self.assertTrue(np.all(np.isfinite(df["Lagrangian_val"].astype(float))))
        rejected = df["Line_search_outcome"].isin(["no_descent", "infeasible"]).to_numpy()
        batch = df["Line_search_batch"].astype(int).to_numpy()
        np.testing.assert_array_equal(batch, np.cumsum(rejected))

    def test_new_knobs_default_off(self):
        st = TrainerSettings()
        self.assertEqual(st.line_search_on_exhaust, "keep_last")
        self.assertFalse(st.line_search_resample)

    def test_legacy_rows_unchanged(self):
        df = self.trainer.train(settings=self._settings("legacy")).df
        self.assertNotIn("Line_search_tries", df.columns)


if __name__ == "__main__":
    unittest.main()
