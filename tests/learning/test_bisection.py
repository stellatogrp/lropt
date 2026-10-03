import unittest
import warnings

import cvxpy as cp
import numpy as np

import cvxro


def _sell_problem(data, n_samples, z_lower=0):
    """The sell-z problem from test_feasibility_metrics. Raising z_lower makes large
    radii genuinely unsolvable, which is what the unsolved guard has to survive."""
    u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(p=2, data=data))
    ctx = cvxro.ContextParameter(1, data=np.zeros((n_samples, 1)))
    z = cp.Variable(name="z")
    t = cp.Variable(name="t")
    return cvxro.RobustProblem(
        cp.Minimize(t),
        [cvxro.max_of_uncertain([-u[1] * z - t, z - u[0]], epigraph_index=0) <= 0,
         z >= z_lower, z <= 3, ctx >= -1e3],
        eval_exp=-u[1] * z,
    )


def _fixture(z_lower=0, n_samples=200):
    rng = np.random.RandomState(0)
    data = np.column_stack([
        rng.uniform(0.5, 4.0, n_samples), rng.uniform(1.0, 2.0, n_samples)
    ])
    third = n_samples // 3
    indices = {"train": np.arange(third),
               "validate": np.arange(third, 2 * third),
               "test": np.arange(2 * third, n_samples)}
    settings = cvxro.TrainerSettings()
    settings.data = data
    settings.indices_dict = indices
    settings.num_iter = 0
    settings.contextual = False
    settings.init_rho = 1.0
    settings.seed = 5
    settings.target_eta = 0.1
    trainer = cvxro.Trainer(_sell_problem(data, n_samples, z_lower))
    return trainer, settings


def _grid_choice(trainer, settings, rholst):
    """What gen_csv.select_best picks off a grid: among radii meeting the target, the
    best validation worst-case."""
    settings.track_unsolved = True
    df = trainer.grid(rholst=rholst, settings=settings).df
    acceptable = df[(df["Avg_prob_validate"].astype(float) <= settings.target_eta)
                    & (df["Frac_unsolved_validate"].astype(float) == 0)]
    if acceptable.empty:
        return None, df
    return acceptable.loc[
        acceptable["Validate_worst_feasible"].astype(float).idxmin()], df


class TestBisection(unittest.TestCase):
    def test_matches_a_fine_grid_and_costs_far_less(self):
        """The radius found must be at least as good as an exhaustive grid's, on the
        grid's own criterion, using an order of magnitude fewer evaluations."""
        rholst = np.logspace(np.log10(0.05), np.log10(3.0), 60)
        trainer, settings = _fixture()
        best, _ = _grid_choice(trainer, settings, rholst)

        trainer, settings = _fixture()
        result = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings, tol=0.01)
        row = result.df.loc[
            (result.df["Rho"].astype(float) - float(result.rho)).abs().idxmin()]

        self.assertLessEqual(float(row["Avg_prob_validate"]), settings.target_eta)
        # Not merely close to the grid's answer: the boundary generally falls between
        # grid points, so resolving it adaptively finds a smaller radius that still
        # meets the target, and hence a better worst-case.
        self.assertLessEqual(float(row["Validate_worst_feasible"]),
                             float(best["Validate_worst_feasible"]) + 1e-9)
        self.assertLess(len(result.df), 20)

    def test_probe_count_is_logarithmic(self):
        """Halving tol must cost a constant number of extra probes, not double them."""
        counts = []
        for tol in (0.08, 0.04, 0.02):
            trainer, settings = _fixture()
            counts.append(len(trainer.bisection(
                rho_lo=0.05, rho_hi=3.0, settings=settings, tol=tol).df))
        for coarse, fine in zip(counts, counts[1:]):
            self.assertEqual(fine - coarse, 1)

    def test_rejects_the_unsolvable_region(self):
        """Where the robust problem has no feasible decision, cvxpylayers returns a
        degenerate decision rather than failing. Such a radius must never be selected,
        and its recorded violation probability must say so: Avg_prob_* is the full
        failed-decision event, so failed solves count as violations (they used to read
        ~0 here, the degenerate decision's residuals looking clean)."""
        trainer, settings = _fixture(z_lower=1)
        result = trainer.bisection(rho_lo=0.05, rho_hi=20.0, settings=settings, tol=0.02)

        probes = result.df[["Rho", "Avg_prob_validate", "Frac_unsolved_validate"]]
        probes = probes.astype(float)
        unsolved = probes[probes["Frac_unsolved_validate"] > 0]
        self.assertGreater(len(unsolved), 0, "bracket did not reach the degenerate region")
        # The recorded rate now dominates the failed-solve fraction at every radius.
        gaps = unsolved["Avg_prob_validate"] - unsolved["Frac_unsolved_validate"]
        self.assertGreaterEqual(float(gaps.min()), -1e-9)

        chosen = result.df.loc[
            (result.df["Rho"].astype(float) - float(result.rho)).abs().idxmin()]
        self.assertEqual(float(chosen["Frac_unsolved_validate"]), 0.0)
        self.assertLess(float(result.rho), float(unsolved["Rho"].min()))

    def test_falls_back_and_warns_when_no_radius_meets_the_target(self):
        trainer, settings = _fixture()
        settings.target_eta = 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = trainer.bisection(
                rho_lo=0.01, rho_hi=0.05, settings=settings, tol=0.02)
        messages = [str(w.message) for w in caught if "target_eta" in str(w.message)]
        self.assertTrue(messages, "silent fallback")

        violations = result.df["Avg_prob_validate"].astype(float)
        chosen = result.df.loc[
            (result.df["Rho"].astype(float) - float(result.rho)).abs().idxmin()]
        self.assertAlmostEqual(float(chosen["Avg_prob_validate"]), violations.min())

    def test_df_carries_every_probe_and_the_usual_columns(self):
        trainer, settings = _fixture()
        result = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings, tol=0.05)
        self.assertGreater(len(result.df), 2)
        self.assertEqual(len(result.df["Rho"].astype(float).unique()), len(result.df))
        for column in ("Test_worst", "Validate_worst", "Avg_prob_validate",
                       "Validate_worst_feasible", "Frac_feasible_validate",
                       "Frac_unsolved_validate", "Test_time", "Validate_time"):
            self.assertIn(column, result.df.columns)

    def test_restores_track_unsolved(self):
        """bisection forces the setting on; DS is a shared global, so it must go back."""
        trainer, settings = _fixture()
        settings.track_unsolved = False
        trainer.bisection(rho_lo=0.1, rho_hi=2.0, settings=settings, tol=0.4)
        self.assertFalse(settings.track_unsolved)

    def test_target_eta_argument_tightens_the_selection_only(self):
        """The `target_eta` override must gate ACCEPTANCE without touching
        settings.target_eta, which is also the quantile the CVaR columns report at.
        A tighter target can only move the radius up, never down."""
        trainer, settings = _fixture()
        loose = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings, tol=0.01)
        self.assertEqual(settings.target_eta, 0.1)
        tight = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings, tol=0.01,
                                  target_eta=0.05)
        self.assertEqual(settings.target_eta, 0.1, "settings must be left alone")
        self.assertGreater(float(tight.rho), float(loose.rho))
        chosen = tight.df[np.isclose(tight.df["Rho"].astype(float), float(tight.rho))]
        self.assertLessEqual(float(chosen["Avg_prob_validate"].iloc[0]), 0.05)

    def test_target_eta_none_is_a_no_op(self):
        """The default must reproduce settings.target_eta exactly, so every run made
        before the argument existed is unaffected."""
        trainer, settings = _fixture()
        without = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings, tol=0.01)
        explicit = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings,
                                     tol=0.01, target_eta=settings.target_eta)
        self.assertEqual(float(without.rho), float(explicit.rho))

    def test_rejects_a_backwards_bracket(self):
        trainer, settings = _fixture()
        for lo, hi in ((3.0, 0.5), (0.0, 1.0), (-1.0, 1.0)):
            with self.assertRaises(ValueError):
                trainer.bisection(rho_lo=lo, rho_hi=hi, settings=settings)

    def test_stops_early_when_the_bracket_is_already_acceptable(self):
        """If the narrowest radius already meets the target, nothing narrower was in
        the bracket to find, so the search must not keep probing."""
        trainer, settings = _fixture()
        settings.target_eta = 0.95
        result = trainer.bisection(rho_lo=0.05, rho_hi=3.0, settings=settings, tol=0.01)
        self.assertEqual(len(result.df), 2)
        self.assertAlmostEqual(float(result.rho), 0.05)


if __name__ == "__main__":
    unittest.main()


class TestBisectionUnsolvedBracket(unittest.TestCase):
    """An unsolved probe must move the bracket UP, not down.

    Regression test for a search that converged onto an uncalibrated radius: at
    rho=0.283 the probe had Frac_unsolved_validate=0.005, the loop treated
    "not violating" as "acceptable" and pulled the upper bound down onto it, and every
    later probe stayed in a region whose violation rate was 0.97 against a 0.10 target
    (inventory_enhanced inve_miss_strong2/pretrain_nl seed 4).
    """

    @staticmethod
    def _stub_trainer(unsolved_band):
        """A trainer whose radii violate below 1.0 and are unsolvable inside the band."""
        import pandas as pd

        trainer, settings = _fixture()

        def fake_evaluate_rho(rho, settings, ctx, grid_stats):
            lo, hi = unsolved_band
            row = pd.Series({
                "Rho": rho,
                "Avg_prob_validate": 0.9 if rho < 1.0 else 0.0,
                "Frac_unsolved_validate": 0.005 if lo <= rho <= hi else 0.0,
                "Validate_worst_feasible": float(rho),
            })
            return row, None

        trainer._evaluate_rho = fake_evaluate_rho
        return trainer, settings

    def _probe_radii(self, unsolved_band):
        trainer, settings = self._stub_trainer(unsolved_band)
        probes = trainer._bisect_rho(0.004, 20.0, settings, tol=1e-3, max_iters=40,
                                     violation_col="Avg_prob_validate")
        return probes, np.array([p.rho for p in probes])

    def test_converges_above_an_unsolved_probe(self):
        # The band straddles 0.283, the radius the real search stalled on.
        probes, radii = self._probe_radii((0.25, 0.35))
        acceptable = [p for p in probes if p.status is cvxro.train.trainer._OK]
        self.assertTrue(acceptable, "no acceptable radius found; the search stalled")
        best = min(p.rho for p in acceptable)
        self.assertAlmostEqual(best, 1.0, delta=0.05)
        self.assertLess((radii < 0.5).sum(), len(radii),
                        "every probe fell below the unsolved band")

    def test_unchanged_when_no_probe_is_unsolved(self):
        _, with_band = self._probe_radii((1e9, 2e9))   # band never hit
        _, without = self._probe_radii((1e9, 2e9))
        np.testing.assert_allclose(with_band, without)
        self.assertAlmostEqual(min(r for r in with_band if r >= 1.0), 1.0, delta=0.05)
