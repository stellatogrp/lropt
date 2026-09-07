import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt

import cvxro
from cvxro.train.settings import TOL


def _sell_problem(data, epigraph_index=0):
    """Sell z units. Epigraph row -u1*z - t <= 0 is the certificate; the other row,
    z - u0 <= 0, says one cannot sell more than the demand that showed up. Feasibility
    excluding the epigraph is therefore exactly z <= u0, which is hand-checkable."""
    n_samples = data.shape[0]
    u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(p=2, data=data))
    ctx = cvxro.ContextParameter(1, data=np.zeros((n_samples, 1)))
    z = cp.Variable(name="z")
    t = cp.Variable(name="t")
    rows = [-u[1] * z - t, z - u[0]]
    problem = cvxro.RobustProblem(
        cp.Minimize(t),
        [cvxro.max_of_uncertain(rows, epigraph_index=epigraph_index) <= 0,
         z >= 0, z <= 3, ctx >= -1e3],
        eval_exp=-u[1] * z,
    )
    return problem


def _settings(data, indices):
    settings = cvxro.TrainerSettings()
    settings.data = data
    settings.indices_dict = indices
    settings.num_iter = 0
    settings.contextual = False
    settings.init_rho = 1.0
    settings.seed = 5
    return settings


class TestEpigraphIndex(unittest.TestCase):
    def test_rejects_out_of_range(self):
        u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(rho=1.0))
        rows = [u[0], u[1], u[0] + u[1]]
        for bad in (3, -1):
            with self.assertRaises(ValueError):
                cvxro.max_of_uncertain(rows, epigraph_index=bad)

    def test_rejects_several_lists(self):
        u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(rho=1.0))
        with self.assertRaises(ValueError):
            cvxro.sum_of_max_of_uncertain([[u[0], u[1]], [u[0], u[1]]], epigraph_index=0)

    def test_defaults_to_none(self):
        u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(rho=1.0))
        self.assertIsNone(cvxro.max_of_uncertain([u[0], u[1]]).epigraph_index)


class TestFeasibilityMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n_samples = 400
        self.data = np.column_stack([
            rng.uniform(0.5, 4.0, n_samples), rng.uniform(1.0, 2.0, n_samples)
        ])
        self.indices = {
            "train": np.arange(200),
            "validate": np.arange(200, 300),
            "test": np.arange(300, 400),
        }
        self.rholst = np.array([0.2, 0.5, 1.0])

    def _grid(self, epigraph_index=0):
        problem = _sell_problem(self.data, epigraph_index)
        trainer = cvxro.Trainer(problem)
        df = trainer.grid(rholst=self.rholst,
                          settings=_settings(self.data, self.indices)).df
        z_pos = [v.name() for v in trainer._reduced_variables].index("z")
        return df, z_pos

    def test_matches_hand_computation(self):
        """Every feasibility-filtered statistic must equal a direct numpy computation."""
        df, z_pos = self._grid(epigraph_index=0)
        u_test = self.data[self.indices["test"]]
        for _, row in df.iterrows():
            z = float(np.ravel(np.asarray(row["z_vals"][z_pos]))[0])
            feasible = (z - u_test[:, 0]) <= TOL
            raw = -u_test[:, 1] * z
            npt.assert_allclose(float(row["Frac_feasible_test"]), feasible.mean())

            kept = raw[feasible]
            count = len(kept)
            quantile = np.sort(kept)[min(count - 1, int(0.9 * count) + 1)]
            npt.assert_allclose(float(row["Test_val_feasible"]), kept.mean())
            npt.assert_allclose(float(row["Test_worst_feasible"]), quantile)
            npt.assert_allclose(float(row["Test_cvar_feasible"]),
                                kept[kept >= quantile].mean())

    def test_pct_infeasible_complements_frac_feasible(self):
        df, _ = self._grid()
        npt.assert_allclose(df["Pct_infeasible_test"].astype(float),
                            1 - df["Frac_feasible_test"].astype(float))

    def test_validate_block_is_present_and_distinct(self):
        df, _ = self._grid()
        self.assertIn("Frac_feasible_validate", df.columns)
        self.assertFalse(np.allclose(df["Frac_feasible_validate"].astype(float),
                                     df["Frac_feasible_test"].astype(float)))

    def test_unmarked_epigraph_leaves_behaviour_unchanged(self):
        """With epigraph_index=None nothing is excluded, so feasibility is the
        complement of the violation probability and the existing columns are untouched."""
        marked, _ = self._grid(epigraph_index=0)
        unmarked, _ = self._grid(epigraph_index=None)
        for column in ("Test_worst", "Test_val", "Test_cvar", "Validate_worst"):
            npt.assert_allclose(marked[column].astype(float),
                                unmarked[column].astype(float))
        avg_prob = np.array([float(np.mean(v)) for v in unmarked["Avg_prob_test"]])
        npt.assert_allclose(unmarked["Frac_feasible_test"].astype(float), 1 - avg_prob)

    def test_excluding_the_epigraph_admits_at_least_as_many_samples(self):
        marked, _ = self._grid(epigraph_index=0)
        unmarked, _ = self._grid(epigraph_index=None)
        self.assertTrue(np.all(marked["Frac_feasible_test"].astype(float)
                               >= unmarked["Frac_feasible_test"].astype(float) - 1e-12))


class TestSolutionTimeApi(unittest.TestCase):
    def test_feasibility_indicator_sol(self):
        rng = np.random.RandomState(0)
        data = np.column_stack([rng.uniform(0.5, 4.0, 60), rng.uniform(1.0, 2.0, 60)])
        u = cvxro.UncertainParameter(2,
                                     uncertainty_set=cvxro.Ellipsoidal(p=2, rho=0.5, data=data))
        z = cp.Variable(name="z")
        t = cp.Variable(name="t")
        problem = cvxro.RobustProblem(
            cp.Minimize(t),
            [cvxro.max_of_uncertain([-u[1] * z - t, z - u[0]], epigraph_index=0) <= 0,
             z >= 0, z <= 3],
            eval_exp=-u[1] * z,
        )
        u.eval_data = data
        problem.solve()
        feasible = problem.feasibility_indicator_sol()
        npt.assert_allclose(feasible.mean(), np.mean(z.value <= data[:, 0] + TOL))
        violation = problem.violation_indicator_sol(exclude_epigraph=True)
        npt.assert_allclose(feasible, violation.sum(axis=0) == 0)


class TestUnsolvedTracking(unittest.TestCase):
    def test_flags_radii_with_no_feasible_decision(self):
        """cvxpylayers returns a degenerate decision rather than failing when the robust
        problem is infeasible, which makes the violation probability read ~0. The
        feasibility columns must not be fooled by that."""
        rng = np.random.RandomState(0)
        n_samples = 300
        data = np.column_stack([
            rng.uniform(0.5, 4.0, n_samples), rng.uniform(1.0, 2.0, n_samples)
        ])
        indices = {"train": np.arange(150), "validate": np.arange(150, 225),
                   "test": np.arange(225, 300)}
        u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(p=2, data=data))
        ctx = cvxro.ContextParameter(1, data=np.zeros((n_samples, 1)))
        z = cp.Variable(name="z")
        t = cp.Variable(name="t")
        problem = cvxro.RobustProblem(
            cp.Minimize(t),
            [cvxro.max_of_uncertain([-u[1] * z - t, z - u[0]], epigraph_index=0) <= 0,
             z >= 1, z <= 3, ctx >= -1e3],
            eval_exp=-u[1] * z,
        )
        settings = _settings(data, indices)
        settings.track_unsolved = True
        df = cvxro.Trainer(problem).grid(rholst=np.array([0.2, 6.0]), settings=settings).df

        small, large = df.iloc[0], df.iloc[1]
        self.assertAlmostEqual(float(small["Frac_unsolved_test"]), 0.0)
        self.assertAlmostEqual(float(large["Frac_unsolved_test"]), 1.0)
        # cvxro's own violation probability is the misleading one here
        self.assertAlmostEqual(float(np.mean(large["Avg_prob_test"])), 0.0, places=6)
        self.assertAlmostEqual(float(large["Frac_feasible_test"]), 0.0)

    def _unsolvable_at_large_rho(self):
        """The fixture above, factored out: rho=0.2 is solvable, rho=6.0 is not."""
        rng = np.random.RandomState(0)
        n_samples = 300
        data = np.column_stack([
            rng.uniform(0.5, 4.0, n_samples), rng.uniform(1.0, 2.0, n_samples)
        ])
        indices = {"train": np.arange(150), "validate": np.arange(150, 225),
                   "test": np.arange(225, 300)}
        u = cvxro.UncertainParameter(2, uncertainty_set=cvxro.Ellipsoidal(p=2, data=data))
        ctx = cvxro.ContextParameter(1, data=np.zeros((n_samples, 1)))
        z = cp.Variable(name="z")
        t = cp.Variable(name="t")
        problem = cvxro.RobustProblem(
            cp.Minimize(t),
            [cvxro.max_of_uncertain([-u[1] * z - t, z - u[0]], epigraph_index=0) <= 0,
             z >= 1, z <= 3, ctx >= -1e3],
            eval_exp=-u[1] * z,
        )
        return problem, _settings(data, indices)

    def test_grid_populates_frac_unsolved_even_when_the_caller_left_it_off(self):
        """`track_unsolved` is not merely a reporting flag: monte_carlo ANDs the
        solvability mask into the feasibility mask, so with it off the samples whose
        robust problem had no feasible decision are counted as FEASIBLE and their
        degenerate cvxpylayers decisions are averaged into the *_feasible columns.
        grid() therefore forces it on, exactly as bisection() does."""
        problem, settings = self._unsolvable_at_large_rho()
        settings.track_unsolved = False
        df = cvxro.Trainer(problem).grid(
            rholst=np.array([0.2, 6.0]), settings=settings).df

        self.assertAlmostEqual(float(df.iloc[1]["Frac_unsolved_test"]), 1.0)
        # and the feasibility columns are not fooled by the degenerate decision
        self.assertAlmostEqual(float(df.iloc[1]["Frac_feasible_test"]), 0.0)
        # the caller's setting is restored, not clobbered
        self.assertFalse(settings.track_unsolved)

    def test_grid_never_selects_an_unsolvable_radius(self):
        """The radius written onto the uncertainty set must come from a radius where a
        decision actually exists. GridStats minimized the TEST objective with no regard
        to solvability or to target_eta, so an unsolvable radius -- whose degenerate
        decision reads as violation ~0 -- could win the selection outright."""
        problem, settings = self._unsolvable_at_large_rho()
        settings.track_unsolved = True
        result = cvxro.Trainer(problem).grid(
            rholst=np.array([0.2, 6.0]), settings=settings)

        self.assertAlmostEqual(float(np.ravel(result.rho)[0]), 0.2)
        # both radii are still REPORTED; only the selection excludes the unsolvable one
        self.assertEqual(len(result.df), 2)

    def test_grid_and_bisection_agree_on_the_selected_radius(self):
        """`rho_search` in the experiment folders switches between the two and documents
        them as finding "the same thing ... differ only in how they sample it". That was
        not true of the selection before this: grid() picked on the test objective while
        bisection() picked the best radius meeting target_eta among solvable ones."""
        problem, settings = self._unsolvable_at_large_rho()
        settings.track_unsolved = True
        rholst = np.array([0.2, 0.4, 0.8, 1.6, 3.2, 6.0])
        grid_rho = float(np.ravel(cvxro.Trainer(problem).grid(
            rholst=rholst, settings=settings).rho)[0])

        problem2, settings2 = self._unsolvable_at_large_rho()
        settings2.track_unsolved = True
        bis_rho = float(np.ravel(cvxro.Trainer(problem2).bisection(
            rho_lo=float(rholst.min()), rho_hi=float(rholst.max()),
            settings=settings2, tol=0.05).rho)[0])

        # Bisection samples continuously and grid only at its listed radii, so they need
        # not land on the identical number; they must land in the same regime -- both
        # solvable, and within the grid's own spacing of each other.
        self.assertLess(abs(np.log(grid_rho) - np.log(bis_rho)), np.log(2.0))


if __name__ == "__main__":
    unittest.main()
