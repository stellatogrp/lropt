"""settings.batch_per_step: a fresh training batch at every step.

Without it the training batch is seeded by `seed_num`, which moves only at AL updates
(aug_lag_update_interval, default 20), so every step inside an interval differentiates
the SAME batch. These tests record the seed each training rollout asks for on the tiny
portfolio problem from test_al_strategies, in both line-search modes.
"""

import unittest

from .test_al_strategies import _base_settings, _make_portfolio_problem


def _train_seeds(mode, batch_per_step, num_iter=6, interval=20):
    prob, trainer, data = _make_portfolio_problem()
    settings = _base_settings(data)
    settings.num_iter = num_iter
    settings.aug_lag_update_interval = interval
    settings.line_search_mode = mode
    settings.batch_per_step = batch_per_step
    seen = []
    orig = trainer.loss_and_constraints

    def spy(*args, **kwargs):
        if not trainer._validate_flag and not trainer._test_flag:
            seen.append(kwargs["seed"])
        return orig(*args, **kwargs)

    trainer.loss_and_constraints = spy
    trainer.train(settings=settings)
    return settings.seed, seen


class TestBatchPerStep(unittest.TestCase):

    def test_default_keeps_one_batch_per_al_interval(self):
        # interval 20 > num_iter: step 0 uses seed_num 0, every later step seed_num 1.
        for mode in ("legacy", "backtrack"):
            seed, seen = _train_seeds(mode, batch_per_step=False)
            self.assertEqual(seen[0], seed + 1, mode)
            self.assertEqual(set(seen[1:]), {seed + 2}, mode)

    def test_batch_per_step_draws_a_new_batch_every_step(self):
        for mode in ("legacy", "backtrack"):
            seed, seen = _train_seeds(mode, batch_per_step=True)
            # one seed per step, all distinct; repeats within a step (line-search tries)
            # reuse the step's own batch
            distinct = list(dict.fromkeys(seen))
            self.assertEqual(len(distinct), 6, (mode, seen))
            self.assertEqual(distinct, [seed + 20011 + k + 1 for k in range(6)], mode)


if __name__ == "__main__":
    unittest.main()
