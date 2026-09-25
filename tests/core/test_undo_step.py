import unittest

import torch

from cvxro.train.utils import take_step, undo_step


class TestUndoStep(unittest.TestCase):
    """undo_step must restore the pre-step parameters however many times the line
    search retakes the step. Assigning `param.data = snapshot` aliased the parameter
    to the snapshot, so the next optimizer step moved the snapshot as well and only
    the first undo ever restored anything."""

    def _setup(self):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(4, dtype=torch.double))
        opt = torch.optim.Adam([p], lr=0.1)
        p.grad = torch.ones(4, dtype=torch.double)
        rho = torch.tensor(1.0, dtype=torch.double)
        return p, opt, rho

    def test_repeated_undo_retake_returns_to_start(self):
        p, opt, rho = self._setup()
        start = p.detach().clone()
        prev = take_step(opt, rho, None)                       # the step under test
        for _ in range(5):                                     # line search retakes
            undo_step(opt, prev)
            self.assertTrue(torch.equal(p.detach(), start),
                            "undo must return to the pre-step parameters every time")
            prev = take_step(opt, rho, None, update_state=False, prev_states=prev)
        undo_step(opt, prev)
        self.assertTrue(torch.equal(p.detach(), start))

    def test_snapshot_is_not_aliased(self):
        p, opt, rho = self._setup()
        prev = take_step(opt, rho, None)
        undo_step(opt, prev)
        self.assertFalse(p.data.data_ptr() == prev[0].data_ptr(),
                         "parameter must not share storage with its snapshot")


if __name__ == "__main__":
    unittest.main()
