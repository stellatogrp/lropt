import numpy as np
from cvxpy import Constraint, Parameter
from cvxpylayers.torch import CvxpyLayer
from torch import Tensor

from cvxro.parameter import ContextParameter, ShapeParameter
from cvxro.violation_checker.settings import VIOLATION_TOL
from cvxro.violation_checker.utils import (
    CONSTRAINT_STATUS,
    assert_batch_dim,
    check_constraint,
    get_batch_size,
    set_cp_parameters,
    set_rho_mult_parameter,
    set_shape_parameters,
    set_x_parameter,
)


class ViolationChecker():
    """
    A constraint checker class.

    Args:
        cvxpylayer (CvxpyLayer):
            The layer whose variables the checked decisions belong to.
        constraints (list[Constraint]):
            The (non-uncertain) constraints to check.
        tol (float):
            Absolute violation tolerance per constraint entry (see
            utils.check_constraint). Default VIOLATION_TOL.
    """
    def __init__(self, cvxpylayer: CvxpyLayer, constraints: list[Constraint],
                 tol: float = VIOLATION_TOL):
        self._cvxpylayer = cvxpylayer
        self._constraints = constraints
        self._tol = float(tol)
        self.last_sample_feasible = None

    def _set_z_batch(self, z_batch: list[Tensor], batch_number: int) -> bool:
        """
        Assign one sample's decision to the CVXPYLayer's variables.

        Returns False when the decision violates a variable's own DOMAIN (nonneg,
        nonpos, PSD, ...) by more than `tol`: such a sample is infeasible before any
        constraint is evaluated, and the caller records it as one. Values inside the
        tolerance are projected onto the domain first.

        Why this is not simply `var.value = val`: cvxpy validates domain attributes
        EXACTLY at assignment, and an interior-point solver returns a nonneg variable
        at +-1e-13 around zero (measured on lropt_power/multistage: minima of
        1e-13..1e-10 across seven nonneg variables at every radius probed). One sign
        flip at that scale, anywhere in a 48 x 4 rollout, raised
        `ValueError: Variable value must be nonnegative` out of a radius search and
        killed a slurm job (2026-09-20, ms_mean_var 14213476) -- an outcome no
        tolerance elsewhere in this class would have permitted for a CONSTRAINT
        violated by the same 1e-13. The variable domains are not in
        `problem_no_unc.constraints`, so this assignment is the only place they are
        checked at all; hence it classifies rather than raises.
        """
        domain_ok = True
        for i, z_tch in enumerate(z_batch):
            var = self._cvxpylayer.variables[i]
            batch_size = get_batch_size([var], [z_tch])
            z_tch = z_tch[batch_number] if batch_size > 1 else z_tch
            val = z_tch.reshape(var.shape).detach().numpy()
            if var.is_nonneg():
                if val.size and val.min() < -self._tol:
                    domain_ok = False
                val = np.maximum(val, 0.0)
            elif var.is_nonpos():
                if val.size and val.max() > self._tol:
                    domain_ok = False
                val = np.minimum(val, 0.0)
            try:
                var.value = val
            except ValueError:
                # A domain the projections above do not cover (PSD, symmetric,
                # boolean, ...). Not assignable is not feasible; the caller skips the
                # constraint loop for this sample, so no stale value is ever read.
                return False
        return domain_ok

    def _set_parameters(self, rho_mult_parameter: list[Parameter], rho_tch: Tensor,
                            cp_parameters: list[Parameter], cp_param_tch: list[Tensor],
                            x_parameters: list[ContextParameter], x_batch: list[Tensor],
                            shape_parameters: list[ShapeParameter], shape_torches: list[Tensor],
                            batch_number: int) -> None:
        """
        This function sets all the parameters.
        """
        set_rho_mult_parameter(rho_mult_parameter=rho_mult_parameter, rho_tch=rho_tch)
        set_cp_parameters(cp_parameters=cp_parameters, cp_param_tch=cp_param_tch,
                            batch_number=batch_number)
        set_x_parameter(x_parameters=x_parameters, x_batch=x_batch, batch_number=batch_number)
        set_shape_parameters(shape_parameters=shape_parameters, shape_torches=shape_torches,
                            batch_number=batch_number)

    def check_constraints(self, z_batch: list[Tensor], rho_mult_parameter: list[Parameter],
                                    rho_tch: Tensor, cp_parameters: list[Parameter],
                                    cp_param_tch: list[Tensor],
                                    x_parameters: list[ContextParameter],
                                    x_batch: list[Tensor], shape_parameters: list[ShapeParameter],
                                    shape_torches: list[Tensor],
                                    per_sample: bool = False) -> CONSTRAINT_STATUS:
        """
        This function checks if the constraints of self.problem_no_unc are violated or not for the
        given z_batch.
        z_batch should be given in the same order as self.problem_no_unc.variables
        (this is the default behavior)

        Args:
            z_batch (list[Tensor]):
                A list of z_batch to check (output of CVXPYLayer call).

            per_sample (bool):
                Whether to record which individual samples are feasible, in
                self.last_sample_feasible.

        Returns:
            status (CONSTRAINT_STATTUS):
                CONSTRAINT_STATUS.INFEASIBLE if any constraint of problem_no_unc is violated,
                else CONSTRAINT_STATUS.FEASIBLE.
        """
        batch_size = get_batch_size(in_cp=x_parameters, in_tch=x_batch)
        # The loop below walks batch_size samples and uses the same counter to index
        # every batched argument, so they all have to be the same length.
        if z_batch:
            assert_batch_dim(batch_size, self._cvxpylayer.variables, z_batch, "z_batch")
        assert_batch_dim(batch_size, cp_parameters, cp_param_tch, "cp_param_tch")
        assert_batch_dim(batch_size, x_parameters, x_batch, "x_batch")
        assert_batch_dim(batch_size, shape_parameters, shape_torches, "shape_torches")
        #eps_tch is never batched
        set_rho_mult_parameter(rho_mult_parameter, rho_tch)
        sample_feasible = np.ones(batch_size, dtype=bool) if per_sample else None
        for batch_number in range(batch_size):
            # z_batch, cp_parameters, shape_parameters may be batched,
            # so they need to be in the loop.
            # x_parameter is always batched.
            domain_ok = self._set_z_batch(z_batch, batch_number)
            set_cp_parameters(cp_parameters, cp_param_tch, batch_number)
            set_x_parameter(x_parameters, x_batch, batch_number)
            set_shape_parameters(shape_parameters, shape_torches, batch_number)
            if not domain_ok:
                # The decision left a variable's own domain by more than tol: this
                # sample is infeasible whatever the constraints say (see _set_z_batch).
                if not per_sample:
                    return CONSTRAINT_STATUS.INFEASIBLE
                sample_feasible[batch_number] = False
                continue
            for constraint in self._constraints:
                if check_constraint(constraint, self._tol)==CONSTRAINT_STATUS.INFEASIBLE:
                    if not per_sample:
                        return CONSTRAINT_STATUS.INFEASIBLE
                    sample_feasible[batch_number] = False
                    break
        self.last_sample_feasible = sample_feasible
        if per_sample and not sample_feasible.all():
            return CONSTRAINT_STATUS.INFEASIBLE
        return CONSTRAINT_STATUS.FEASIBLE
