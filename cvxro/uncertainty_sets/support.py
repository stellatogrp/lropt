"""The uncertainty set's SUPPORT, and projection of a center into it.

A set declared with `lb`, `ub`, `sum_eq`, or an explicit `c`/`d` is the ellipsoid (or
norm ball) INTERSECTED with the polyhedron `{u : C u <= d}`. The robust counterpart
handles that intersection exactly -- verified against direct maximization for box, vector
bound, general polyhedral and equality supports alike -- so nothing here touches the
reformulation.

What the counterpart cannot do is rescue an EMPTY intersection. If the set's center lies
outside the support and the radius is too small to reach back in, the intersection is
empty, the robust constraint quantifies over nothing, and so it bounds nothing: the
problem becomes unbounded in the directions that constraint was meant to limit, and the
solver returns an enormous decision while reporting `optimal`. No violation metric catches
it, because the decision really is feasible for a constraint that says nothing.

The fix is one condition: keep the center IN the support. Then the center itself lies in
both the ellipsoid (at z = 0) and the support, so the intersection is nonempty by
construction -- for any support, not merely a box. This module provides the two pieces
that need: reading the support off a set, and projecting a center into it.

    support_matrices(uncset)      -> (C, d), or None when no support is declared
    project_into_support(b, C, d) -> b projected onto {u : C u <= d}

The projection is exact and closed-form when the support is a box (the common case:
`lb`/`ub`), and otherwise runs Dykstra's alternating projection over the halfspaces, which
converges to the true projection for a polyhedron. Both paths are pure torch, batched, and
differentiable, so they can sit inside the training loop.
"""

import numpy as np
import torch

# Rows whose coefficients are all (numerically) zero constrain nothing. They appear for a
# real reason: `Norm.conjugate` lazily fills `_c`/`_d` with zeros when no support was
# declared, so an all-zero `C` is the library's own "no support" sentinel and must not be
# mistaken for a support that happens to be vacuous.
_ZERO_ROW_ATOL = 1e-12


def support_matrices(uncset):
    """The support of `uncset` as `(C, d)` with `C u <= d`, or None if it declares none.

    Reads the ASSEMBLED `_c`/`_d`, which is where `add_support_type` folds `lb`, `ub` and
    `sum_eq` (an equality becomes the two inequalities `c u <= d` and `-c u <= -d`), and
    where an explicitly passed `c`/`d` already lives. Reading the assembled form rather
    than the individual keywords is what makes this agree with the counterpart: the
    conjugate consumes exactly these matrices.
    """
    c = getattr(uncset, "_c", None)
    d = getattr(uncset, "_d", None)
    if c is None or d is None:
        return None
    rows = np.atleast_2d(np.asarray(c, dtype=float))
    rhs = np.atleast_1d(np.asarray(d, dtype=float)).ravel()
    if rows.shape[0] != rhs.shape[0]:
        raise ValueError(
            f"support has {rows.shape[0]} rows in c but {rhs.shape[0]} entries in d")
    keep = ~np.all(np.abs(rows) <= _ZERO_ROW_ATOL, axis=1)
    if not keep.any():
        return None
    return rows[keep], rhs[keep]


def box_bounds(rows, rhs, dim):
    """`(lo, hi)` if `{C u <= d}` is a box, else None.

    A row is a bound when it has exactly one nonzero coefficient. If every row is, the
    polyhedron is a box and the projection onto it is a clamp -- exact, one operation, and
    with a gradient torch already knows. Rows are combined by taking the TIGHTEST bound per
    coordinate, so duplicated or redundant bounds behave.
    """
    lo = np.full(dim, -np.inf)
    hi = np.full(dim, np.inf)
    for row, bound in zip(rows, rhs):
        nz = np.flatnonzero(np.abs(row) > _ZERO_ROW_ATOL)
        if len(nz) != 1:
            return None
        j = int(nz[0])
        coef = row[j]
        if coef > 0:                      # coef * u_j <= bound  ->  u_j <= bound / coef
            hi[j] = min(hi[j], bound / coef)
        else:                             # coef < 0             ->  u_j >= bound / coef
            lo[j] = max(lo[j], bound / coef)
    return lo, hi


def _clamp(b_tch, lo, hi):
    """Projection onto a box, honouring infinite bounds by leaving that side alone."""
    out = b_tch
    if np.any(np.isfinite(lo)):
        lo_t = torch.as_tensor(np.where(np.isfinite(lo), lo, -np.inf), dtype=b_tch.dtype)
        out = torch.maximum(out, lo_t)
    if np.any(np.isfinite(hi)):
        hi_t = torch.as_tensor(np.where(np.isfinite(hi), hi, np.inf), dtype=b_tch.dtype)
        out = torch.minimum(out, hi_t)
    return out


def _dykstra(b_tch, rows, rhs, max_iter, tol):
    """Projection onto `{u : C u <= d}` by Dykstra's alternating projections.

    Cycling the projections alone (POCS) converges to SOME point of the intersection, not
    to the nearest one; Dykstra's correction terms are what make the limit the actual
    projection, which matters here because the center carries meaning -- it is the
    predicted conditional mean, and we want the smallest change to it that restores a
    nonempty set.

    Each halfspace projection is closed-form,
        P_i(y) = y - max(0, (c_i'y - d_i) / ||c_i||^2) c_i,
    so the whole loop is differentiable and runs on batched `b` without a solver.
    """
    c_t = torch.as_tensor(rows, dtype=b_tch.dtype)
    d_t = torch.as_tensor(rhs, dtype=b_tch.dtype)
    sq_norms = torch.clamp((c_t * c_t).sum(dim=1), min=_ZERO_ROW_ATOL)

    # Already inside: nothing to project, and this is the common case once training has
    # settled, so it is worth one matmul to skip the whole loop.
    if float(_violation(b_tch, c_t, d_t)) <= tol:
        return b_tch, 0.0

    x = b_tch
    # One correction term per halfspace, shaped like the (possibly batched) center.
    corrections = [torch.zeros_like(b_tch) for _ in range(c_t.shape[0])]
    for _ in range(max_iter):
        start = x
        for i in range(c_t.shape[0]):
            y = x + corrections[i]
            excess = (y * c_t[i]).sum(dim=-1, keepdim=True) - d_t[i]
            x = y - torch.clamp(excess, min=0.0) / sq_norms[i] * c_t[i]
            corrections[i] = y - x
        # Stop on CONVERGENCE of the sweep, not on feasibility. Dykstra's iterates become
        # feasible well before they reach the nearest point, so stopping at the first
        # feasible iterate returns some point of the polyhedron rather than the projection
        # -- measured 0.6 away from it on a rotated three-row example, while reporting a
        # residual of 1e-16. Feasibility is still required, since an early sweep can also
        # be nearly stationary while outside.
        moved = float((x - start).abs().max().detach())
        if moved <= tol and float(_violation(x, c_t, d_t)) <= tol:
            break
    return x, float(_violation(x, c_t, d_t))


def _violation(x, c_t, d_t):
    """The largest amount by which `x` breaks any support row (0 when inside)."""
    slack = x @ c_t.T - d_t
    return torch.clamp(slack.max(), min=0.0).detach()


def project_into_support(b_tch, rows, rhs, max_iter=100, tol=1e-9):
    """Project a center -- batched `(N, m)` or single `(m,)` -- into `{u : C u <= d}`.

    Returns `(projected, residual)`, where `residual` is the largest remaining violation.
    A residual above `tol` means Dykstra did not converge in `max_iter` sweeps, which for a
    nonempty polyhedron means the iteration budget was too small and for an EMPTY one
    (contradictory rows, e.g. `lb > ub`) means no projection exists at all. The caller
    decides which, since only it knows whether to raise or to warn.
    """
    dim = b_tch.shape[-1]
    box = box_bounds(rows, rhs, dim)
    if box is not None:
        lo, hi = box
        if np.any(lo > hi + 1e-12):
            return b_tch, float("inf")      # empty box: no projection exists
        # No residual to compute: clamping onto a NONEMPTY box is the exact projection, so
        # the result satisfies every row by construction. The emptiness check above is what
        # makes that true, and skipping the check here keeps the common path free of a
        # matmul that runs on every forward pass of training.
        return _clamp(b_tch, lo, hi), 0.0
    return _dykstra(b_tch, rows, rhs, max_iter=max_iter, tol=tol)


def support_is_empty(rows, rhs):
    """Whether `{u : C u <= d}` has no points at all, by a small LP.

    Only called once, and only when the projection fails to converge: it separates "the
    support is contradictory" (a modelling error worth raising on) from "Dykstra needed
    more sweeps", which the residual alone cannot distinguish.
    """
    import cvxpy as cp

    u = cp.Variable(rows.shape[1])
    prob = cp.Problem(cp.Minimize(0), [rows @ u <= rhs])
    try:
        prob.solve()
    except Exception:                                             # noqa: BLE001
        return False
    return prob.status in ("infeasible", "infeasible_inaccurate")
