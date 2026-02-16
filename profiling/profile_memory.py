"""
Memory profiling of the CVXRO canonicalization pipeline using tracemalloc.

Measures peak memory and per-stage allocations across problem sizes.
"""

import tracemalloc

import cvxpy as cp
import numpy as np
from cvxpy.problems.objective import Maximize

import cvxro
from cvxro.uncertain_canon.remove_uncertain_maximum import RemoveSumOfMaxOfUncertain
from cvxro.uncertain_canon.remove_uncertainty import RemoveUncertainty
from cvxro.uncertain_canon.uncertain_canonicalization import UncertainCanonicalization
from cvxro.utils import gen_and_apply_chain


def make_problem(n, num_constraints=5):
    """Create a robust optimization problem of given size."""
    x = cp.Variable(n)
    u = cvxro.UncertainParameter(n, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))

    objective = cp.Minimize(cp.sum(x))
    constraints = [x >= -10, x <= 10]
    for i in range(num_constraints):
        coeff = np.random.randn(n)
        constraints.append(coeff @ x + u @ np.ones(n) <= 5 + i)

    return cvxro.RobustProblem(objective, constraints)


def format_bytes(size):
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def top_allocators(snapshot, limit=10):
    """Print top memory allocators from a tracemalloc snapshot."""
    stats = snapshot.statistics("lineno")
    print(f"  Top {limit} allocators:")
    for stat in stats[:limit]:
        print(f"    {stat}")


def profile_stages(n, num_constraints=5):
    """Profile memory usage per stage of the canonicalization pipeline."""
    print(f"\n{'='*70}")
    print(f"Memory profile: n={n}, constraints={num_constraints}")
    print(f"{'='*70}")

    np.random.seed(42)
    prob = make_problem(n, num_constraints)
    prob._solver = "CLARABEL"

    tracemalloc.start()

    # Stage 0: baseline
    snap_baseline = tracemalloc.take_snapshot()
    mem_baseline = tracemalloc.get_traced_memory()

    # Stage 1a: RemoveSumOfMaxOfUncertain + UncertainCanonicalization
    from cvxro.uncertain_canon.flip_objective import FlipObjective

    reductions_canon = []
    if isinstance(prob.objective, Maximize):
        reductions_canon += [FlipObjective()]
    reductions_canon += [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
    chain_canon, problem_canon, inverse_data_canon = gen_and_apply_chain(
        problem=prob, reductions=reductions_canon
    )

    snap_after_canon = tracemalloc.take_snapshot()
    mem_after_canon = tracemalloc.get_traced_memory()

    # Stage 2: generate_torch_expressions
    from cvxro.torch_expression_generator import generate_torch_expressions

    generate_torch_expressions(problem_canon)

    snap_after_torch = tracemalloc.take_snapshot()
    mem_after_torch = tracemalloc.get_traced_memory()

    # Stage 3: RemoveUncertainty
    chain_no_unc, problem_no_unc, inverse_data_no_unc = gen_and_apply_chain(
        problem_canon, reductions=[RemoveUncertainty()]
    )

    snap_after_remove = tracemalloc.take_snapshot()
    mem_after_remove = tracemalloc.get_traced_memory()

    # Stage 4: Solve
    for x in problem_no_unc.parameters():
        if x.value is None:
            x.value = x.data[0] if hasattr(x, "data") and x.data is not None else np.zeros(x.shape)
    problem_no_unc.solve(solver="CLARABEL")

    snap_after_solve = tracemalloc.take_snapshot()
    mem_after_solve = tracemalloc.get_traced_memory()

    peak_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Report
    stages = [
        ("Baseline", mem_baseline),
        ("After canonicalization (1a+1b)", mem_after_canon),
        ("After generate_torch_expressions", mem_after_torch),
        ("After RemoveUncertainty", mem_after_remove),
        ("After solve", mem_after_solve),
    ]

    print(f"\n  {'Stage':<45} {'Current':>12} {'Peak':>12}")
    print(f"  {'-'*69}")
    for name, (current, peak) in stages:
        print(f"  {name:<45} {format_bytes(current):>12} {format_bytes(peak):>12}")

    print(f"\n  Overall peak memory: {format_bytes(peak_mem)}")

    # Deltas
    print("\n  Memory deltas between stages:")
    snapshots = [
        ("Canonicalization", snap_baseline, snap_after_canon),
        ("generate_torch_expressions", snap_after_canon, snap_after_torch),
        ("RemoveUncertainty", snap_after_torch, snap_after_remove),
        ("Solve", snap_after_remove, snap_after_solve),
    ]
    for name, before, after in snapshots:
        delta_stats = after.compare_to(before, "lineno")
        total_delta = sum(s.size_diff for s in delta_stats if s.size_diff > 0)
        print(f"\n  --- {name}: +{format_bytes(total_delta)} ---")
        for stat in delta_stats[:5]:
            if stat.size_diff > 0:
                print(f"    {stat}")

    return peak_mem


def main():
    sizes = [5, 50, 200]

    print("CVXRO Memory Profiling")
    print("=" * 70)

    results = {}
    for n in sizes:
        peak = profile_stages(n, num_constraints=max(3, n // 10))
        results[n] = peak

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Peak memory by problem size")
    print(f"{'='*70}")
    print(f"{'n':<10} {'Constraints':<15} {'Peak Memory':>15}")
    print("-" * 40)
    for n in sizes:
        m = max(3, n // 10)
        print(f"{n:<10} {m:<15} {format_bytes(results[n]):>15}")


if __name__ == "__main__":
    main()
