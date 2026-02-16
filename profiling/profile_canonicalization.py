"""
Function-level cProfile of the CVXRO solve() pipeline.

Tests multiple problem configurations and outputs sorted timing tables.
Saves .prof files for further analysis with pstats/snakeviz.
"""

import cProfile
import os
import pstats
import sys
import time

import cvxpy as cp
import numpy as np

import cvxro

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def make_problem_simple(n, num_constraints, uncertainty_set):
    """Create a simple robust optimization problem."""
    x = cp.Variable(n)
    u = cvxro.UncertainParameter(n, uncertainty_set=uncertainty_set)

    objective = cp.Minimize(cp.sum(x))
    constraints = [x >= -10, x <= 10]
    for i in range(num_constraints):
        coeff = np.random.randn(n)
        constraints.append(coeff @ x + u @ np.ones(n) <= 5 + i)

    return cvxro.RobustProblem(objective, constraints)


def make_problem_multi_unc(n, num_constraints, sets):
    """Create a problem with multiple uncertain parameters."""
    x = cp.Variable(n)
    u_params = [cvxro.UncertainParameter(n, uncertainty_set=s) for s in sets]

    objective = cp.Minimize(cp.sum(x))
    constraints = [x >= -10, x <= 10]
    for i in range(num_constraints):
        coeff = np.random.randn(n)
        expr = coeff @ x
        for u in u_params:
            expr = expr + u @ np.ones(n)
        constraints.append(expr <= 5 + i)

    return cvxro.RobustProblem(objective, constraints)


def profile_config(name, problem_factory, num_runs=3):
    """Profile a problem configuration and print results."""
    print(f"\n{'='*70}")
    print(f"Configuration: {name}")
    print(f"{'='*70}")

    # Warmup run (first solve triggers lazy imports and solver initialization)
    prob = problem_factory()
    prob.solve(solver="CLARABEL")
    del prob

    # Timed runs
    times = []
    for i in range(num_runs):
        prob = problem_factory()
        start = time.perf_counter()
        prob.solve(solver="CLARABEL")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del prob

    print(f"Wall time: {np.mean(times):.4f}s +/- {np.std(times):.4f}s "
          f"(min={min(times):.4f}s, max={max(times):.4f}s)")

    # cProfile run
    prob = problem_factory()
    profiler = cProfile.Profile()
    profiler.enable()
    prob.solve(solver="CLARABEL")
    profiler.disable()

    # Print top functions by cumulative time
    print("\n--- Top 30 by cumulative time ---")
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print("\n--- Top 30 by total time ---")
    stats.sort_stats("tottime")
    stats.print_stats(30)

    # Save .prof file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = name.replace(" ", "_").replace(",", "").replace("=", "")
    prof_path = os.path.join(OUTPUT_DIR, f"{safe_name}.prof")
    profiler.dump_stats(prof_path)
    print(f"Saved profile to {prof_path}")

    return times


def main():
    np.random.seed(42)

    configs = [
        ("small_n5_m3_ellipsoidal", lambda: make_problem_simple(
            n=5, num_constraints=3, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))),

        ("medium_n50_m10_ellipsoidal", lambda: make_problem_simple(
            n=50, num_constraints=10, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))),

        ("large_n200_m20_ellipsoidal", lambda: make_problem_simple(
            n=200, num_constraints=20, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))),

        ("multi_unc_n20_ellipsoidal_box", lambda: make_problem_multi_unc(
            n=20, num_constraints=5,
            sets=[cvxro.Ellipsoidal(rho=2.0), cvxro.Box(rho=1.5)])),

        ("n20_m5_box", lambda: make_problem_simple(
            n=20, num_constraints=5, uncertainty_set=cvxro.Box(rho=1.5))),

        ("n20_m5_budget", lambda: make_problem_simple(
            n=20, num_constraints=5, uncertainty_set=cvxro.Budget(rho1=1.0, rho2=1.0))),

        ("n20_m5_polyhedral", lambda: make_problem_simple(
            n=20, num_constraints=5,
            uncertainty_set=cvxro.Polyhedral(
                lhs=np.eye(20), rhs=np.ones(20)))),
    ]

    all_results = {}
    for name, factory in configs:
        times = profile_config(name, factory)
        all_results[name] = times

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<40} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10}")
    print("-" * 70)
    for name, times in all_results.items():
        print(f"{name:<40} {np.mean(times):>10.4f} {np.std(times):>10.4f} {min(times):>10.4f}")


if __name__ == "__main__":
    main()
