"""
Scaling analysis of the CVXRO canonicalization pipeline.

Measures how solve() time scales with:
- Problem dimension n
- Number of constraints m
- Number of uncertain parameters

Outputs tables and optionally generates matplotlib plots.
"""

import time

import cvxpy as cp
import numpy as np

import cvxro


def make_problem(n, num_constraints, num_uncertain=1, set_type="ellipsoidal"):
    """Create a robust optimization problem."""
    x = cp.Variable(n)

    if set_type == "ellipsoidal":
        sets = [cvxro.Ellipsoidal(rho=2.0) for _ in range(num_uncertain)]
    elif set_type == "box":
        sets = [cvxro.Box(rho=1.5) for _ in range(num_uncertain)]
    else:
        sets = [cvxro.Ellipsoidal(rho=2.0) for _ in range(num_uncertain)]

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


def time_solve(problem_factory, num_runs=3, warmup=True):
    """Time the solve() method over multiple runs."""
    if warmup:
        prob = problem_factory()
        prob.solve(solver="CLARABEL")
        del prob

    times = []
    for _ in range(num_runs):
        prob = problem_factory()
        start = time.perf_counter()
        prob.solve(solver="CLARABEL")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        del prob

    return np.mean(times), np.std(times), np.min(times)


def scaling_dimension():
    """Measure scaling with problem dimension n."""
    print("\n" + "=" * 70)
    print("SCALING WITH DIMENSION (n)")
    print("Fixed: 5 constraints, 1 uncertain param (Ellipsoidal)")
    print("=" * 70)

    dims = [5, 10, 20, 50, 100, 200]
    results = []

    print(f"\n{'n':>6} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'Ratio':>8}")
    print("-" * 50)

    prev_mean = None
    for n in dims:
        np.random.seed(42)

        def factory(n=n):
            return make_problem(n, num_constraints=5, num_uncertain=1)

        mean, std, mn = time_solve(factory)
        ratio = mean / prev_mean if prev_mean else 1.0
        print(f"{n:>6} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {ratio:>8.2f}x")
        results.append((n, mean, std, mn))
        prev_mean = mean

    return dims, results


def scaling_constraints():
    """Measure scaling with number of constraints m."""
    print("\n" + "=" * 70)
    print("SCALING WITH CONSTRAINTS (m)")
    print("Fixed: n=20, 1 uncertain param (Ellipsoidal)")
    print("=" * 70)

    constraints = [2, 5, 10, 20, 50]
    results = []

    print(f"\n{'m':>6} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'Ratio':>8}")
    print("-" * 50)

    prev_mean = None
    for m in constraints:
        np.random.seed(42)

        def factory(m=m):
            return make_problem(20, num_constraints=m, num_uncertain=1)

        mean, std, mn = time_solve(factory)
        ratio = mean / prev_mean if prev_mean else 1.0
        print(f"{m:>6} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {ratio:>8.2f}x")
        results.append((m, mean, std, mn))
        prev_mean = mean

    return constraints, results


def scaling_uncertain_params():
    """Measure scaling with number of uncertain parameters."""
    print("\n" + "=" * 70)
    print("SCALING WITH UNCERTAIN PARAMETERS")
    print("Fixed: n=20, 5 constraints (Ellipsoidal)")
    print("=" * 70)

    num_params = [1, 2, 3]
    results = []

    print(f"\n{'#params':>8} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'Ratio':>8}")
    print("-" * 50)

    prev_mean = None
    for p in num_params:
        np.random.seed(42)

        def factory(p=p):
            return make_problem(20, num_constraints=5, num_uncertain=p)

        mean, std, mn = time_solve(factory)
        ratio = mean / prev_mean if prev_mean else 1.0
        print(f"{p:>8} {mean:>10.4f} {std:>10.4f} {mn:>10.4f} {ratio:>8.2f}x")
        results.append((p, mean, std, mn))
        prev_mean = mean

    return num_params, results


def try_plot(dim_data, cons_data, param_data):
    """Attempt to generate matplotlib plots. Skips silently if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not available, skipping plots.")
        return

    import os
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Dimension scaling
    dims, dim_results = dim_data
    means = [r[1] for r in dim_results]
    stds = [r[2] for r in dim_results]
    axes[0].errorbar(dims, means, yerr=stds, marker="o", capsize=3)
    axes[0].set_xlabel("Problem dimension (n)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Scaling with dimension")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    # Constraint scaling
    constraints, cons_results = cons_data
    means = [r[1] for r in cons_results]
    stds = [r[2] for r in cons_results]
    axes[1].errorbar(constraints, means, yerr=stds, marker="s", capsize=3)
    axes[1].set_xlabel("Number of constraints (m)")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Scaling with constraints")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    # Uncertain parameter scaling
    params, param_results = param_data
    means = [r[1] for r in param_results]
    stds = [r[2] for r in param_results]
    axes[2].errorbar(params, means, yerr=stds, marker="^", capsize=3)
    axes[2].set_xlabel("Number of uncertain params")
    axes[2].set_ylabel("Time (s)")
    axes[2].set_title("Scaling with uncertain params")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "scaling_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlots saved to {plot_path}")


def main():
    dim_data = scaling_dimension()
    cons_data = scaling_constraints()
    param_data = scaling_uncertain_params()
    try_plot(dim_data, cons_data, param_data)


if __name__ == "__main__":
    main()
