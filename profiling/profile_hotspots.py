"""
Targeted micro-benchmarks for suspected bottlenecks in the CVXRO pipeline.

Isolates and benchmarks specific operations:
1. reshape_tensor() — lil_matrix row-by-row copy loop
2. get_problem_data() — CVXPY internal canonicalization
3. _gen_constraints() loop — CVXPY expression construction
4. generate_torch_expressions() — overhead vs rest of pipeline
5. Variable creation in remove_uncertain_terms()
6. has_unc_param() repeated calls
"""

import statistics
import time

import cvxpy as cp
import numpy as np
from cvxpy.problems.objective import Maximize
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse import random as sp_random

import cvxro
from cvxro.uncertain_canon.remove_uncertain_maximum import RemoveSumOfMaxOfUncertain
from cvxro.uncertain_canon.remove_uncertainty import RemoveUncertainty
from cvxro.uncertain_canon.uncertain_canonicalization import UncertainCanonicalization
from cvxro.uncertain_canon.utils import reshape_tensor
from cvxro.utils import gen_and_apply_chain, has_unc_param


def make_problem(n, num_constraints=5):
    """Create a robust optimization problem."""
    x = cp.Variable(n)
    u = cvxro.UncertainParameter(n, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))

    objective = cp.Minimize(cp.sum(x))
    constraints = [x >= -10, x <= 10]
    for i in range(num_constraints):
        coeff = np.random.randn(n)
        constraints.append(coeff @ x + u @ np.ones(n) <= 5 + i)

    return cvxro.RobustProblem(objective, constraints)


def benchmark(func, num_runs=10, label=""):
    """Run a function multiple times and report timing statistics."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    median = statistics.median(times)
    print(f"  {label:<55} mean={mean:.6f}s  std={std:.6f}s  "
          f"median={median:.6f}s  (n={num_runs})")
    return mean


def benchmark_reshape_tensor():
    """Benchmark 1: reshape_tensor() — optimized vs old approach.

    The old lil_matrix row-by-row loop is kept here as a reference baseline
    to verify the optimization remains effective and to detect regressions.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 1: reshape_tensor()")
    print("Compares optimized (current) vs old lil_matrix loop (reference)")
    print("=" * 70)

    def reshape_tensor_old(T_Ab, n_var):
        """Old approach: lil_matrix row-by-row copy (pre-optimization baseline)."""
        T_Ab = csc_matrix(T_Ab)
        n_var_full = n_var + 1
        num_rows = T_Ab.shape[0]
        num_constraints = num_rows // n_var_full
        T_Ab_res = lil_matrix(T_Ab.shape)
        for target_row in range(num_rows):
            constraint_num = 0 if n_var_full == 0 else target_row % n_var_full
            var_num = target_row if n_var_full == 0 else target_row // n_var_full
            source_row = constraint_num * num_constraints + var_num
            T_Ab_res[target_row, :] = T_Ab[source_row, :]
        return T_Ab_res

    for size_label, n_var, density in [
        ("small (n=5)", 5, 0.3),
        ("medium (n=50)", 50, 0.1),
        ("large (n=200)", 200, 0.05),
    ]:
        print(f"\n  --- {size_label} ---")
        n_var_full = n_var + 1
        n_constraints = max(3, n_var // 10)
        num_rows = n_var_full * n_constraints
        num_cols = n_var * 2  # arbitrary parameter count

        T_Ab = sp_random(num_rows, num_cols, density=density, format="coo")

        benchmark(
            lambda T=T_Ab, nv=n_var: reshape_tensor_old(T, nv),
            num_runs=50, label="Old (lil_matrix loop)")

        benchmark(
            lambda T=T_Ab, nv=n_var: reshape_tensor(T, nv),
            num_runs=50, label="Current (permutation index)")


def benchmark_get_problem_data():
    """Benchmark 2: get_problem_data() — CVXPY internal canonicalization."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: get_problem_data() — CVXPY internal cost")
    print("Fraction of total canonicalization time spent in CVXPY internals")
    print("=" * 70)

    for n, m in [(5, 3), (50, 10), (200, 20)]:
        print(f"\n  --- n={n}, m={m} ---")
        np.random.seed(42)

        # Time get_problem_data alone
        def time_get_problem_data():
            prob = make_problem(n, m)
            # Apply the reductions up to canonicalization
            reductions = [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            chain, problem_canon, inv_data = gen_and_apply_chain(prob, reductions)
            # Now time get_problem_data on the canonical problem
            # We actually need a fresh problem to call get_problem_data
            return prob

        # Create the epigraph problem that UncertainCanonicalization uses internally
        prob = make_problem(n, m)
        epigraph_obj_var = cp.Variable()
        epi_cons = prob.objective.expr <= epigraph_obj_var
        new_constraints = [epi_cons] + prob.constraints
        epigraph_problem = cvxro.RobustProblem(
            cp.Minimize(epigraph_obj_var), new_constraints
        )

        # Time just get_problem_data
        benchmark(
            lambda: epigraph_problem.get_problem_data(solver="CLARABEL"),
            num_runs=20, label="get_problem_data() alone")

        # Time full canonicalization (for comparison)
        def full_canon():
            p = make_problem(n, m)
            p._solver = "CLARABEL"
            reductions = [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            gen_and_apply_chain(p, reductions)

        benchmark(
            full_canon, num_runs=20,
            label="Full canonicalization (includes get_problem_data)")


def benchmark_generate_torch_expressions():
    """Benchmark 4: generate_torch_expressions() cost vs rest of pipeline."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: generate_torch_expressions() overhead")
    print("Compares remove_uncertainty with and without torch expression generation")
    print("=" * 70)

    for n, m in [(5, 3), (20, 5), (50, 10), (200, 20)]:
        print(f"\n  --- n={n}, m={m} ---")
        np.random.seed(42)

        from cvxro.torch_expression_generator import generate_torch_expressions
        from cvxro.uncertain_canon.flip_objective import FlipObjective

        def full_remove_uncertainty():
            prob = make_problem(n, m)
            prob.remove_uncertainty(solver="CLARABEL")

        def without_torch_expressions():
            """remove_uncertainty minus generate_torch_expressions."""
            prob = make_problem(n, m)
            prob._solver = "CLARABEL"
            reductions_canon = []
            if isinstance(prob.objective, Maximize):
                reductions_canon += [FlipObjective()]
            reductions_canon += [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            chain_canon, problem_canon, inv_data = gen_and_apply_chain(prob, reductions_canon)
            # Skip generate_torch_expressions!
            chain_no_unc, problem_no_unc, inv_data_no_unc = gen_and_apply_chain(
                problem_canon, reductions=[RemoveUncertainty()]
            )

        def only_torch_expressions():
            """Just generate_torch_expressions on an already-canonicalized problem."""
            prob = make_problem(n, m)
            prob._solver = "CLARABEL"
            reductions_canon = [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            _, problem_canon, _ = gen_and_apply_chain(prob, reductions_canon)
            generate_torch_expressions(problem_canon)

        t_full = benchmark(full_remove_uncertainty, num_runs=10,
                           label="Full remove_uncertainty()")
        t_without = benchmark(without_torch_expressions, num_runs=10,
                              label="Without generate_torch_expressions()")
        t_torch_only = benchmark(only_torch_expressions, num_runs=10,
                                 label="Only generate_torch_expressions()")

        if t_full > 0:
            pct = (t_torch_only / t_full) * 100
            print(f"  --> generate_torch_expressions is ~{pct:.1f}% of total "
                  f"remove_uncertainty time")
            print(f"  --> Potential speedup from deferral: {t_full / t_without:.2f}x")


def benchmark_variable_creation():
    """Benchmark 5: Variable creation in remove_uncertain_terms()."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Variable(shape) creation cost")
    print("Measures the cost of creating CVXPY Variable objects")
    print("=" * 70)

    for shape in [1, 5, 20, 50, 200]:
        def create_vars(shape=shape, k_num=3, u_num=2):
            for _ in range(u_num):
                for _ in range(k_num):
                    cp.Variable(shape)
                    cp.Variable(shape)  # supp_cons

        benchmark(create_vars, num_runs=100,
                  label=f"shape={shape}, k_num=3, u_num=2 ({12} vars)")


def benchmark_has_unc_param():
    """Benchmark 6: has_unc_param() repeated calls."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: has_unc_param() — parameter iteration cost")
    print("Measures cost of calling has_unc_param on the same expressions")
    print("=" * 70)

    for n, m in [(5, 3), (20, 10), (50, 20)]:
        np.random.seed(42)
        prob = make_problem(n, m)

        constraints = prob.constraints

        def call_has_unc_param(constraints=constraints):
            for c in constraints:
                has_unc_param(c)

        benchmark(call_has_unc_param, num_runs=100,
                  label=f"n={n}, m={m}: {len(constraints)} constraints")

        # Also measure parameters() call separately
        expr = constraints[-1]  # An uncertain constraint

        def call_parameters(expr=expr):
            expr.parameters()

        benchmark(call_parameters, num_runs=100,
                  label="  -> expr.parameters() on one constraint")


def benchmark_full_solve_breakdown():
    """Comprehensive breakdown: what fraction of solve() is each stage."""
    print("\n" + "=" * 70)
    print("FULL SOLVE BREAKDOWN")
    print("Time per stage as a fraction of total solve()")
    print("=" * 70)

    for n, m in [(5, 3), (50, 10), (200, 20)]:
        print(f"\n  --- n={n}, m={m} ---")
        np.random.seed(42)

        from cvxro.torch_expression_generator import generate_torch_expressions

        # Total solve time
        def full_solve():
            prob = make_problem(n, m)
            prob.solve(solver="CLARABEL")

        t_total = benchmark(full_solve, num_runs=5, label="Total solve()")

        # Stage 1: RemoveSumOfMaxOfUncertain + UncertainCanonicalization
        def stage_canon():
            prob = make_problem(n, m)
            prob._solver = "CLARABEL"
            reductions = [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            gen_and_apply_chain(prob, reductions)

        t_canon = benchmark(stage_canon, num_runs=5, label="Stage 1: Canonicalization")

        # Stage 2: generate_torch_expressions
        prob = make_problem(n, m)
        prob._solver = "CLARABEL"
        reductions = [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
        _, problem_canon, _ = gen_and_apply_chain(prob, reductions)

        def stage_torch(pc=problem_canon):
            generate_torch_expressions(pc)

        # Note: can only run once since it modifies problem_canon in place
        start = time.perf_counter()
        generate_torch_expressions(problem_canon)
        t_torch = time.perf_counter() - start
        print(f"  {'Stage 2: generate_torch_expressions':<55} "
              f"time={t_torch:.6f}s  (single run, modifies in-place)")

        # Stage 3: RemoveUncertainty
        def stage_remove():
            prob = make_problem(n, m)
            prob._solver = "CLARABEL"
            reductions = [RemoveSumOfMaxOfUncertain(), UncertainCanonicalization()]
            _, pc, _ = gen_and_apply_chain(prob, reductions)
            generate_torch_expressions(pc)
            gen_and_apply_chain(pc, reductions=[RemoveUncertainty()])

        # We want just the RemoveUncertainty step, but it needs the preceding stages.
        # So we measure the total and subtract.
        t_remove_total = benchmark(stage_remove, num_runs=5,
                                   label="Stages 1+2+3 (through RemoveUncertainty)")

        # Stage 4: Solve the deterministic problem
        prob = make_problem(n, m)
        prob.remove_uncertainty(solver="CLARABEL")

        def stage_solve():
            p = prob.problem_no_unc
            for x in p.parameters():
                if x.value is None:
                    x.value = np.zeros(x.shape)
            p.solve(solver="CLARABEL")

        t_solve = benchmark(stage_solve, num_runs=5, label="Stage 4: CVXPY solve (deterministic)")

        if t_total > 0:
            # Percentages are approximate: each stage is timed independently with
            # fresh problem construction, so run-to-run variance means they may not
            # sum to exactly 100%. The max(0, ...) on RemoveUncertainty handles
            # cases where timing noise makes the subtraction negative.
            print("\n  Approximate breakdown (% of total):")
            print(f"    Canonicalization:              ~{t_canon/t_total*100:.1f}%")
            print(f"    generate_torch_expressions:    ~{t_torch/t_total*100:.1f}%")
            print(f"    RemoveUncertainty (estimated): ~"
                  f"{max(0, (t_remove_total-t_canon-t_torch)/t_total*100):.1f}%")
            print(f"    CVXPY solve:                   ~{t_solve/t_total*100:.1f}%")


def main():
    benchmark_reshape_tensor()
    benchmark_get_problem_data()
    benchmark_generate_torch_expressions()
    benchmark_variable_creation()
    benchmark_has_unc_param()
    benchmark_full_solve_breakdown()


if __name__ == "__main__":
    main()
