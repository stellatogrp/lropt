"""Do cvxro's native feasibility columns reproduce inve_feasibility.filter_grid_df?

Rebuilds seed 0's Pretrained grid and compares the new native columns against the
completed old-path run in experiments/results/inve/pretrain/0_pretrained_grid.csv.
"""
import os, sys, time
import numpy as np, pandas as pd, cvxro
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiments", "inventory_enhanced"))
sys.path.insert(0, "experiments/inventory_enhanced")
from hydra import compose, initialize_config_dir
import inve_feasibility
from inve_common import base_settings, build_problem, make_dataset, problem_dims, rho_grid_baseline

CFG_DIR = os.path.abspath("experiments/inventory_enhanced/configs")
with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(config_name="inve_pretrain")

N, n, _, _, _, _ = problem_dims(cfg)
data, r_data, ctx, idx, init_a, init_b, _ = make_dataset(cfg, 0)
_, trainer, _, c, C, t, h = build_problem(cfg, data, r_data, ctx)
rholst = rho_grid_baseline(cfg.grid_scale)

st = base_settings(cfg, data, idx, init_a, init_b)
st.num_iter = 0
st.contextual = True
st.initialize_predictor = True
st.predictor = cvxro.LinearPredictor(
    predict_mean=True, predict_cov=True, pretrain=True, lr=0.001, epochs=100,
    n_neighbors=int(N * 0.1 * 0.3), cov_lr_mult=float(cfg.get("cov_lr_mult", 1.0)))
res = trainer.train(settings=st)
st.init_A, st.init_b = res.A, res.b
st.predictor = res._predictor

t0 = time.time()
grid = trainer.grid(rholst=rholst, settings=st)
t_native = time.time() - t0
new = grid.df.drop(columns=["z_vals", "x_vals"])

# old path, recomputed here so the timing comparison is like-for-like
t0 = time.time()
trainer.settings.predictor = res._predictor
a_te, b_te = inve_feasibility.get_ab_tensors(trainer, trainer.x_test_tch, True)
a_va, b_va = inve_feasibility.get_ab_tensors(trainer, trainer.x_validate_tch, True)
old = inve_feasibility.filter_grid_df(
    grid.df.drop(columns=["z_vals", "x_vals"]), new["Rho"].to_numpy(),
    a_te, b_te, r_data[idx["test"]], data[idx["test"]],
    a_va, b_va, r_data[idx["validate"]], data[idx["validate"]],
    t, h, c, C, cfg.target_eta, cfg.reg_eps)
t_old = time.time() - t0

pairs = [("Test_worst_feasible", "Test_worst"), ("Test_val_feasible", "Test_val"),
         ("Test_cvar_feasible", "Test_cvar"), ("Frac_feasible_test", "Frac_feasible_test"),
         ("Validate_worst_feasible", "Validate_worst"),
         ("Frac_feasible_validate", "Frac_feasible_validate")]
print(f"{'column':26} {'max |abs diff|':>15} {'max |rel diff|':>15}")
worst_rel = 0.0
for new_col, old_col in pairs:
    a = new[new_col].astype(float).to_numpy()
    b = old[old_col].astype(float).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    ad = np.abs(a[ok] - b[ok])
    rd = ad / np.maximum(np.abs(b[ok]), 1e-9)
    worst_rel = max(worst_rel, float(rd.max()) if len(rd) else 0.0)
    print(f"{new_col:26} {ad.max() if len(ad) else float('nan'):15.3e} "
          f"{rd.max() if len(rd) else float('nan'):15.3e}   (n={ok.sum()})")
print(f"\nworst relative deviation over all columns: {worst_rel:.3e}")
print(f"native grid: {t_native:6.1f}s    old filter_grid_df on top: {t_old:6.1f}s "
      f"({t_old / max(t_native,1e-9):.1f}x the native cost)")
disk = pd.read_csv("experiments/results/inve/pretrain/0_pretrained_grid.csv")
d = np.abs(disk["Test_worst"].to_numpy() - old["Test_worst"].astype(float).to_numpy())
print(f"recomputed old path vs the completed run on disk: max |diff| = {np.nanmax(d):.3e}")
