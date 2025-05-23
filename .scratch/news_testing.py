import os
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from utils import (
    calc_eval,
    gen_demand_cor,
    gen_set,
    gen_weights_bias,
    plot_contours_line,
    plot_iters,
)

import lropt

warnings.filterwarnings("ignore")

plt.rcParams.update({"text.usetex": True, "font.size": 24, "font.family": "serif"})

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


# Generate data
n = 2
N = 600
k_init = np.array([4.0, 5.0])

# in order for scenario to make sense, generate only 20 contexts
s = 1
np.random.seed(s)
num_context = 20
num_reps = int(N / num_context)
init_k_data = np.maximum(0.5, k_init + np.random.normal(0, 3, (num_context, n)))
init_p_data = init_k_data + np.maximum(0, np.random.normal(0, 3, (num_context, n)))
p_data = np.repeat(init_p_data, num_reps, axis=0)
k_data = np.repeat(init_k_data, num_reps, axis=0)

# generate uncertain data that on the contexts
init_data = [
    gen_demand_cor(num_reps, seed=5, x1=init_p_data[i], x2=init_k_data[i])
    for i in range(num_context)
]
data = np.vstack(init_data)

# split dataset
test_p = 0.5
test_indices = np.random.choice(num_reps, int(test_p * num_reps), replace=False)
train_indices = [i for i in range(num_reps) if i not in test_indices]
train = np.array([init_data[j][i] for i in train_indices for j in range(num_context)])
test = np.array([init_data[j][i] for i in test_indices for j in range(num_context)])

# find best initialization for the linear predictor
mults_mean_weight, mults_mean_bias = gen_weights_bias(k_data, p_data, data, N)


# create lropt problem
# Formulate uncertainty set
u = lropt.UncertainParameter(n, uncertainty_set=lropt.Ellipsoidal(data=data))
# Formulate the Robust Problem
x_r = cp.Variable(n)
t = cp.Variable()
k = lropt.ContextParameter(2, data=k_data)
p = lropt.ContextParameter(2, data=p_data)
p_x = cp.Variable(n)
objective = cp.Minimize(t)
constraints = [
    lropt.max_of_uncertain(
        [
            -p[0] * x_r[0] - p[1] * x_r[1],
            -p[0] * x_r[0] - p_x[1] * u[1],
            -p_x[0] * u[0] - p[1] * x_r[1],
            -p_x[0] * u[0] - p_x[1] * u[1],
        ]
    )
    + k @ x_r
    <= t
]
constraints += [p_x == p]
constraints += [x_r >= 0]

eval_exp = k @ x_r + cp.maximum(
    -p[0] * x_r[0] - p[1] * x_r[1],
    -p[0] * x_r[0] - p[1] * u[1],
    -p[0] * u[0] - p[1] * x_r[1],
    -p[0] * u[0] - p[1] * u[1],
)

prob = lropt.RobustProblem(objective, constraints, eval_exp=eval_exp)

# train lropt problem - linear predictor
num_iters = 500
initn = sc.linalg.sqrtm(np.cov(train.T))
init_bvaln = np.mean(train, axis=0)

# initialize linear weights and bias
init_bias = np.hstack([initn.flatten(), mults_mean_bias])
init_weight = np.vstack([np.zeros((4, 4)), mults_mean_weight])


trainer = lropt.Trainer(prob)
trainer_settings = lropt.TrainerSettings()

trainer_settings.lr = 0.0001
trainer_settings.num_iter = num_iters  # number of training iterations
trainer_settings.optimizer = "SGD"
trainer_settings.seed = 5
trainer_settings.init_A = initn
trainer_settings.init_b = init_bvaln
trainer_settings.init_lam = 0.5
trainer_settings.init_mu = 0.5
trainer_settings.mu_multiplier = 1.001
trainer_settings.test_percentage = test_p
trainer_settings.save_history = True
trainer_settings.lr_step_size = 50  # scheduler - every 50 steps, multiply by lr_gamma
trainer_settings.lr_gamma = 0.5
trainer_settings.random_init = False  # if true, initializes at random sets
trainer_settings.num_random_init = 5
trainer_settings.parallel = False
trainer_settings.position = False
trainer_settings.eta = 0.3
trainer_settings.contextual = True
trainer_settings.init_weight = init_weight  # initialization with linear pred
trainer_settings.init_bias = init_bias  # initialization with linear pred
result = trainer.train(trainer_settings=trainer_settings)
df = result.df
result.df.to_csv("output/linear_training.csv")
result.df_test.to_csv("output/linear_testing.csv")
A_fin = result.A
b_fin = result.b


# untrained linear initailization (steps = 1, look at initalized set for linear predictor)
trainer_settings.num_iter = 1
result2 = trainer.train(trainer_settings=trainer_settings)
df2 = result.df
A_fin2 = result.A
b_fin2 = result.b


# cov_pred initialization, untrained (look at initalized set for covariance predictor)
trainer_settings.num_iter = 1
trainer_settings.covpred = True
result3 = trainer.train(trainer_settings=trainer_settings)
df3 = result3.df
A_fin3 = result3.A
b_fin3 = result3.b

# covpred untrained grid (vary epsilon only)
eps_list = np.linspace(0.5, 2.5, 10)
result_grid4 = trainer.grid(
    rholst=eps_list,
    init_A=A_fin3,
    init_b=b_fin3,
    seed=s,
    init_alpha=0.0,
    test_percentage=test_p,
    quantiles=(0.3, 0.7),
    contextual=True,
    covpred=True,
)
dfgrid4 = result_grid4.df
dfgrid4.to_csv("output/covpred_untrained_grid.csv")

# cov_pred initialization, trained
trainer_settings.lr = 0.00001
trainer_settings.num_iter = num_iters
trainer_settings.covpred = True
result4 = trainer.train(trainer_settings=trainer_settings)
df4 = result4.df
A_fin4 = result4.A
b_fin4 = result4.b
result4.df.to_csv("output/covpred_training.csv")
result4.df_test.to_csv("output/covpred_testing.csv")

# Analyze results
# training with linear NN - erratic balues
plot_iters(
    result.df,
    result.df_test,
    steps=num_iters,
    title="linear_training",
    num_iters=num_iters,
    trainer_settings=trainer_settings,
)

# training with covpred initialization - converges better but objective is much worse
plot_iters(
    result4.df,
    result4.df_test,
    steps=num_iters,
    title="covpred_training",
    num_iters=num_iters,
    trainer_settings=trainer_settings,
)

# Grid search for all methods
# mean variance set
eps_list = np.linspace(0.5, 2.5, 10)
result_grid = trainer.grid(
    rholst=eps_list,
    init_A=initn,
    init_b=init_bvaln,
    seed=s,
    init_alpha=0.0,
    test_percentage=test_p,
    quantiles=(0.3, 0.7),
)
dfgrid = result_grid.df
dfgrid.to_csv("output/mean_var_grid.csv")

# trained linear NN
result_grid2 = trainer.grid(
    rholst=eps_list,
    init_A=A_fin,
    init_b=b_fin,
    seed=s,
    init_alpha=0.0,
    test_percentage=test_p,
    quantiles=(0.3, 0.7),
    contextual=True,
    linear=result._linear,
)
dfgrid2 = result_grid2.df
dfgrid2.to_csv("output/linear_trained_grid.csv")

# untrained linear NN
result_grid3 = trainer.grid(
    rholst=eps_list,
    init_A=A_fin2,
    init_b=b_fin2,
    seed=s,
    init_alpha=0.0,
    test_percentage=test_p,
    quantiles=(0.3, 0.7),
    contextual=True,
    linear=result2._linear,
)
dfgrid3 = result_grid3.df
dfgrid3.to_csv("output/linear_untrained_grid.csv")

# covpred trained
result_grid5 = trainer.grid(
    rholst=eps_list,
    init_A=A_fin4,
    init_b=b_fin4,
    seed=s,
    init_alpha=0.0,
    test_percentage=test_p,
    quantiles=(0.3, 0.7),
    contextual=True,
    covpred=True,
)
dfgrid5 = result_grid5.df
dfgrid5.to_csv("output/covpred_trained_grid.csv")


# get indices for plotting
epslst = eps_list
prob_list = np.array([0.0, 0.01, 0.05, 0.1])
inds_standard = []
inds_reshaped = []
inds_untrained = []
inds_covpred_untrained = []
inds_covpred_trained = []
for i in prob_list:
    inds_standard.append(
        np.absolute(np.mean(np.vstack(dfgrid["Avg_prob_test"]), axis=1) - i).argmin()
    )
    inds_reshaped.append(
        np.absolute(np.mean(np.vstack(dfgrid2["Avg_prob_test"]), axis=1) - i).argmin()
    )
    inds_untrained.append(
        np.absolute(np.mean(np.vstack(dfgrid3["Avg_prob_test"]), axis=1) - i).argmin()
    )
    inds_covpred_untrained.append(
        np.absolute(np.mean(np.vstack(dfgrid4["Avg_prob_test"]), axis=1) - i).argmin()
    )
    inds_covpred_trained.append(
        np.absolute(np.mean(np.vstack(dfgrid5["Avg_prob_test"]), axis=1) - i).argmin()
    )

# scenario approach
context_evals = 0
context_probs = 0
# solve for each context and average
for context in range(num_context):
    u = lropt.UncertainParameter(
        n, uncertainty_set=lropt.Scenario(data=init_data[context][train_indices])
    )
    x_s = cp.Variable(n)
    t1 = cp.Variable()
    k1 = init_k_data[context]
    p1 = init_p_data[context]
    objective = cp.Minimize(t1)
    constraints = [
        lropt.max_of_uncertain(
            [
                -p1[0] * x_s[0] - p1[1] * x_s[1],
                -p1[0] * x_s[0] - p1[1] * u[1],
                -p1[0] * u[0] - p1[1] * x_s[1],
                -p1[0] * u[0] - p1[1] * u[1],
            ]
        )
        + k1 @ x_s
        <= t1
    ]
    constraints += [x_s >= 0]

    prob_sc = lropt.RobustProblem(objective, constraints)
    prob_sc.solve()
    eval, prob_vio = calc_eval(
        x_s.value,
        init_p_data[context],
        init_k_data[context],
        init_data[context][test_indices],
        t1.value,
    )
    context_evals += eval
    context_probs += prob_vio
print("contextual test values", context_evals / num_context, context_probs / num_context)

# plot comparisons
beg1, end1 = 0, 100
beg2, end2 = 0, 100
plt.figure(figsize=(15, 5))
plt.plot(
    np.mean(np.vstack(dfgrid["Avg_prob_test"]), axis=1)[beg1:end1],
    np.mean(np.vstack(dfgrid["Test_val"]), axis=1)[beg1:end1],
    color="tab:blue",
    label=r"Mean-Var set",
    marker="v",
    zorder=0,
)
plt.plot(
    np.mean(np.vstack(dfgrid2["Avg_prob_test"]), axis=1)[beg2:end2],
    np.mean(np.vstack(dfgrid2["Test_val"]), axis=1)[beg2:end2],
    color="tab:orange",
    label="Linear Trained set",
    marker="^",
    zorder=1,
)
plt.plot(
    np.mean(np.vstack(dfgrid3["Avg_prob_test"]), axis=1)[beg2:end2],
    np.mean(np.vstack(dfgrid3["Test_val"]), axis=1)[beg2:end2],
    color="tab:green",
    label="Linear Untrained set",
    marker="^",
    zorder=2,
)
plt.plot(
    np.mean(np.vstack(dfgrid4["Avg_prob_test"]), axis=1)[beg2:end2],
    np.mean(np.vstack(dfgrid4["Test_val"]), axis=1)[beg2:end2],
    color="tab:red",
    label="Covpred Untrained set",
    marker="^",
    zorder=3,
)
plt.plot(
    np.mean(np.vstack(dfgrid5["Avg_prob_test"]), axis=1)[beg2:end2],
    np.mean(np.vstack(dfgrid5["Test_val"]), axis=1)[beg2:end2],
    color="tab:purple",
    label="Covpred Trained set",
    marker="^",
    zorder=4,
)
plt.ylabel("Objective value")
plt.xlabel(r"Probability of constraint violation $(\hat{\eta})$")
# plt.ylim([-9, 0])
plt.grid()
plt.scatter(
    context_probs / num_context, context_evals / num_context, color="black", label="Scenario"
)
plt.legend()
plt.savefig("output/news_objective_vs_violations.pdf", bbox_inches="tight")


K = 5
num_p = 50
offset = 5
x_min, x_max = np.min(train[:, 0]) - offset, np.max(train[:, 0]) + offset
y_min, y_max = np.min(train[:, 1]) - offset, np.max(train[:, 1]) + offset
X = np.linspace(x_min, x_max, num_p)
Y = np.linspace(y_min, y_max, num_p)
x, y = np.meshgrid(X, Y)

Amat = result._a_history[-1] * result.rho
bvec = result._b_history[-1]
fin_set = gen_set(Amat, bvec, inds_reshaped, K, num_p=num_p, x=x, y=y, eps_list=eps_list)
init_set = gen_set(initn, init_bvaln, inds_standard, K=1, num_p=num_p, x=x, y=y, eps_list=eps_list)
Amat2 = result2._a_history[-1] * result2.rho
bvec2 = result2._b_history[-1]
untrained_set = gen_set(Amat2, bvec2, inds_untrained, K, num_p=num_p, x=x, y=y, eps_list=eps_list)
Amat3 = result3._a_history[-1] * result3.rho
bvec3 = result3._b_history[-1]
covpred_set_untrained = gen_set(
    Amat3, bvec3, inds_covpred_untrained, K, num_p=num_p, x=x, y=y, eps_list=eps_list
)
Amat4 = result4._a_history[-1] * result4.rho
bvec4 = result4._b_history[-1]
covpred_set_trained = gen_set(
    Amat4, bvec4, inds_covpred_trained, K, num_p=num_p, x=x, y=y, eps_list=eps_list
)

plot_contours_line(
    x,
    y,
    init_set,
    prob_list,
    train,
    "standard",
    num_reps=num_reps,
    init_data=init_data,
    num_context=num_context,
)
plot_contours_line(
    x,
    y,
    fin_set,
    prob_list,
    train,
    "linear_trained",
    K=5,
    num_reps=num_reps,
    init_data=init_data,
    num_context=num_context,
)
plot_contours_line(
    x,
    y,
    untrained_set,
    prob_list,
    train,
    "linear_untrained",
    K=5,
    num_reps=num_reps,
    init_data=init_data,
    num_context=num_context,
)
plot_contours_line(
    x,
    y,
    covpred_set_untrained,
    prob_list,
    train,
    "covpred_untrained",
    K=5,
    num_reps=num_reps,
    init_data=init_data,
    num_context=num_context,
)
plot_contours_line(
    x,
    y,
    covpred_set_trained,
    prob_list,
    train,
    "covpred_trained",
    K=5,
    num_reps=num_reps,
    init_data=init_data,
    num_context=num_context,
)
