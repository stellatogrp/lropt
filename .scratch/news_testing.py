import warnings

import cvxpy as cp
import ipdb  # For debugging remove later
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

import lropt

warnings.filterwarnings("ignore")
plt.rcParams.update({"text.usetex": True, "font.size": 24, "font.family": "serif"})


# Formulate constants
n = 2
N = 600
k_init = np.array([4.0, 5.0])


def gen_demand_cor(N, seed, x1, x2):
    np.random.seed(seed)
    sig = np.eye(2)
    mu = np.array((6, 7))
    points_list = []
    for i in range(N):
        mu_shift = -0.4 * x1 - 0.1 * x2
        newpoint = np.random.multivariate_normal(mu + mu_shift, sig)
        points_list.append(newpoint)
    return np.vstack(points_list)


s = 1
# in order for scenario to make sense, generate only 20 contexts
np.random.seed(s)
num_context = 20
num_reps = int(N / num_context)
init_k_data = np.maximum(0.5, k_init + np.random.normal(0, 3, (num_context, n)))
init_p_data = init_k_data + np.maximum(0, np.random.normal(0, 3, (num_context, n)))
p_data = np.repeat(init_p_data, num_reps, axis=0)
k_data = np.repeat(init_k_data, num_reps, axis=0)

# uncertain data depends on the contexts
init_data = [
    gen_demand_cor(num_reps, seed=5, x1=init_p_data[i], x2=init_k_data[i])
    for i in range(num_context)
]
data = np.vstack(init_data)
test_p = 0.5

# split dataset
test_indices = np.random.choice(num_reps, int(test_p * num_reps), replace=False)
train_indices = [i for i in range(num_reps) if i not in test_indices]
train = np.array([init_data[j][i] for i in train_indices for j in range(num_context)])
test = np.array([init_data[j][i] for i in test_indices for j in range(num_context)])


def gen_weights_bias(k_data, p_data, data):
    stacked_context = np.hstack([p_data, k_data, np.ones((N, 1))])
    mults_1 = np.linalg.lstsq(stacked_context, data[:, 0])[0]
    mults_2 = np.linalg.lstsq(stacked_context, data[:, 1])[0]
    mults_mean = np.vstack([mults_1, mults_2])
    mults_mean_weight = mults_mean[:, :-1]
    mults_mean_bias = mults_mean[:, -1]
    return mults_mean_weight, mults_mean_bias


mults_mean_weight, mults_mean_bias = gen_weights_bias(k_data, p_data, data)


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

initn = sc.linalg.sqrtm(np.cov(train.T))
init_bvaln = np.mean(train, axis=0)

# initialize linear weights and bias
init_bias = np.hstack([initn.flatten(), mults_mean_bias])
init_weight = np.vstack([np.zeros((4, 4)), mults_mean_weight])

# Train A and b
trainer = lropt.Trainer(prob)
trainer_settings = lropt.TrainerSettings()
trainer_settings.lr = 0.0001
trainer_settings.num_iter = 300  # number of training iterations
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
trainer_settings.parallel = True
trainer_settings.position = False
trainer_settings.eta = 0.3
trainer_settings.contextual = True
trainer_settings.init_weight = init_weight  # initialization with ltsq pred
trainer_settings.init_bias = init_bias
result = trainer.train(trainer_settings=trainer_settings)
df = result.df
A_fin = result.A
b_fin = result.b

# COMMENT
# - Until now it is really slow and I get an error many times.
# trainer.py:161: UserWarning: To copy construct from a tensor, ...


ipdb.set_trace()

# no training (steps = 1, look at initalized set)
trainer_settings.num_iter = 1
result2 = trainer.train(trainer_settings=trainer_settings)
df2 = result.df
A_fin2 = result.A
b_fin2 = result.b


ipdb.set_trace()

# cov_pred initialization, untrained
trainer_settings.num_iter = 1
trainer_settings.covpred = True
result3 = trainer.train(trainer_settings=trainer_settings)
df3 = result3.df
A_fin3 = result3.A
b_fin3 = result3.b

eps_list = np.linspace(0.5, 2.5, 10)
# covpred untrained
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


ipdb.set_trace()

# cov_pred initialization, trained
trainer_settings.lr = 0.00001
trainer_settings.num_iter = 500
trainer_settings.covpred = True
result4 = trainer.train(trainer_settings=trainer_settings)
df4 = result4.df
A_fin4 = result4.A
b_fin4 = result4.b


ipdb.set_trace()

# training df
result.df


ipdb.set_trace()

# testing df
result.df_test


ipdb.set_trace()


def plot_iters(dftrain, dftest, title, steps=2000, logscale=True):
    plt.rcParams.update({"text.usetex": True, "font.size": 22, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))

    # ax1.plot(dftrain["Violation_val"][:steps],
    #          label="Out-of-sample empirical CVaR")
    ax1.plot(dftrain["Violations_train"][:steps], label="In-sample empirical CVaR", linestyle="--")
    ax1.plot(dftest["Violations_train"][:steps], label="In-sample empirical CVaR", linestyle="--")

    ax1.set_xlabel("Iterations")
    ax1.hlines(
        xmin=0,
        xmax=dftrain["Violations_train"][:steps].shape[0],
        y=-0.0,
        linestyles="--",
        color="black",
        label="Target threshold: 0",
    )
    ax1.legend()
    ax2.plot(dftrain["Train_val"][:steps], label="Objective value")
    ax2.set_xlabel("Iterations")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax2.legend()
    if logscale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    plt.savefig(title + "_iters.pdf", bbox_inches="tight")


ipdb.set_trace()

# training with linear NN - erratic balues
plot_iters(result.df, result.df_test, steps=trainer_settings.num_iter, title="training")


ipdb.set_trace()

# training with covpred initialization - converges better but objective is much worse
plot_iters(result4.df, result4.df_test, steps=trainer_settings.num_iter, title="training")


ipdb.set_trace()

# Grid search epsilon
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


ipdb.set_trace()

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


ipdb.set_trace()


def calc_eval(x, p, k, u, t):
    val = 0
    vio = 0
    for i in range(u.shape[0]):
        val_cur = k @ x + np.max(
            [
                -p[0] * x[0] - p[1] * x[1],
                -p[0] * x[0] - p[1] * u[i][1],
                -p[0] * u[i][0] - p[1] * x[1],
                -p[0] * u[i][0] - p[1] * u[i][1],
            ]
        )
        val += val_cur
        vio += val_cur >= t
    return val / u.shape[0], vio / u.shape[0]


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
print(context_evals / num_context, context_probs / num_context)


ipdb.set_trace()

# indss = [5,9,13,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35]
# dfgrid, dfgrid2 = results_grid4[(16,0.3)]
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
plt.savefig("news_objective_vs_violations.pdf", bbox_inches="tight")


ipdb.set_trace()

K = 5
num_p = 50
offset = 5
x_min, x_max = np.min(train[:, 0]) - offset, np.max(train[:, 0]) + offset
y_min, y_max = np.min(train[:, 1]) - offset, np.max(train[:, 1]) + offset
X = np.linspace(x_min, x_max, num_p)
Y = np.linspace(y_min, y_max, num_p)
x, y = np.meshgrid(X, Y)


def gen_set(Amat, bvec, inds, K):
    fin_set = {}
    for ind in range(4):
        fin_set[ind] = {}
        for k_ind in range(K):
            fin_set[ind][k_ind] = np.zeros((num_p, num_p))
    for ind in range(4):
        for i in range(num_p):
            for j in range(num_p):
                u_vec = [x[i, j], y[i, j]]
                if K == 1:
                    fin_set[ind][0][i, j] = np.linalg.norm(
                        (1 / eps_list[inds[ind]])
                        * (Amat.T @ np.linalg.inv(Amat @ Amat.T))
                        @ (u_vec - bvec)
                    )
                else:
                    for k_ind in range(K):
                        fin_set[ind][k_ind][i, j] = np.linalg.norm(
                            (1 / eps_list[inds[ind]])
                            * (Amat[k_ind].T @ np.linalg.inv(Amat[k_ind] @ Amat[k_ind].T))
                            @ (u_vec - bvec[k_ind])
                        )
    return fin_set


Amat = result._a_history[-1] * result.rho
bvec = result._b_history[-1]
fin_set = gen_set(Amat, bvec, inds_reshaped, K)
init_set = gen_set(initn, init_bvaln, inds_standard, K=1)
Amat2 = result2._a_history[-1] * result2.rho
bvec2 = result2._b_history[-1]
untrained_set = gen_set(Amat2, bvec2, inds_untrained, K)
Amat3 = result3._a_history[-1] * result3.rho
bvec3 = result3._b_history[-1]
covpred_set_untrained = gen_set(Amat3, bvec3, inds_covpred_untrained, K)
Amat4 = result4._a_history[-1] * result4.rho
bvec4 = result4._b_history[-1]
covpred_set_trained = gen_set(Amat4, bvec4, inds_covpred_trained, K)


ipdb.set_trace()


def plot_contours_line(x, y, set, prob_list, train, title, standard=True, K=1):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    ax_lst = [ax1, ax2, ax3, ax4]
    np.random.seed(0)
    newtrain = np.random.choice(num_reps, size=15, replace=False)
    cur_ind = 0
    for axis in ax_lst:
        axis.set_title(r"$\hat{\eta}$" + " = {}".format(prob_list[cur_ind]))
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        axis.scatter(np.mean(train, axis=0)[0], np.mean(train, axis=0)[1], color=["tab:green"])
        for k_ind in range(num_context):
            axis.scatter(
                init_data[k_ind][:, 0][newtrain],
                init_data[k_ind][:, 1][newtrain],
                edgecolor="black",
            )
        for k_ind in range(K):
            axis.contour(x, y, set[cur_ind][k_ind], [1], colors=["red"], linewidths=[2])
        cur_ind += 1
    if standard:
        post = "Mean-Variance"
    else:
        post = title
    fig.suptitle(post + " set", fontsize=30)
    plt.savefig(title + "_" + post + ".pdf", bbox_inches="tight")


ipdb.set_trace()

plot_contours_line(x, y, init_set, prob_list, train, "standard", standard=True)
plot_contours_line(x, y, fin_set, prob_list, train, "reshaped", standard=False, K=5)
plot_contours_line(x, y, untrained_set, prob_list, train, "untrained", standard=False, K=5)
plot_contours_line(
    x, y, covpred_set_untrained, prob_list, train, "covpred_untrained", standard=False, K=5
)
plot_contours_line(
    x, y, covpred_set_trained, prob_list, train, "covpred_trained", standard=False, K=5
)


ipdb.set_trace()
