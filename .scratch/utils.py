import matplotlib.pyplot as plt
import numpy as np


def plot_iters(
    dftrain, dftest, title, steps=2000, logscale=True, num_iters=None, trainer_settings=None
):
    """Plot training and testing metrics over iterations."""
    plt.rcParams.update({"text.usetex": True, "font.size": 22, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))

    ax1.plot(dftrain["Violations_train"][:steps], label="In-sample empirical CVaR", linestyle="--")
    ax1.plot(
        np.arange(0, num_iters, trainer_settings.test_frequency),
        dftest["Violations_test"][:steps],
        label="out-of-sample empirical CVaR",
        linestyle="--",
    )

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
    ax2.plot(dftrain["Train_val"][:steps], label="In-sample objective value")
    ax2.plot(
        np.arange(0, num_iters, trainer_settings.test_frequency),
        dftest["Test_val"][:steps],
        label="Out-of-sample objective value",
    )

    ax2.set_xlabel("Iterations")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax2.legend()
    if logscale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    plt.savefig(f"output/{title}_iters.pdf", bbox_inches="tight")


def calc_eval(x, p, k, u, t):
    """Calculate evaluation metrics."""
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


def gen_set(Amat, bvec, inds, K, num_p, x, y, eps_list):
    """Generate uncertainty sets."""
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


def plot_contours_line(
    x, y, set, prob_list, train, title, K=1, num_reps=None, init_data=None, num_context=None
):
    """Plot contour lines for uncertainty sets."""
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
    fig.suptitle(title + " set", fontsize=30)
    plt.savefig(f"output/{title}_set.pdf", bbox_inches="tight")


def gen_demand_cor(N, seed, x1, x2):
    """Generate correlated demand data."""
    np.random.seed(seed)
    sig = np.eye(2)
    mu = np.array((6, 7))
    points_list = []
    for i in range(N):
        mu_shift = -0.4 * x1 - 0.1 * x2
        newpoint = np.random.multivariate_normal(mu + mu_shift, sig)
        points_list.append(newpoint)
    return np.vstack(points_list)


def gen_weights_bias(k_data, p_data, data, N):
    """Find best initialization for the linear predictor."""
    stacked_context = np.hstack([p_data, k_data, np.ones((N, 1))])
    mults_1 = np.linalg.lstsq(stacked_context, data[:, 0])[0]
    mults_2 = np.linalg.lstsq(stacked_context, data[:, 1])[0]
    mults_mean = np.vstack([mults_1, mults_2])
    mults_mean_weight = mults_mean[:, :-1]
    mults_mean_bias = mults_mean[:, -1]
    return mults_mean_weight, mults_mean_bias
