# %%
import warnings

import cvxpy as cp
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

import torch

import lropt

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "text.usetex": True,

    "font.size": 18,
    "font.family": "serif"
})
colors = ["tab:blue", "tab:green", "tab:orange",
          "blue", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "red"]
# %%


def pareto_frontier(Xs, Ys, maxX=False, maxY=False):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:  # Look for higher values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]:  # Look for lower values of Y…
                p_front.append(pair)  # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

# %%


# Formulate constants
n = 2
N = 50
# k = npr.uniform(1,4,n)
# p = k + npr.uniform(2,5,n)
k = np.array([4., 5.])
p = np.array([5, 6.5])
k_tch = torch.tensor(k, requires_grad=True)
p_tch = torch.tensor(p, requires_grad=True)


def loss(t, x, alpha, data, mu=1, l=5, quantile=0.95, target=1.):
    sums = torch.mean(torch.maximum(
        torch.maximum(k_tch@x - data@(p_tch), k_tch@x - x@(p_tch)) - t - alpha,
        torch.tensor(0., requires_grad=True)))
    sums = sums/(1-quantile) + alpha
    return t + l*(sums - target) + (mu/2)*(sums - target)**2, t, torch.mean((torch.maximum(
        torch.maximum(k_tch@x - data@(p_tch), k_tch@x - x@(p_tch)) - t,
        torch.tensor(0., requires_grad=True)) >= 0.001).float()), sums.detach().numpy()

# import ipdb
# ipdb.set_trace()


data = np.load("data4.npy")

# %%
# data = gen_demand(n,N)
# Formulate uncertainty set
u = lropt.UncertainParameter(n,
                             uncertainty_set=lropt.Ellipsoidal(p=2,
                                                               data=data, loss=loss))
# Formulate the Robust Problem
x_r = cp.Variable(n)
t = cp.Variable()
# y = cp.Variable()

objective = cp.Minimize(t)

constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]


constraints += [x_r >= 0]

prob = lropt.RobustProblem(objective, constraints)
target = -0.2
s = 13


init = sc.linalg.sqrtm(sc.linalg.inv(np.cov(data.T)))
init_bval = -init@np.mean(data, axis=0)
# Train A and b
result1 = prob.train(lr=0.001, step=100, momentum=0.8, optimizer="SGD", seed=s, init_A=init, init_b=init_bval,
                     fixb=False, init_mu=1, init_lam=5, target_cvar=target, init_alpha=-0.1, save_iters=True)
df1 = result1.df
A_fin = result1.A
b_fin = result1.b
A1_iters, b1_iters = result1.uncset_iters

result2 = prob.train(eps=True, lr=0.001, step=100, momentum=0.8, optimizer="SGD", seed=s,
                     init_A=A_fin, init_b=b_fin, init_mu=1, init_lam=5, target_cvar=target, init_alpha=-0.1)
df_r1 = result2.df

result3 = prob.train(eps=True, lr=0.001, step=100, momentum=0.8, optimizer="SGD", seed=s, init_A=init,
                     init_b=init_bval, init_mu=1, init_lam=5,  target_cvar=target, init_alpha=-0.1)
df_r2 = result3.df

# Grid search epsilon
result4 = prob.grid(epslst=np.linspace(0.01, 3, 40), init_A=result3.A, init_b=result3.b, seed=s, init_alpha=-0.1)
dfgrid = result4.df

result5 = prob.grid(epslst=np.linspace(0.01, 3, 40), init_A=A_fin, init_b=b_fin, seed=s, init_alpha=-0.1)
dfgrid2 = result5.df


# %%
def final_solve(A_final, b_final):

    n = 2
    u = lropt.UncertainParameter(n,
                                 uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                   A=A_final, b=b_final))
    # Formulate the Robust Problem
    x_r = cp.Variable(n)
    t = cp.Variable()

    objective = cp.Minimize(t)

    constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]
    constraints += [x_r >= 0]

    prob = lropt.RobustProblem(objective, constraints)
    prob.solve()
    # result3.reform_problem.solve()
    x_final = x_r.value
    t_final = t.value

    return x_final, t_final


x_opt_learned, t_opt_learned = final_solve(A_fin, b_fin)
x_opt_base, t_opt_base = final_solve(init, init_bval)
# %%


def final_solve(A_final, b_final):

    n = 2
    u = lropt.UncertainParameter(n,
                                 uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                   A=A_final, b=b_final))
    # Formulate the Robust Problem
    x_r = cp.Variable(n)
    t = cp.Variable()

    objective = cp.Minimize(t)

    constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]
    constraints += [x_r >= 0]

    prob = lropt.RobustProblem(objective, constraints)
    prob.solve()
    # result3.reform_problem.solve()
    x_final = x_r.value
    t_final = t.value

    return x_final, t_final


x_opt_learned, t_opt_learned = final_solve(A_fin, b_fin)
x_opt_base, t_opt_base = final_solve(init, init_bval)

# %%
# Gifs

offset = 1
x_min, x_max = np.min(data[:, 0]) - offset, np.max(data[:, 0]) + offset
y_min, y_max = np.min(data[:, 1]) - offset, np.max(data[:, 1]) + offset
n_points = 100
X = np.linspace(x_min, x_max, n_points)
Y = np.linspace(y_min, y_max, n_points)
x_mesh, y_mesh = np.meshgrid(X, Y)


def level_set(A_final, b_final, offset=2, n=n_points, x_mesh=x_mesh, y_mesh=y_mesh):
    x_opt, t_opt = final_solve(A_final, b_final)
    unc_level_set = np.zeros((n, n))
    g_level_set = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u_vec = [x_mesh[i, j], y_mesh[i, j]]
            unc_level_set[i, j] = np.linalg.norm(A_final @ u_vec + b_final)
            g_level_set[i, j] = np.maximum(k @ x_opt - p @ x_opt, k @ x_opt - p @ u_vec) - t_opt

    return unc_level_set, g_level_set


unc_level_learned, g_level_learned = level_set(A_fin, b_fin)
unc_level_base, g_level_base = level_set(init, init_bval)


filenames = []
for i in range(len(A1_iters)):
    unc_level, g_level = level_set(A1_iters[i], b1_iters[i])
    plt.figure(figsize=(5, 5))
    plt.title("Original and reshaped sets")
    # Set axis label for the contour plot
    plt.xlabel(r"$u_1$")
    plt.ylabel(r"$u_2$")

    plt.contour(x_mesh, y_mesh, unc_level_base, [1], colors=["tab:blue"], label="Initial Set")
    plt.contour(x_mesh, y_mesh, unc_level, [1], colors=["tab:red"], label="trained Set")
    plt.contour(x_mesh, y_mesh, unc_level_learned, [1], colors=["tab:green"], label="Final Set")

    filename = f'gif_images/{i}.png'
    filenames.append(filename)

    plt.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black", s=10)
    plt.savefig(filename)
    plt.close()

# %%


with imageio.get_writer('intro.gif', mode='I') as writer:
    for i, filename in enumerate(filenames):
        if i < 50:
            image = imageio.imread(filename)
            writer.append_data(image)

# %%


eps_list = np.linspace(0.01, 3, 40)
inds = [20, 13, 10, 7]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# ax1.plot((df_r2['A_norm']), df_r2['Violations'], color="tab:blue", label=r"Original set")
# ax1.plot((df_r1['A_norm']), df_r1['Violations'], color="tab:orange", label="Reshaped set")
# # ax1.set_yscale("log")
# ax1.set_xlabel("Size parameter $\epsilon$")
# ax1.set_ylabel("Value of constraint violation")
plt.figure(figsize=(10, 5))
plt.plot(dfgrid['Violations'][:], dfgrid['Test_val'][:], color="tab:blue", label=r"Standard set", marker="v", zorder=0)
for ind in range(4):
    plt.scatter(dfgrid['Violations'][inds[ind]], dfgrid['Test_val']
                [inds[ind]], color="tab:green", s=50, marker="v", zorder=10)
    plt.annotate(r"$\epsilon$ = {}".format(round(eps_list[inds[ind]], 2)),  # this is the text
                 # these are the coordinates to position the label
                 (dfgrid['Violations'][inds[ind]], dfgrid['Test_val'][inds[ind]]),
                 textcoords="offset points",  # how to position the text
                 xytext=(5, 3),  # distance from text to points (x,y)
                 ha='left', color="tab:green", fontsize=15)

plt.plot(dfgrid2['Violations'], dfgrid2['Test_val'], color="tab:orange", label="Reshaped set", marker="^", zorder=1)
for ind in [0, 1, 3]:
    plt.scatter(dfgrid2['Violations'][inds[ind]], dfgrid2['Test_val'][inds[ind]], color="black", s=50, marker="^")
    plt.annotate(r"$\epsilon$ = {}".format(round(eps_list[inds[ind]], 2)),  # this is the text
                 # these are the coordinates to position the label
                 (dfgrid2['Violations'][inds[ind]], dfgrid2['Test_val'][inds[ind]]),
                 textcoords="offset points",  # how to position the text
                 xytext=(5, 1),  # distance from text to points (x,y)
                 ha='left', color="black", fontsize=15)
plt.ylabel("Objective valye")
# ax2.set_xlim([-1,20])
plt.xlabel("Probability of constraint violation")
plt.legend()
# lgd = plt.legend(loc = "lower right", bbox_to_anchor=(1.5, 0.3))
plt.savefig("obj_vs_constr_violation.pdf", bbox_inches='tight')
# plt.savefig("ex1_curves1.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

# %%
eps_list = np.linspace(0.01, 3, 40)
inds = [20, 13, 10, 7]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
# ax1.plot((df_r2['A_norm']), df_r2['Violations'], color="tab:blue", label=r"Original set")
# ax1.plot((df_r1['A_norm']), df_r1['Violations'], color="tab:orange", label="Reshaped set")
# # ax1.set_yscale("log")
# ax1.set_xlabel("Size parameter $\epsilon$")
# ax1.set_ylabel("Value of constraint violation")
plt.figure(figsize=(10, 5))
plt.plot(dfgrid['Violation_val'][:], dfgrid['Test_val'][:],
         color="tab:blue", label=r"Standard set", marker="v", zorder=0)
for ind in range(4):
    plt.scatter(dfgrid['Violation_val'][inds[ind]], dfgrid['Test_val']
                [inds[ind]], color="tab:green", s=50, marker="v", zorder=10)
    plt.annotate(r"$\epsilon$ = {}".format(round(eps_list[inds[ind]], 2)),  # this is the text
                 # these are the coordinates to position the label
                 (dfgrid['Violation_val'][inds[ind]], dfgrid['Test_val'][inds[ind]]),
                 textcoords="offset points",  # how to position the text
                 xytext=(5, 3),  # distance from text to points (x,y)
                 ha='left', color="tab:green", fontsize=15)

plt.plot(dfgrid2['Violation_val'], dfgrid2['Test_val'], color="tab:orange", label="Reshaped set", marker="^", zorder=1)
for ind in [1, 3]:
    plt.scatter(dfgrid2['Violation_val'][inds[ind]], dfgrid2['Test_val'][inds[ind]], color="black", s=50, marker="^")
    plt.annotate(r"$\epsilon$ = {}".format(round(eps_list[inds[ind]], 2)),  # this is the text
                 # these are the coordinates to position the label
                 (dfgrid2['Violation_val'][inds[ind]], dfgrid2['Test_val'][inds[ind]]),
                 textcoords="offset points",  # how to position the text
                 xytext=(5, 1),  # distance from text to points (x,y)
                 ha='left', color="black", fontsize=15)
plt.ylabel("Objective valye")
# ax2.set_xlim([-1,20])
plt.xlabel("Empirical $\mathbf{CVaR}$")
plt.legend()
# lgd = plt.legend(loc = "lower right", bbox_to_anchor=(1.5, 0.3))
plt.savefig("obj_vs_cvar.pdf", bbox_inches='tight')
# plt.savefig("ex1_curves1.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')


# %%
x_opt_base = {}
x_opt_base1 = {}
x_opt_learned = {}
x_opt_learned1 = {}
t_learned = {}
t_base = {}
for ind in range(4):
    n = 2
    u = lropt.UncertainParameter(n,
                                 uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                   A=(1/eps_list[inds[ind]])*result3.A, b=result3.b))
    # Formulate the Robust Problem
    x_r = cp.Variable(n)
    t = cp.Variable()
    # y = cp.Variable()

    objective = cp.Minimize(t)

    constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]
    # constraints += [cp.maximum(k1@(x_r-u),0) <= y]
    # constraints = [cp.maximum(u@k + k1@x_r -u@(k + k1), u@k + k1@x_r - x_r@(k + k1))<=t]

    constraints += [x_r >= 0]

    prob = lropt.RobustProblem(objective, constraints)
    prob.solve()
    # result3.reform_problem.solve()
    x_opt_base[ind] = x_r.value

    t_base[ind] = t.value

    n = 2
    u = lropt.UncertainParameter(n,
                                 uncertainty_set=lropt.Ellipsoidal(p=2,
                                                                   A=(1/eps_list[inds[ind]])*A_fin, b=b_fin))
    # Formulate the Robust Problem
    x_r = cp.Variable(n)
    t = cp.Variable()
    # y = cp.Variable()

    objective = cp.Minimize(t)

    constraints = [cp.maximum(k@x_r - p@x_r, k@x_r - p@u) <= t]
    # constraints += [cp.maximum(k1@(x_r-u),0) <= y]
    # constraints = [cp.maximum(u@k + k1@x_r -u@(k + k1), u@k + k1@x_r - x_r@(k + k1))<=t]

    constraints += [x_r >= 0]

    prob = lropt.RobustProblem(objective, constraints)
    prob.solve()
    # result3.reform_problem.solve()
    x_opt_learned[ind] = x_r.value

    t_learned[ind] = t.value
    x_opt_learned, x_opt_base, t_learned, t_base
    # A_fin, init, b_fin, init_bval
# x_opt_learned,x_opt_base,t_learned, t_base

# %%


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


K = 1
plt.figure(figsize=(5, 5))
num_p = 50
offset = 2
x_min, x_max = np.min(data[:, 0]) - offset, np.max(data[:, 0]) + offset
y_min, y_max = np.min(data[:, 1]) - offset, np.max(data[:, 1]) + offset
X = np.linspace(x_min, x_max, num_p)
Y = np.linspace(y_min, y_max, num_p)
x, y = np.meshgrid(X, Y)
# Z values as a matrix
fin_set = {}
init_set = {}
for ind in range(4):
    fin_set[ind] = {}
    init_set[ind] = {}
    for k_ind in range(K):
        fin_set[ind][k_ind] = np.zeros((num_p, num_p))
        init_set[ind][k_ind] = np.zeros((num_p, num_p))


g_level_learned = {}
g_level_base = {}

for ind in range(4):
    # init_set = np.zeros((num_p,num_p))
    g_level_learned[ind] = np.zeros((num_p, num_p))
    g_level_base[ind] = np.zeros((num_p, num_p))

    # # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z
    # for i in range(num_p):
    #     for j in range(num_p):
    #         u_vec = [x[i,j], y[i,j]]
    #         fin_set[i,j] = np.linalg.norm(A_fin@ u_vec  + b_fin)
    #         init_set[i,j] = np.linalg.norm(init@ u_vec + init_bval)
    #         g_level_learned[i,j] = np.maximum(k @ x_opt_learned - p @ x_opt_learned, k @ x_opt_learned - p @ u_vec)
    #         g_level_base[i,j] = np.maximum(k @ x_opt_base - p @ x_opt_base, k @ x_opt_base - p @ u_vec)

    # Populate Z Values (a 7x7 matrix) - For a circle x^2+y^2=z
    for i in range(num_p):
        for j in range(num_p):
            u_vec = [x[i, j], y[i, j]]
            for k_ind in range(K):
                fin_set[ind][k_ind][i, j] = np.linalg.norm(
                    (1/eps_list[inds[ind]])*A_fin[k_ind*n:(k_ind+1)*n, 0:n] @ u_vec + (1/eps_list[inds[ind]])*b_fin)

            for k_ind in range(K):
                init_set[ind][k_ind][i, j] = np.linalg.norm(
                    (1/eps_list[inds[ind]])*result3.A[k_ind*n:(k_ind+1)*n, 0:n] @ u_vec + (1/eps_list[inds[ind]])*result3.b)
            # init_set[i,j] = np.linalg.norm(init@ u_vec - init@np.mean(data, axis = 0))
            g_level_learned[ind][i, j] = np.maximum(k @ x_opt_learned[ind] - p @
                                                    x_opt_learned[ind], k @ x_opt_learned[ind] - p @ u_vec)
            g_level_base[ind][i, j] = np.maximum(
                k @ x_opt_base[ind] - p @ x_opt_base[ind], k @ x_opt_base[ind] - p @ u_vec)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)

ax1.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[0]], 2)))
ax1.set_xlabel(r"$u_1$")
ax1.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax1.contour(x, y, init_set[0][k_ind], [1], colors=["red"], linewidths=[2])
a1 = ax1.contourf(x, y, g_level_base[0], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax1.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax1.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
aa = fig.colorbar(a1, ax=ax1)

ax2.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[1]], 2)))
ax2.set_xlabel(r"$u_1$")
ax2.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax2.contour(x, y, init_set[1][k_ind], [1], colors=["red"], linewidths=[2])
a2 = ax2.contourf(x, y, g_level_base[1], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax2.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax2.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
fig.colorbar(a2, ax=ax2, boundaries=np.linspace(-10, 20, 10))


ax3.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[2]], 2)))
# Set x axis label for the contour plot
ax3.set_xlabel(r"$u_1$")
# Set y axis label for the contour plot
ax3.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax3.contour(x, y, init_set[2][k_ind], [1], colors=["red"], linewidths=[2])
a3 = ax3.contourf(x, y, g_level_base[2], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax3.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax3.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
fig.colorbar(a3, ax=ax3, boundaries=np.linspace(-10, 20, 10))

ax4.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[3]], 2)))
ax4.set_xlabel(r"$u_1$")
ax4.set_ylabel(r"$u_2$")

for k_ind in range(K):
    ax4.contour(x, y, init_set[3][k_ind], [1], colors=["red"], linewidths=[2])
a4 = ax4.contourf(x, y, g_level_base[3], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax4.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax4.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
fig.colorbar(a4, ax=ax4, boundaries=np.linspace(-10, 20, 10))
fig.suptitle("Standard set", fontsize=30)

plt.savefig("original_sets.pdf", bbox_inches='tight')

# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)

ax1.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[0]], 2)))
ax1.set_xlabel(r"$u_1$")
ax1.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax1.contour(x, y, fin_set[0][k_ind], [1], colors=["red"], linewidths=[2])
a1 = ax1.contourf(x, y, g_level_learned[0], np.arange(-10, 40, 10), extend='both', alpha=0.7)
ax1.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax1.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
aa = fig.colorbar(a1, ax=ax1)

ax2.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[1]], 2)))
ax2.set_xlabel(r"$u_1$")
ax2.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax2.contour(x, y, fin_set[1][k_ind], [1], colors=["red"], linewidths=[2])
a2 = ax2.contourf(x, y, g_level_learned[1], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax2.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax2.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
fig.colorbar(a2, ax=ax2, boundaries=np.linspace(-10, 20, 10))


ax3.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[2]], 2)))
ax3.set_xlabel(r"$u_1$")
ax3.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax3.contour(x, y, fin_set[2][k_ind], [1], colors=["red"], linewidths=[2])
a3 = ax3.contourf(x, y, g_level_learned[2], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax3.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax3.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
fig.colorbar(a3, ax=ax3, boundaries=np.linspace(-10, 20, 10))

ax4.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[3]], 2)))
ax4.set_xlabel(r"$u_1$")
ax4.set_ylabel(r"$u_2$")
for k_ind in range(K):
    ax4.contour(x, y, fin_set[3][k_ind], [1], colors=["red"], linewidths=[2])
a4 = ax4.contourf(x, y, g_level_learned[3], np.arange(-10, 40, 10), extend='both', alpha=0.7)

ax4.scatter(data[:, 0], data[:, 1], color="white", edgecolors="black")
ax4.scatter(np.mean(data, axis=0)[0], np.mean(data, axis=0)[1], color=["red"])
fig.colorbar(a4, ax=ax4, boundaries=np.linspace(-10, 20, 10))
fig.suptitle("Reshaped set", fontsize=30)

plt.savefig("reshaped_sets.pdf", bbox_inches='tight')


# %%
