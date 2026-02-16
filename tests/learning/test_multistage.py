import unittest

import cvxpy as cp
import numpy as np
import scipy as sc
import torch

import cvxro
from cvxro import TrainerSettings


class TestMultiStage(unittest.TestCase):
    def test_multistage(self):
        T = 12
        torch.set_default_dtype(torch.double)
        K = 4
        c = (
            torch.tensor([(1 + 0.5 * np.sin(np.pi * (t - 1) / (T * 0.5))) for t in range(1, T + 1)])
            * 0.1
        )
        p = c * 2.5
        h = c * 1.2
        c = torch.hstack([c, torch.zeros(K - 1, dtype=torch.double)])
        p = torch.hstack([p, torch.zeros(K - 1, dtype=torch.double)])
        h = torch.hstack([h, torch.zeros(K - 1, dtype=torch.double)])
        d_star = np.array(
            [1000 * (1 + 0.5 * np.sin(np.pi * (t - 1) / (T * 0.5))) for t in range(1, T + 1)]
        )
        # d_star = 1000*np.ones(T)
        d_star_cat = torch.hstack(
            [torch.tensor(d_star, dtype=torch.double), torch.zeros(K, dtype=torch.double)]
        )
        proportion = 0.1
        lhs = np.concatenate((np.eye(T), -np.eye(T)), axis=0)
        rhs_upper = (1 + proportion) * d_star
        rhs_lower = (-1 + proportion) * d_star
        rhs = np.hstack((rhs_upper, rhs_lower))
        lhs_new = np.concatenate((np.eye(K), -np.eye(K)), axis=0)
        rhs_new = np.hstack((1800 * np.ones(K), np.zeros(K)))
        init_eps = 2
        cov = 2000 * np.eye(T)
        Qmax = 2000
        Vmin = -1000
        Vmax = 5000
        alpha1 = 0.0
        alpha2 = 0.0
        beta1 = 1
        beta2 = 2
        # init_size = 100
        demand_dist = torch.distributions.MultivariateNormal(
            torch.tensor(d_star, dtype=torch.double), torch.tensor(cov, dtype=torch.double)
        )
        cov = sc.linalg.sqrtm(cov)
        torch.manual_seed(100)
        # sample_val = demand_dist.sample((100,))
        # cov = sc.linalg.sqrtm(np.cov(sample_val.T))
        # init_dist = torch.distributions.Uniform(low = 0, high = 100)
        eye_concat = np.eye(T + K - 1)
        eye_concat[T:, T:] = 0
        Y_matref = np.ones((K, T + K - 1))
        Y_matref[:, T:] = 0

        # t_vals = torch.tensor(np.arange(T))
        # ones = np.ones(T)
        # zeros = np.zeros(T)
        # et_vals = torch.tensor(np.array([np.concatenate([ones[:t+1],
        #  zeros[t+1:]]) for t in range(T)]))
        def plot_inventory_levels(ax, low_values, mid_values, high_values, random_values, rho):
            ax.plot(
                range(0, T + 1),
                low_values,
                label="Disturbance Lower Bound",
                color="blue",
                linestyle="--",
            )
            ax.plot(
                range(0, T + 1),
                mid_values,
                label="Nominal Disturbance",
                color="green",
                linestyle="-",
            )
            ax.plot(
                range(0, T + 1),
                high_values,
                label="Disturbance Upper Bound",
                color="red",
                linestyle="--",
            )
            ax.plot(
                range(0, T + 1),
                random_values,
                label="Random Trajectory",
                color="orange",
                linestyle="--",
            )
            ax.set_xlabel("Time Period")
            ax.set_ylabel("Inventory Level")
            ax.set_title(f"Inventory Level Over Time\nrho = {rho}")
            ax.legend()
            ax.grid(True)

        def baseline_problem(x_init, cval, pval, hval):
            d = cvxro.UncertainParameter(
                T,
                uncertainty_set=cvxro.Ellipsoidal(p=2, rho=init_eps, c=lhs, d=rhs, a=cov, b=d_star),
            )
            # d = d_star
            q = cp.Variable(T)
            y = cp.Variable(T)
            u = cp.Variable(T)
            z = cp.Variable(T - 1)
            w = cp.Variable(T)

            objective = cval @ q + cp.sum(y) + cp.sum(u) + cp.sum(z)
            constraints = [0 <= q, q <= Qmax]
            for time in range(T):
                constraints += [Vmin <= x_init + cp.sum(q[: time + 1]) - cp.sum(d[: (time + 1)])]
                constraints += [x_init + cp.sum(q[: (time + 1)]) - cp.sum(d[: (time + 1)]) <= Vmax]
                constraints += [
                    y[time]
                    >= hval[time] * x_init
                    + hval[time] * cp.sum(q[: (time + 1)])
                    - hval[time] * cp.sum(d[: (time + 1)])
                ]
                constraints += [
                    y[time]
                    >= -pval[time] * x_init
                    - pval[time] * cp.sum(q[: (time + 1)])
                    + pval[time] * cp.sum(d[: (time + 1)])
                ]
            constraints += [u >= alpha1 * (q - w)]
            constraints += [u >= alpha2 * (w - q)]
            constraints += [z >= beta1 * (w[1:] - w[:-1])]
            constraints += [z >= beta2 * (w[:-1] - w[1:])]
            prob = cvxro.RobustProblem(cp.Minimize(objective), constraints)
            prob.solve()
            return prob.objective.value, w.value, q.value, y.value, u.value, z.value

        _, _, qval1, _, _, _ = baseline_problem(
            x_init=100,
            cval=c[:T].detach().numpy(),
            pval=p[:T].detach().numpy(),
            hval=h[:T].detach().numpy(),
        )

        x_init = 100
        low_values = [x_init]
        mid_values = [x_init]
        high_values = [x_init]
        random_values1 = [x_init]
        torch.manual_seed(0)
        random_traj = demand_dist.sample((5,))[0]
        for i in range(1, T + 1):
            low = (cp.sum(qval1[:i]) - cp.sum(rhs_upper[:i]) + x_init).value
            mid = (cp.sum(qval1[:i]) - cp.sum(d_star[:i]) + x_init).value
            high = (cp.sum(qval1[:i]) - cp.sum(-rhs_lower[:i]) + x_init).value
            traj = (cp.sum(qval1[:i]) - cp.sum(random_traj[:i]) + x_init).value
            low_values.append(low)
            mid_values.append(mid)
            high_values.append(high)
            random_values1.append(traj)
        # fig, ax = plt.subplots(figsize = (6,3))
        # plot_inventory_levels(ax, low_values, mid_values, high_values,
        # random_values1, 0.1)
        # print(objval1-sum(zval1),(c[:T].detach().numpy()@(qval1) + cp.sum
        # (yval1) + cp.sum(uval1)).value, c[:T].detach().numpy()@(qval1) + sum
        # (np.maximum(h[:T]*np.array(random_values1[1:]), -p[:T]*np.array
        # (random_values1[1:]))))

        def baseline_problem_aro(init_val, cval, pval, hval):
            d = cvxro.UncertainParameter(
                T,
                uncertainty_set=cvxro.Ellipsoidal(p=2, rho=init_eps, c=lhs, d=rhs, a=cov, b=d_star),
            )
            # d = d_star
            q = cp.Variable(T)
            y = cp.Variable(T)
            u = cp.Variable(T)
            z = cp.Variable(T - 1)
            w = cp.Variable(T)
            u_var = cp.Variable((T, T))
            y_var = cp.Variable((T, T))
            q_var = cp.Variable((T, T))
            C = cp.Variable()
            x_init = cp.Parameter()
            x_init.value = init_val

            objective = C
            constraints = [
                cval @ (q + q_var @ d) + cp.sum(y + y_var @ d) + cp.sum(u + u_var @ d) + cp.sum(z)
                <= C,
                0 <= q + q_var @ d,
                q + q_var @ d <= Qmax,
            ]
            # constraints += [y_var == 0, u_var==0]
            for time in range(T):
                for time2 in range(time, T):
                    constraints += [q_var[time, time2] == 0]
                    # constraints += [y_var[time,time2] == 0]
                    # constraints += [u_var[time,time2] == 0]

            for time in range(T):
                constraints += [
                    Vmin <= x_init + cp.sum((q + q_var @ d)[: time + 1]) - cp.sum(d[: (time + 1)])
                ]
                constraints += [
                    x_init + cp.sum((q + q_var @ d)[: (time + 1)]) - cp.sum(d[: (time + 1)]) <= Vmax
                ]
                constraints += [
                    (y + y_var @ d)[time]
                    >= hval[time] * x_init
                    + hval[time] * cp.sum((q + q_var @ d)[: (time + 1)])
                    - hval[time] * cp.sum(d[: (time + 1)])
                ]
                constraints += [
                    (y + y_var @ d)[time]
                    >= -pval[time] * x_init
                    - pval[time] * cp.sum((q + q_var @ d)[: (time + 1)])
                    + pval[time] * cp.sum(d[: (time + 1)])
                ]
            constraints += [u + u_var @ d >= alpha1 * (q + q_var @ d - w)]
            constraints += [u + u_var @ d >= alpha2 * (w - q - q_var @ d)]
            constraints += [z >= beta1 * (w[1:] - w[:-1])]
            constraints += [z >= beta2 * (w[:-1] - w[1:])]
            prob = cvxro.RobustProblem(cp.Minimize(objective), constraints)
            return prob, x_init, w, q, q_var, y, y_var, u, u_var, z

        baseline_prob, x_init, w_baseline, q, q_var, y, y_var, u, u_var, z = baseline_problem_aro(
            init_val=100,
            cval=c[:T].detach().numpy(),
            pval=p[:T].detach().numpy(),
            hval=h[:T].detach().numpy(),
        )
        baseline_prob.solve()
        _, _, qval, qmat, _, _, _, _, _ = (
            baseline_prob.objective.value,
            w_baseline.value,
            q.value,
            q_var.value,
            y.value,
            y_var.value,
            u.value,
            u_var.value,
            z.value,
        )
        w_tch = torch.tensor(w_baseline.value, dtype=torch.double)
        w_tch = torch.hstack([w_tch, torch.zeros(K - 1, dtype=torch.double)])

        # if y and u are not adjustable
        low_values = [x_init.value]
        mid_values = [x_init.value]
        high_values = [x_init.value]
        random_values = [x_init.value]
        for i in range(1, T + 1):
            low = (cp.sum((qval + qmat @ rhs_upper)[:i]) - cp.sum(rhs_upper[:i]) + x_init).value
            mid = (cp.sum((qval + qmat @ d_star)[:i]) - cp.sum(d_star[:i]) + x_init).value
            high = (cp.sum((qval - qmat @ rhs_lower)[:i]) - cp.sum(-rhs_lower[:i]) + x_init).value
            traj = (
                cp.sum((qval + qmat @ (random_traj.detach().numpy()))[:i])
                - cp.sum(random_traj.detach().numpy()[:i])
                + x_init
            ).value
            low_values.append(low)
            mid_values.append(mid)
            high_values.append(high)
            random_values.append(traj)
        # fig, ax = plt.subplots(figsize = (6,3))
        # plot_inventory_levels(ax, low_values, mid_values,
        # high_values, random_values, 0.1)
        # print(objval-sum(zval),(c[:T].detach().numpy()@(q +
        # q_var@random_traj) + cp.sum(y + y_var@random_traj) +
        # cp.sum(u + u_var@random_traj)).value, (c[:T].detach().numpy()@(q +
        # q_var@random_traj) + sum(np.maximum(h[:T]*np.array(random_values
        # [1:]), -p[:T]*np.array(random_values[1:])))).value  )

        d = cvxro.UncertainParameter(
            K,
            uncertainty_set=cvxro.Ellipsoidal(
                p=2, rho=1, c=lhs_new, d=rhs_new, data=np.zeros((2, K))
            ),
        )
        # d = cvxro.UncertainParameter(K,uncertainty_set =
        # cvxro.Ellipsoidal(p=2,rho=1,c = lhs, d = rhs, data = np.zeros((1,K))))
        # d = cvxro.UncertainParameter(K,uncertainty_set = cvxro.Ellipsoidal
        # (p=2,rho=1,c = -np.eye(K), d = np.zeros(K), a = cov[:K,:K],
        #  b = d_star[:K]))
        # d = cvxro.UncertainParameter(T,uncertainty_set = cvxro.Ellipsoidal
        # (p=2,rho=init_eps,c = lhs, d = rhs, a = cov, b = d_star))

        x_endind = T + 1 + K

        x_hat = cp.Parameter(T + 1)
        q_hat = cp.Parameter(T)
        d_hat = cp.Parameter(T)
        y_hat = cp.Parameter(T)
        u_hat = cp.Parameter(T)
        p_xhat = cp.Parameter(K)
        h_xhat = cp.Parameter(K)
        w = cp.Parameter(K)
        t = cp.Parameter(1)
        e_ind = cp.Parameter((K, K))
        p_ind = cp.Parameter((K, K))
        h_ind = cp.Parameter((K, K))
        c_ind = cp.Parameter(K)
        d_star_val = cp.Parameter(K)
        Ymat_ref = cp.Parameter((K, K))

        all_params = [
            x_hat,
            d_star_val,
            q_hat,
            d_hat,
            y_hat,
            u_hat,
            p_xhat,
            h_xhat,
            w,
            t,
            e_ind,
            p_ind,
            h_ind,
            c_ind,
            Ymat_ref,
        ]

        e_inds = cp.sum([e_ind[j] for j in range(K)])  # t...t+K

        uall = cp.Variable(4 * K)
        q = uall[:K]
        y = uall[K : 2 * K]
        u = uall[2 * K : 3 * K]
        ymat_saved = uall[3 * K : 4 * K]

        Qmat = cp.Variable((K, K))
        Qmat_new = cp.Variable((K, K))
        Qmat_masked = cp.Variable((K, K))
        Ymat = cp.Variable((K, K))
        Hmat = cp.Variable((K, K))
        Pmat = cp.Variable((K, K))
        Cvec = cp.Variable(K)
        Yvec = cp.Variable(K)
        Ymat_new = cp.Variable((K, K))

        e_indvar = cp.Variable((K, K))
        h_indvar = cp.Variable((K, K))
        p_indvar = cp.Variable((K, K))
        C = cp.Variable()

        objective = C
        constraints = []
        constraints += [Ymat_new[0] == ymat_saved]
        # constraints = [cp.sum(cp.hstack([param.flatten() for param in all_params])) >= 0]
        constraints += [cp.multiply(Ymat_ref, Ymat) == Ymat_new]
        constraints += [c_ind @ Qmat_masked == Cvec]
        constraints += [e_inds @ Ymat_new == Yvec]
        constraints += [c_ind @ q + Cvec @ d + e_inds @ y + Yvec @ d + e_inds @ u <= C]
        constraints += [t >= 0, y_hat >= 0, u_hat >= 0, d_star_val >= 0]
        constraints += [0 <= q + Qmat_masked @ d]
        constraints += [q + Qmat_masked @ d <= Qmax]
        constraints += [Qmat_masked == cp.multiply(Ymat_ref, Qmat)]
        # constraints += [Ymat==0]
        for time in range(K):
            for time2 in range(time, K):
                constraints += [Qmat[time, time2] == 0]
                constraints += [Ymat[time, time2] == 0]

        for i in range(K):
            constraints += [
                Vmin
                <= x_hat[0]
                + cp.sum(q_hat)
                + cp.sum([e_ind[j] for j in range(i + 1)]) @ q
                + Qmat_new[i] @ d
                - cp.sum(d_hat)
                - cp.sum([e_indvar[j] for j in range(i + 1)]) @ d
            ]

            constraints += [cp.sum([e_ind[j] for j in range(i + 1)]) @ Qmat == Qmat_new[i]]

            constraints += [
                x_hat[0]
                + cp.sum(q_hat)
                + cp.sum([e_ind[j] for j in range(i + 1)]) @ q
                + Qmat_new[i] @ d
                - cp.sum(d_hat)
                - cp.sum([e_indvar[j] for j in range(i + 1)]) @ d
                <= Vmax
            ]

            constraints += [h_ind[i] @ Qmat == Hmat[i]]
            constraints += [p_ind[i] @ Qmat == Pmat[i]]
            constraints += [
                y[i] + Ymat_new[i] @ d >= h_xhat[i] + h_ind[i] @ q + Hmat[i] @ d - h_indvar[i] @ d
            ]

            constraints += [
                y[i] + Ymat_new[i] @ d >= -p_xhat[i] - p_ind[i] @ q - Pmat[i] @ d + p_indvar[i] @ d
            ]

        constraints += [u >= alpha1 * (q - w), u >= alpha2 * (w - q)]
        constraints += [e_ind == e_indvar, h_ind == h_indvar, p_ind == p_indvar]

        prob = cvxro.RobustProblem(cp.Minimize(objective), constraints)
        trainer = cvxro.Trainer(prob)
        policy = trainer.create_cvxpylayer(parameters=all_params, variables=[uall])

        class InvSimulator(cvxro.Simulator):
            def simulate(self, x, u, **kwargs):
                u = u[0]
                x_hat, d_star_val, q_hat, d_hat, y_hat, u_hat, _, _, _, tval, _, _, _, _, _ = x
                t = int(tval[0])
                assert x_hat.shape[0] == u.shape[0]
                batch_size = x_hat.shape[0]
                demand = demand_dist.sample((batch_size,))[:, t]
                # demand = torch.tensor(random_traj[t]).repeat(batch_size,)
                x_orig = x_hat[:, t]
                q_newest = u[:, 0]
                q_hat[:, t] = q_newest
                d_hat[:, t] = demand
                d_star_val = d_star_cat[t + 1 : t + K + 1].repeat(batch_size, 1)
                x_hat[:, t + 1] = x_orig + q_newest - demand
                y_hat[:, t] = u[:, K]
                u_hat[:, t] = u[:, 2 * K]
                t_new = tval + 1
                e_new = eye_concat[t : t + K, t : t + K]
                w_new = w_tch[t : t + K].repeat(batch_size, 1)
                t = T - 2 if t == T - 1 else t
                p_x = torch.vstack([p[t + i + 1] * x_hat[:, t + 1] for i in range(K)]).T
                h_x = torch.vstack([h[t + i + 1] * x_hat[:, t + 1] for i in range(K)]).T
                p_new = torch.vstack([
                    p[t + j + 1] * torch.stack(
                        [torch.from_numpy(e) for e in e_new[:j + 1]]).sum(dim=0)
                    for j in range(K)
                ]).repeat(batch_size, 1, 1)
                h_new = torch.vstack([
                    h[t + j + 1] * torch.stack(
                        [torch.from_numpy(e) for e in e_new[:j + 1]]).sum(dim=0)
                    for j in range(K)
                ]).repeat(batch_size, 1, 1)
                c_new = (
                    c[t : t + K] * torch.stack([torch.from_numpy(e) for e in e_new[:K]]).sum(dim=0)
                ).repeat(batch_size, 1)
                e_new = torch.tensor(e_new).repeat(batch_size, 1, 1)
                Ymat_ref = torch.tensor(Y_matref[:, t : t + K]).repeat(batch_size, 1, 1)
                x = [
                    x_hat,
                    d_star_val,
                    q_hat,
                    d_hat,
                    y_hat,
                    u_hat,
                    p_x,
                    h_x,
                    w_new,
                    t_new,
                    e_new,
                    p_new,
                    h_new,
                    c_new,
                    Ymat_ref,
                ]
                # print(x)
                return x

            def stage_cost_avg(self, x, u):
                u = u[0]
                x_hat, _, q_hat, _, _, u_hat, _, _, _, t, _, _, _, _, _ = x
                assert x_hat.shape[0] == u.shape[0]
                t = int(t[0])
                x_hat = x_hat[:, t]
                _ = x_hat.shape[0]
                # return (torch.sum(c[:T].repeat((batch_size,1))*q_hat,axis=1)
                # + torch.sum(y_hat,axis=1)+torch.sum(u_hat,axis=1))
                return (
                    c[t - 1] * q_hat[:, t - 1]
                    + torch.max(-p[t - 1] * x_hat, h[t - 1] * x_hat)
                    + u_hat[:, t - 1]
                ).mean()

            def stage_cost_cvar(self,x,u):
                return (self.stage_cost_avg(x,u),torch.tensor(0))

            def in_sample_obj(self,x,u):
                return self.stage_cost(x,u)

            def stage_cost(self, x, u):
                u = u[0]
                x_hat, _, q_hat, _, y_hat, u_hat, _, _, _, _, _, _, _, _, _ = x
                assert x_hat.shape[0] == u.shape[0]
                batch_size = x_hat.shape[0]
                return (
                    torch.sum(c[:T].repeat((batch_size, 1)) * q_hat, axis=1)
                    + torch.sum(y_hat, axis=1)
                    + torch.sum(u_hat, axis=1)
                ).mean() / T

            def constraint_cost(self, x, u, alpha):
                u = u[0]
                eta = 0.05
                x_hat, _, _, _, y_hat, _, _, _, _, t, _, _, _, _, _ = x
                _ = u[:, 3 * K : 4 * K]
                t = int(t[0])
                x_hat = x_hat[:, t]
                y_hat = y_hat[:, t - 1]
                assert x_hat.shape[0] == u.shape[0]
                batch_size = x_hat.shape[0]
                # - y_hatmat@d_star_cat[t-1:t+K-1]
                cvar_term = (1 / eta) * torch.max(
                    torch.max(-p[t - 1] * x_hat, h[t - 1] * x_hat) - y_hat - alpha,
                    torch.zeros(batch_size),
                ) + alpha
                return (cvar_term + 0.01).mean() / T

            def init_state(self, batch_size, seed=None):
                if seed is not None:
                    torch.manual_seed(seed)
                t = 0
                x_new = torch.zeros((batch_size, T + 1), dtype=torch.double)
                x_orig = 100 * torch.ones((batch_size,), dtype=torch.double)
                w_new = w_tch[t : t + K].repeat((batch_size, 1))
                x_new[:, 0] = x_orig
                q_new = torch.zeros((batch_size, T), dtype=torch.double)
                d_star_val = d_star_cat[t : t + K].repeat(batch_size, 1)
                d_new = torch.zeros((batch_size, T), dtype=torch.double)
                y_new = torch.zeros((batch_size, T), dtype=torch.double)
                u_new = torch.zeros((batch_size, T), dtype=torch.double)
                p_x = torch.vstack([p[t + i] * x_new[:, t] for i in range(K)]).T
                h_x = torch.vstack([h[t + i] * x_new[:, t] for i in range(K)]).T
                t_new = torch.tensor(t, dtype=torch.double).repeat(batch_size, 1)
                e_new = eye_concat[t : t + K, t : t + K]
                p_new = torch.vstack([
                    p[t + j] * torch.from_numpy(np.stack(e_new[:j + 1])).sum(dim=0)
                    for j in range(K)
                ]).repeat(batch_size, 1, 1)
                h_new = torch.vstack([
                    h[t + j] * torch.stack([torch.from_numpy(e) for e in e_new[:j + 1]]).sum(dim=0)
                    for j in range(K)
                ]).repeat(batch_size, 1, 1)
                c_new = (c[t : t + K] * torch.stack(
                    [torch.from_numpy(e) for e in e_new[:K]]).sum(dim=0)).repeat(batch_size, 1)
                e_new = torch.tensor(e_new).repeat(batch_size, 1, 1)
                Ymat_ref_new = torch.tensor(Y_matref[:, t : t + K]).repeat(batch_size, 1, 1)
                x = [
                    x_new,
                    d_star_val,
                    q_new,
                    d_new,
                    y_new,
                    u_new,
                    p_x,
                    h_x,
                    w_new,
                    t_new,
                    e_new,
                    p_new,
                    h_new,
                    c_new,
                    Ymat_ref_new,
                ]
                return x

            def prob_constr_violation(self, x, u, **kwargs):
                return super().prob_constr_violation(x, u, **kwargs)

        simulator = InvSimulator()

        # Perform training
        epochs = 10
        batch_size = 5
        test_batch_size = 5
        lr = 0.0001
        # init_x0 = simulator.init_state(seed = 0, batch_size = 100)
        init_a = cov[:K, :K]
        init_b = np.zeros(K)
        # init_b = d_star[:K]
        init_weights = torch.zeros((K, x_endind))
        init_weights[:, T + 1 :] = torch.eye(K)
        settings = TrainerSettings()
        settings.set(
            simulator=simulator,
            multistage=True,
            policy=policy,
            time_horizon=T,
            num_iter=epochs,
            batch_size=batch_size,
            init_rho=1.5,
            seed=0,
            init_A=init_a,
            init_b=init_b,
            optimizer="SGD",
            lr=lr,
            momentum=0,
            init_alpha=0.0,
            scheduler=True,
            lr_step_size=20,
            lr_gamma=0.5,
            contextual=True,
            test_batch_size=test_batch_size,
            x_endind=x_endind,
            init_weight=init_weights,
            init_lam=0.001,
            init_mu=0.001,
            mu_multiplier=1.01,
            parallel = False
        )
        _ = trainer.train(settings=settings)
