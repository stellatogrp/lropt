import cvxpy as cp
import numpy as np

'''
Worst case a for ellipsoid inequality
'''
# Reformulate explicitly
np.random.seed(0)
n = 5
c = np.random.rand(n)
b = 10.
x = cp.Variable(n)
objective = cp.Minimize(c * x)
# Robust set
rho = 0.2
p = 2

# Formulate robust problem explicitly with cvxpy
constraints = [rho * cp.norm(x, p=p) <= b]
prob_cvxpy = cp.Problem(objective, constraints)
prob_cvxpy.solve(solver=cp.GUROBI)
x_ellip = x.value
a_wc = rho * x_ellip / np.linalg.norm(x_ellip)

viol = np.linalg.norm(a_wc.dot(x_ellip) - b)
print("Violation ellip: ", viol)

'''
Worst case for polyhedral uncertainty
'''
# Polyhedral constraint (make a box)
n_poly = 2 * n
A_poly = np.vstack((np.eye(n),
                    -np.eye(n)))
b_poly = np.concatenate((.1 * np.ones(n),
                         .2 * np.ones(n)))

# Formulate robust problem explicitly with cvxpy
p = cp.Variable(n_poly)
con_poly = [p * b_poly <= b,
            p.T * A_poly == x,
            p >= 0]
prob_poly = cp.Problem(objective, con_poly)
prob_poly.solve(solver=cp.CVXOPT)
x_poly = x.value
p_poly = p.value

# Compute dual variable
a_wc = con_poly[1].dual_value
viol = np.linalg.norm(a_wc.dot(x_poly) - b)
print("Violation poly: ", viol)


a = cp.Variable(n)
obj_prob = cp.Maximize(a * x_poly)
con_prob = [A_poly * a <= b_poly]

cp.Problem(obj_prob, con_prob).solve(solver=cp.GUROBI)

p = cp.Variable(n_poly)
obj_prob_dual = cp.Maximize(p * b_poly)
con_prob_dual = [p * b_poly <= b,
                 p.T * A_poly == x_poly,
                 p >= 0]
cp.Problem(obj_prob_dual, con_prob_dual).solve(solver=cp.GUROBI)


# con_prob[0].dual_value == p.value  # Corresponds!


# Check that it corresponds to worst case value
