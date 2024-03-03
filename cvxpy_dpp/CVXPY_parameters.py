import marimo

__generated_with = "0.2.13"
app = marimo.App()


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return cp, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        CVXPY's conic form is

        $$
            \min\{ c^T x + d \,:\, x \in \mathbb{R}^{n},\, A x + b \in K \}
        $$

        (see [docs](https://www.cvxpy.org/contributing/index.html#cvxpy-s-conic-form)).

        Note:
        Some solvers (e.g., SCS and Clarabel) support a quadratic objective function.
        """
    )
    return


@app.cell
def _(cp, np):
    # Example
    x = cp.Variable(2)
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    c = np.array([1, 1])
    _problem = cp.Problem(cp.Minimize(c.T @ x), [A @ x + b >= 0])
    _data = _problem.get_problem_data(cp.SCS)

    print(_data[0]["A"].toarray())  # Note that here we have -A because SCS has a different conic form Ax + s = b, s \in K
    print(_data[0]["b"])
    print(_data[0]["c"])
    return A, b, c, x


@app.cell
def _(mo):
    mo.md(
        r"""
        When parameters are involved, we add another dimension to each object (e.g., $c$ and $b$ become matrices, $A$ becomes a 3D tensor).

        In the case of $A \in \mathbb{R}^{m \times n}$, we get $T \in \mathbb{R}^{m \times n \times p + 1}$, with each slice in the third dimension corresponding to the linear dependence on that entry of the parameter vector, with the one additional dimension being everything that does not depend on parameters.

        To obtain $A$ for a given parameter vector, we perform a dot product of $T$ with the parameter vector, and then sum over the third dimension (i.e., a tensor contraction).

        Formally, let \( T_{:,:,i} \) denote the \( i \)-th slice of \( T \) along the third dimension for \( i = 1, \ldots, p \), and \( T_{:,:,p+1} \) denote the parameter-independent slice. Then, for a parameter vector \( \mathbf{v} = [v_1, v_2, \ldots, v_p]^\top \), the matrix \( A \) is obtained by:

        \[ A = \sum_{i=1}^{p} (T_{:,:,i} \cdot v_i) + T_{:,:,p+1} \]
        """
    )
    return


@app.cell
def _(A, b, c, cp, x):
    # Example
    p = cp.Parameter(2,value=[1,1])
    problem = cp.Problem(cp.Minimize(c.T @ x), [A @ cp.multiply(x,p) + b >= 0])
    data = problem.get_problem_data(cp.SCS)

    print(data[0]["A"].toarray())  # Note that here we have -A because SCS has a different conic form Ax + s = b, s \in K
    print(data[0]["b"])
    print(data[0]["c"])
    return data, p, problem


@app.cell
def _(data):
    param_cone_prog = data[0]['param_prob']
    # The tensor is concatenated with b and then flattened into 2D because we don't have 3D sparse arrays
    # Since we can work with numpy in this example, we can reshape it back to 3D

    # param_cone_prog.A is the tensor
    A_tensor = param_cone_prog.A.toarray().reshape((2,3,3), order='F')[:,:-1,:]
    for i in range(3):
        print(A_tensor[:,:,i])
    return A_tensor, i, param_cone_prog


@app.cell
def _(A_tensor, np):
    # Define parameter vector p
    _p_val = np.array([1,1,1]) # last entry is always 1 because it is the constant term

    # (1,1,1) gives the orginal problem
    np.dot(A_tensor, _p_val)
    return


@app.cell
def _(A_tensor, np):
    _p_val = np.array([2,1,1])
    # Entries that depend on the first entry of p are multiplied by 2
    np.dot(A_tensor, _p_val)
    return


@app.cell
def _(A_tensor, np):
    _p_val = np.array([1,2,1])
    # Entries that depend on the second entry of p are multiplied by 2
    np.dot(A_tensor, _p_val)
    return


@app.cell
def __(mo):
    p_0 = mo.ui.slider(0.5,1.5, value=1, step=0.001)
    p_1 = mo.ui.slider(0.5,1.5, value=1, step=0.001)
    p_0, p_1
    return p_0, p_1


@app.cell
def __(A_tensor, np, p_0, p_1, plt):
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    p_val = np.array([p_0.value, p_1.value, 1])
    A_voxel = A_tensor * p_val
    A_voxel = np.moveaxis(A_voxel, [0,1,2], [2,0,1])
    A_voxel = np.flip(A_voxel,axis=2)

    # Prepare the data for voxel display
    # The voxels to be displayed need to be in a boolean array, with True indicating the presence of a voxel
    _x, _y, _z = np.indices(np.array(A_voxel.shape) + 1)
    voxels = np.ones(A_voxel.shape, dtype=bool)

    # Use the values in A as colors
    # Normalize A for color mapping
    norm = plt.Normalize(0, 6)
    colors = plt.cm.viridis(norm(A_voxel))
    colors[...,-1] = 0.7

    colors[A_voxel == 0, -1 ] = 0

    # Display the voxels
    ax.voxels(_x, _y, _z, voxels, facecolors=colors, edgecolor='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    _ax = fig.add_subplot(212)

    plt.imshow(np.dot(A_tensor, p_val), cmap='viridis', norm=norm)

    fig
    return A_voxel, ax, colors, fig, norm, p_val, voxels


if __name__ == "__main__":
    app.run()
