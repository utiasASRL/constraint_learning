""" 
Temporary copy of Ben's code, to get the local solver
"""
import numpy as np

M = np.array([[1, 0, 1], [1, 0, -1]]).reshape((2, 3))


def forward_exact(T, p_w):
    assert T.shape == (3, 3)
    assert p_w.shape[1:] == (3, 1) and len(p_w.shape) == 3
    assert np.all(p_w[:, -1, 0] == 1)
    e = np.eye(3)
    y = M @ (T @ p_w) / (e[:, 1:2].T @ T @ p_w)
    return y.reshape(-1, M.shape[0], 1)


def forward_noisy(T, p_w, sigma):
    y = forward_exact(T, p_w) + (sigma * np.random.randn())
    return y


def _T(phi):
    # phi: 3, 1
    if np.ndim(phi) == 2:
        phi_flat = phi[:, 0]
    else:
        phi_flat = phi
    x, y, theta = phi_flat
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1],
        ]
    )


def generate_problem(N, sigma):
    """
    :param N: number of landmarks
    :param sigma: noise in pixel space

    :return: y, p_w, phi, W
    where
    - y is the N x 2 x 1 measurements vector
    - p_w contains the (homogenous) landmarks in world coordinates (N x (2 + 1) x 1)
    - phi contains the pose (x, y, alpha)
    - W contains the weights N x 2 x 2 (currently simply I)
    """

    phi = np.array(
        [np.random.rand(), np.random.rand(), 2 * np.pi * np.random.rand()]
    ).reshape((3, -1))
    T = _T(phi)

    p_s = np.concatenate(
        (
            2 * np.random.rand(N, 1, 1) - 1.0,
            (2.0 * np.random.rand(N, 1, 1)) + 1.0,
            np.ones((N, 1, 1)),
        ),
        axis=1,
    )
    p_w = np.linalg.inv(T) @ p_s

    y = forward_noisy(T, p_w, sigma)
    W = np.stack([np.eye((M.shape[0]))] * N)
    return y, p_w, phi, W


# x = [c_1 ; c_2 ; r ; u_1 ; u_2 ; \dots ; u_N ; \omega], 2N + 7 variables


def E_c(i, dim):
    assert i in [0, 1]
    e = np.zeros((2, dim))
    e[:, 2 * i : 2 * i + 2] = np.eye(2)
    return e


def E_r(dim):
    e = np.zeros((2, dim))
    e[:, 4:6] = np.eye(2)
    return e


def E_v(i, dim):
    e = np.zeros((3, dim))
    e[0, 6 + 2 * i] = 1
    e[1, -1] = 1
    e[-1, 6 + 2 * i + 1] = 1
    return e


def E_omega(dim):
    e = np.zeros((1, dim))
    e[0, -1] = 1
    return e


def E_Tp(p_n, dim):
    e = np.zeros((3, dim))
    e[:2, :2] = np.eye(2) * p_n[0]
    e[:2, 2:4] = np.eye(2) * p_n[1]
    e[:2, 4:6] = np.eye(2) * p_n[2]
    e[-1, -1] = 1
    return e


def build_SDP(p_w, y, W):
    assert len(p_w.shape) == 3 and p_w.shape[1:] == (3, 1) and np.all(p_w[:, -1]) == 1
    N = p_w.shape[0]
    # assert len(W) == N and len(W.shape) == 1
    # assert len(y) == N and len(y.shape) == 1

    dim = 2 * N + 7

    e = np.eye(3)

    As = []
    bs = []

    _E_omega = E_omega(dim)
    # Cost
    Q = sum(
        (y[n] @ _E_omega - M @ E_v(n, dim)).T
        @ W[n]
        @ (y[n] @ _E_omega - M @ E_v(n, dim))
        for n in range(N)
    )

    # homogenization constraint
    As.append(_E_omega.T @ _E_omega)
    bs.append(1)

    # Rotation Matrix constraints
    for i in range(2):
        for j in range(i, 2):
            A = E_c(j, dim).T @ E_c(i, dim)
            As.append(0.5 * (A + A.T))
            bs.append(int(i == j))

    # Measurment constraints
    for n in range(N):
        for k in [0, 2]:
            A = E_v(n, dim).T @ e[:, k : k + 1] @ e[:, 1:2].T @ E_Tp(
                p_w[n], dim
            ) - _E_omega.T @ e[:, k : k + 1].T @ E_Tp(p_w[n], dim)
            As.append(0.5 * (A + A.T))
            bs.append(0)

    return Q, As, bs


def build_x(T, p_w):
    x = [T[:-1, 0:1], T[:-1, 1:2], T[:-1, 2:3]]
    e = np.eye(3)
    v = T @ p_w / (e[:, 1:2].T @ T @ p_w)
    u = v[:, [0, -1]]
    for un in u:
        x.append(un)
    x.append(np.array([1])[:, None])
    x = np.concatenate(x, axis=0)
    return x


# --- local solver functions -----


def _svdsolve(A: np.array, b: np.array):
    """Solve Ax = b using SVD

    Args:
        A (np.array): shape = (N, M)
        b (np.array): shape = (N, 1)

    Returns:
        x (np.array): shape = (M, 1)
    """
    u, s, v = np.linalg.svd(A)
    # if np.any(s <= 1e-10):
    #    print("rank defficient or negative A!", s)
    c = np.dot(u.T, b)
    w = np.linalg.lstsq(np.diag(s), c, rcond=None)[0]
    x = np.dot(v.T, w)
    return x


def _qn(phi, p_w):
    # p_w.shape = (N, 3, 1)
    T = _T(phi)  # (3, 3)
    return T @ p_w  # (N, 3, 1)


def _un(y, p_w, phi):
    # y = (N, 1)
    y = y.reshape((p_w.shape[0], M.shape[0], 1))
    q = _qn(phi, p_w)  # (N, 3, 1)
    I = np.eye(3)  # (1, 3) -> (N, 3, 3)
    u = y - ((M @ q) / (I[:, 1:2].T @ q))  # (N, 1, 1)
    # assert u.shape == (p_w.shape[0], 1, 1)
    return u


def _dq_dphi(p_w, phi):
    # p_w: (N, 3, 1)
    # phi: (3, 1)
    theta = phi[2, 0]
    res = np.zeros((p_w.shape[0], 3, 3))
    res[:, 0, 0] = 1
    res[:, 1, 1] = 1
    res[:, 0, 2] = -np.sin(theta) * p_w[:, 0, 0] - np.cos(theta) * p_w[:, 1, 0]
    res[:, 1, 2] = np.cos(theta) * p_w[:, 0, 0] - np.sin(theta) * p_w[:, 1, 0]
    return res  # (N, 3, 3)


def _du_dq(p_w, phi):
    # p_w: (N, 3, 1)
    # phi: (3, 1)
    q = _qn(phi, p_w)  # (N, 3, 1)
    I = np.eye(3)
    du_dq = (1 / (I[:, 1:2].T @ q)) * (
        (1 / (I[:, 1:2].T @ q)) * (M @ q @ I[:, 1:2].T) - M
    )  # (N, 1, 3)
    # assert du_dq.shape == (p_w.shape[0], 1, 3)
    return du_dq


def _du_dphi(p_w, phi):
    # p_w: (N, 3, 1)
    # phi: (3, 1)
    du_dphi = _du_dq(p_w, phi) @ _dq_dphi(p_w, phi)  # (N, 1, 3)
    # assert du_dphi.shape == (p_w.shape[0], 1, 3)
    return du_dphi


def _cost(p_w, y, phi, W=None):
    # W: (N, 1, 1)
    N = p_w.shape[0]
    u = _un(y, p_w, phi)
    if W is None:
        W = np.stack([np.eye(2)] * N)
    return np.sum(u.transpose((0, 2, 1)) @ W @ u, axis=0)


def local_solver(
    p_w, y, init_phi, W=None, max_iters=500, min_update_norm=1e-10, log=False
):
    from scipy.optimize import minimize

    def fun(x, *args):
        return _cost(*args, x)[0, 0]

    res = minimize(fun, x0=init_phi, args=(p_w, y))
    return res.success, res.x, res.fun


def local_solver_old(
    p_w, y, init_phi, W=None, max_iters=500, min_update_norm=1e-10, log=False
):
    """
    :param y:   the N x 2 x 1 measurements vector
    :param p_w: contains the (homogenous) landmarks in world coordinates (N x (2 + 1) x 1)
    :param init_phi: contains the pose (x, y, alpha)
    :param W:   contains the weights N x 2 x 2 (currently simply I)
    """
    N = p_w.shape[0]
    assert y.shape == (N, 2, 1)
    assert p_w.shape == (N, 3, 1)
    assert init_phi.shape == (3, 1)
    if W is not None:
        assert W.shape == (N, 2, 2)
    else:
        W = np.stack([np.eye(2)] * N)

    # see notes for math
    phi = init_phi
    i = 0
    perturb_mag = np.inf
    solved = True
    alpha = 1.0
    while (perturb_mag > min_update_norm) and (i < max_iters):
        current_cost = _cost(p_w=p_w, W=W, y=y, phi=phi)
        if log:
            print(f"Current cost: {current_cost}")
        du_dphi = _du_dphi(p_w, phi)  # (N, 1, 3)
        # (3, 1)
        u = _un(y, p_w, phi)  # (N, M.shape[0], 1)
        b = -np.sum(u.transpose((0, 2, 1)) @ W @ du_dphi, axis=0)  # (1, 3)
        A = np.sum(du_dphi.transpose((0, 2, 1)) @ W @ du_dphi, axis=0)  # (3, 3)
        dphi = _svdsolve(A.T, b.T)  # (3 , 1)
        next_cost = _cost(p_w=p_w, W=W, y=y, phi=phi + alpha * dphi)
        if next_cost > current_cost:
            alpha *= 0.5
            print(f"reduced alpha to {alpha}")
            if alpha < 1e-5:
                print("stagnated")
                solved = False
                break
            continue
        phi += alpha * dphi

        alpha = 1.0
        perturb_mag = np.linalg.norm(dphi)
        i += 1
        if i == max_iters:
            solved = False
    cost = _cost(p_w=p_w, W=W, y=y, phi=phi)
    return solved, phi, cost


if __name__ == "__main__":
    N = 3
    sigma = 0.5
    y, p_w, phi_gt, W = generate_problem(N, sigma)

    global_min = np.inf
    global_soln = None

    for i in range(20):
        init_phi = np.array(
            [np.random.rand(), np.random.rand(), 2 * np.pi * np.random.rand()]
        ).reshape((3, -1))
        solved, phi_local, local_cost = local_solver(
            p_w=p_w, y=y, W=W, init_phi=init_phi, log=True
        )
        if solved and (local_cost < global_min):
            global_soln = phi_local
            global_min = local_cost

    print(f"Ground truth:\n{phi_gt}")
    print(f"Global soln:\n{global_soln}")

    # make x from best local solution
    T_global = _T(global_soln)
    x_global = T_global[:2, :].T.reshape(-1, 1)
    I = np.eye(3)
    u = (T_global @ p_w) / (I[:, 1:2].T @ T_global @ p_w)
    u = u[:, [0, 2], :].reshape(-1, 1)
    x_global = np.r_[x_global.flatten(), u.flatten(), 1]
    print(x_global)
