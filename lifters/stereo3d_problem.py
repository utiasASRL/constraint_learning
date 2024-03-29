import numpy as np
from pylgmath.se3.operations import vec2tran
from pylgmath.so3.operations import hat

f_u = 484.5
f_v = 484.5
c_u = 322
c_v = 247
b = 0.24
M = np.array(
    [
        [f_u, 0, c_u, f_u * b / 2],
        [0, f_v, c_v, 0],
        [f_u, 0, c_u, -f_u * b / 2],
        [0, f_v, c_v, 0],
    ]
)


def _u(y: np.array, x: np.array, M: np.array):
    """
    Pixel error.

    Args:
        y (np.array): (N, 4, 1)
        x (np.array): (N, 4, 1)
        M (np.array): (4, 4)
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    return y - 1 / (a.transpose((0, 2, 1)) @ x) * (M @ x)


def _du(x: np.array, M: np.array):
    """
    Derivative of pixel error w.r.t. x
    Args:
        x (np.array): (N, 4 ,1)
        M (np.array): (4, 4 ,1)
    """
    a = np.zeros_like(x)
    a[:, 2] = 1
    a_times_x = a.transpose((0, 2, 1)) @ x
    return ((1 / a_times_x) ** 2) * M @ x @ a.transpose((0, 2, 1)) - (1 / a_times_x) * M


def _odot_exp(e: np.array):
    """
    Computes e^{\\odot}. See section 8.1.8 of State Estimation for Robotics (Barfoot 2022)
    Args:
        e (np.array): (N, 4, 1)
    Returns:
        np.array: (N, 4, 6)
    """
    assert len(e.shape) == 3
    eta = e[:, -1]  # (N, 1)
    n = e.shape[0]
    res = np.zeros((n, 4, 6))
    res[:, :3, 3:] = -hat(e[:, :-1])  # (N, 3, 3)
    res[:, :3, :3] = eta[:, :, None] * np.eye(3)  # (3, 3) # (N, 3, 3)
    return res


def Jacobian(x: np.array, M: np.array):
    """
    Internal function used for local solver. See stereo_camera_sim.ipynb for definition.

    Args:
        x (np.array): shape = (N, 4, 1)
        M (np.array): shape = (4, 4)

    Returns:
        np.array: shape = (6, 4)
    """
    return (_du(x, M) @ _odot_exp(x)).transpose((0, 2, 1))


def _svdsolve(A: np.array, b: np.array):
    """Solve Ax = b using SVD

    Args:
        A (np.array): shape = (N, M)
        b (np.array): shape = (N, 1)

    Returns:
        x (np.array): shape = (M, 1)
    """
    u, s, v = np.linalg.svd(A)
    c = np.dot(u.T, b)
    w = np.linalg.lstsq(np.diag(s), c, rcond=None)[0]
    x = np.dot(v.T, w)
    return x


def generative_camera_model(
    M: np.array, T_cw: np.array, homo_p_w: np.array
) -> np.array:
    """
    Args:
        M (np.array): Stereo camera parameters in a matrix
        T_cw (np.array): Rigid transform from world to camera frame
        homo_p_w (np.array): homogenous points in world frame

    Returns:
        y (np.array): (N, 4, 1), points in image space
            y[:, :2] are points in left image (row, col), indexed from the top left
            y[:, 2:] are points in right image (row, col), indexed from the top left
    """
    p_c = T_cw @ homo_p_w
    return M @ p_c / p_c[:, None, 2]


def residuals(T: np.array, p_w: np.array, y: np.array, W: np.array, M: np.array):
    y_pred = generative_camera_model(M, T, p_w)
    e = y - y_pred
    return e.flatten()


def _cost(T: np.array, p_w: np.array, y: np.array, W: np.array, M: np.array):
    """Compute projection error

    Args:
        y (np.array): (N, 4, 1), measurments
        T (np.array): (4, 4), rigid transform estimate
        M (np.array): (4, 4), camera parameters
        p_w (np.array): (N, 4, 1), homogeneous point coordinates in the world frame
        W (np.array): (N, 4, 4) or (4, 4) or scalar, weight matrix/scalar

    Returns:
        error: scalar error value
    """
    y_pred = generative_camera_model(M, T, p_w)
    e = y - y_pred
    return np.sum(e.transpose((0, 2, 1)) @ W @ e, axis=0)[0][0]


def local_solver(
    T_init: np.array,
    y: np.array,
    p_w: np.array,
    W: np.array,
    M: np.array,
    max_iters: int = 100,
    min_update_norm: float = 1e-10,
    gtol=1e-6,
    log: bool = False,
    backtracking: bool = True,
) -> tuple:
    """Solve the stereo localization problem with a gauss-newton method

    Args:
        T_init (np.array): initial guess for transformation matrix T_cw
        y (np.array): stereo camera measurements, shape = (N, 4, 1)
        p_w (np.array): Landmark homogeneous coordinates in world frame, shape = (N, 4, 1)
        W (np.array): weight matrix/scalar shape = (N, 4, 4) or (4, 4) or scalar
        M (np.array): Stereo camera parameter matrix, shape = (4, 4)
        max_iters (int, optional): Maximum iterations before returning. Defaults to 1000.
        min_update_norm (float, optional): . Defaults to 1e-10.
    """
    assert max_iters > 0, "Maximum iterations must be positive"

    info = {}

    i = 0
    perturb_mag = np.inf
    T_op = T_init.copy()

    if log:
        print("step size \t cost \t backtrack")

    rho = 0.5
    alpha_bar = 1.0
    c = 0.1

    while i < max_iters:
        Jk = Jacobian(T_op @ p_w, M)
        rk = _u(y, T_op @ p_w, M)
        A = np.sum(Jk @ Jk.transpose((0, 2, 1)), axis=0)
        b = np.sum(-Jk @ rk, axis=0)
        pk = _svdsolve(A, b)

        if backtracking:
            cost = _cost(T_op, p_w, y, W, M)
            for j in range(10):
                alpha = rho**j * alpha_bar
                T_op_new = vec2tran(pk * alpha) @ T_op

                cost_new = _cost(T_op_new, p_w, y, W, M)
                grad_k = b
                if cost_new <= cost + c * alpha * grad_k.T @ pk:
                    break
            # print(f"it {i}: found valid alpha after {j}")
            perturb_mag = np.linalg.norm(pk * alpha)
            T_op = T_op_new
            cost = cost_new

        # norm_g = np.linalg.norm(b) / np.sqrt(len(b))
        rmse_g = np.max(b)
        if rmse_g <= gtol:  # rmse of gradient
            info["success"] = True
            info["msg"] = f"converged in gradient after {i} iterations."
            break
        elif perturb_mag <= min_update_norm:
            info["success"] = True
            info["msg"] = f"reached minimum stepsize after {i} iterations."
            if rmse_g > 1e-5:
                print(f"Warning: inaccurate gradient {rmse_g}")
            break

        if log:
            print(f"{perturb_mag:.5f} \t {cost:.5f} \t {j}")
        i = i + 1
        if i == max_iters:
            info["success"] = False
            info["msg"] = f"reached maximum iterations ({max_iters})"

    res = residuals(T_op, p_w, y, W, M)
    info["max res"] = np.max(np.abs(res))
    eigs = np.linalg.eigvalsh(A)
    info["cond Hess"] = eigs[-1] / eigs[0]  # max eig / min eig

    cost = _cost(T_op, p_w, y, W, M)
    return info, T_op, cost
