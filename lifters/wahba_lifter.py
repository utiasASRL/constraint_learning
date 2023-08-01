import matplotlib
import matplotlib.pylab as plt

matplotlib.use("TkAgg")
plt.ion()

import autograd.numpy as np

from lifters.mono_lifter import MonoLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.geometry import get_C_r_from_theta

N_TRYS = 10

NOISE = 1e-2
N_OUTLIERS = 0

MAX_DIST = 2.0  # maximum distance of camera from landmarks

# TODO(FD) we need to add a penalty here, otherwise the local solution is not good.
# However, the penalty results in inequality constraints etc. and that's not easy to deal with.
PENALTY_RHO = 10
PENALTY_U = 1e-3


class WahbaLifter(MonoLifter):
    def h_list(self, t):
        """
        We want to inforce that
        - norm(t) <= 10
        as constraints h_j(t)<=0
        """
        return [np.sqrt(np.sum(t**2)) - MAX_DIST]

    def generate_random_setup(self):
        """Generate a new random setup. This is called once and defines the toy problem to be tightened."""
        self.theta
        self.landmarks = np.random.normal(
            loc=0, scale=1.0, size=(self.n_landmarks, self.d)
        )
        self.parameters = np.r_[1.0, self.landmarks.flatten()]
        return

    def generate_random_theta(self):
        """Generate a random new feasible point, this is the ground truth."""

        i = 0
        success = False
        while not success:
            i += 1
            pc_cw = np.random.uniform(0, MAX_DIST**(1/self.d), size=self.d)
            success = np.linalg.norm(pc_cw) < MAX_DIST
            if success:
                break
            elif i > N_TRYS:
                raise ValueError("didn't find feasible point")

        n_angles = self.d * (self.d - 1) // 2
        angles = np.random.uniform(0, 2 * np.pi, size=n_angles)
        if self.robust:
            w = [-1] * N_OUTLIERS + [1.0] * (self.n_landmarks - N_OUTLIERS)
            return np.r_[pc_cw, angles, w]
        return np.r_[pc_cw, angles]

    def get_B_known(self):
        """Get inequality constraints of the form x.T @ B @ x >= 0"""
        dim_x = self.d + self.d**2

        # enforce that norm(t) <= MAX_DIST
        B1 = PolyMatrix(symmetric=True)
        constraint = np.zeros((dim_x, dim_x))
        constraint[range(self.d), range(self.d)] = 1.0
        B1["l", "l"] = -MAX_DIST
        B1["x", "x"] = constraint
        return [B1.get_matrix(self.var_dict)]

    def term_in_norm(self, R, t, pi, ui):
        return R @ pi + t - ui

    def residual(self, R, t, pi, ui):
        # TODO: can easily extend below to matrix-weighted
        W = np.eye(self.d)
        return (R @ pi + t - ui).T @ W @ (R @ pi + t - ui)

    def get_Q(self, noise: float = None):
        if noise is None:
            noise = NOISE
        y = np.empty((self.n_landmarks, self.d))
        n_angles = self.d * (self.d - 1) // 2
        theta = self.theta[: self.d + n_angles]
        R, t = get_C_r_from_theta(theta, self.d)
        for i in range(self.n_landmarks):
            pi = self.landmarks[i]
            # ui = deepcopy(pi) #R @ pi + t
            ui = R @ pi + t
            ui += np.random.normal(scale=noise, loc=0, size=self.d)
            y[i] = ui
        Q = self.get_Q_from_y(y)
        return Q, y

    def get_Q_from_y(self, y):
        """
        every cost term can be written as
        (1 + wi)  [l x'] Qi [l; x] + 1 - wi
        = [l x'] Qi [l; x] + wi * [l x'] Qi [l; x] + 1 - wi
        = Pi_ll l**2 + 2 x.T @ Pi_xl * l + x'T Qi x
        + wi * (Pi_ll l**2 + 2 x.T @ Pi_xl * l + x'T Qi x)
        + 1 - wi
        """
        from poly_matrix.poly_matrix import PolyMatrix

        Q = PolyMatrix(symmetric=True)

        Wi = np.eye(self.d)
        for i in range(self.n_landmarks):
            pi = self.landmarks[i]
            ui = y[i]
            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]
            Pi_ll = ui.T @ Wi @ ui
            Pi_xl = -(Pi.T @ Wi @ ui)[:, None]
            Qi = Pi.T @ Wi @ Pi
            if self.robust:
                Qi /= self.beta**2
                Pi_ll /= self.beta**2
                Pi_xl /= self.beta**2
                Q["x", "x"] += Qi
                Q["x", "l"] += Pi_xl
                Q["l", "l"] += 1 + Pi_ll  # 1 from (1 - wi), Pi_ll from first term.
                Q["l", f"w_{i}"] += -0.5  # from (1 - wi), 0.5 cause on off-diagonal
                if self.level == "xwT":
                    Q[f"z_{i}", "x"] += 0.5 * Qi
                    Q[f"w_{i}", "l"] += 0.5 * Pi_ll
                    Q[f"z_{i}", "l"] += Pi_xl
                elif self.level == "xxT":
                    Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]
                    Q[f"w_{i}", "l"] += 0.5 * Pi_ll
                    Q["x", f"w_{i}"] += Pi_xl
            else:
                Q["x", "x"] += Qi
                Q["x", "l"] += Pi_xl
                Q["l", "l"] += Pi_ll  # on diagonal
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse

    def __repr__(self):
        appendix = "_robust" if self.robust else ""
        return f"wahba_{self.d}d_{self.level}_{self.param_level}{appendix}"
