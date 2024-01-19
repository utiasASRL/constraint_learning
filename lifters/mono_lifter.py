from copy import deepcopy

import matplotlib
import matplotlib.pylab as plt

import autograd.numpy as np

from lifters.robust_pose_lifter import RobustPoseLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.geometry import get_C_r_from_theta

FOV = np.pi / 2  # camera field of view
NOISE = 1e-3  # inlier noise
NOISE_OUT = 0.1  # outlier noise

N_TRYS = 10

# TODO(FD) for some reason this is not required as opposed to what is stated in Heng's paper
# and it currently breaks tightness (might be a bug in my implementation though)
USE_INEQ = False

NORMALIZE = False


class MonoLifter(RobustPoseLifter):
    @property
    def TIGHTNESS(self):
        return "cost" if self.robust else "rank"

    def h_list(self, t):
        """
        We want to inforce that
        - norm(t) <= 10 (default)
        - tan(a/2)*t3 >= sqrt(t1**2 + t2**2)
        as constraints h_j(t)<=0
        """
        default = super().h_list(t)
        return default + [
            np.sum(t[:-1] ** 2) - np.tan(FOV / 2) ** 2 * t[-1] ** 2,
            -t[-1],
        ]

    def generate_random_setup(self):
        """Generate a new random setup. This is called once and defines the toy problem to be tightened."""
        self.landmarks = np.random.normal(
            loc=0, scale=1, size=(self.n_landmarks, self.d)
        )
        self.parameters = np.r_[1.0, self.landmarks.flatten()]
        return

    def get_random_position(self):
        pc_cw = np.random.rand(self.d) * 0.1
        # make sure all landmarks are in field of view:
        # min_dist = max(np.linalg.norm(self.landmarks[:, :self.d-1], axis=1))
        pc_cw[self.d - 1] = np.random.uniform(1, self.MAX_DIST)
        return pc_cw

    def get_B_known(self):
        """Get inequality constraints of the form x.T @ B @ x <= 0"""

        # TODO(FD) for some reason this is not required as opposed to what is stated in Heng's paper
        # and it currently breaks tightness (might be a bug in my implementation though)
        if not USE_INEQ:
            return []

        default = super().get_B_known()
        ## B2 and B3 enforce that tan(FOV/2)*t3 >= sqrt(t1**2 + t2**2)
        # 0 <= - tan**2(FOV/2)*t3**2 + t1**2 + t2**2
        B3 = PolyMatrix(symmetric=True)
        constraint = np.zeros((self.d, self.d))
        constraint[range(self.d - 1), range(self.d - 1)] = 1.0
        constraint[self.d - 1, self.d - 1] = -np.tan(FOV / 2) ** 2
        B3["t", "t"] = constraint

        # t3 >= 0
        constraint = np.zeros(self.d)
        constraint[self.d - 1] = -1
        B2 = PolyMatrix(symmetric=True)
        B2["h", "t"] = constraint[None, :]
        return default + [
            B2.get_matrix(self.var_dict),
            B3.get_matrix(self.var_dict),
        ]

    def term_in_norm(self, R, t, pi, ui):
        return R @ pi + t

    def residual_sq(self, R, t, pi, ui):
        W = np.eye(self.d) - np.outer(ui, ui)
        term = self.term_in_norm(R, t, pi, ui)
        if NORMALIZE:
            return term.T @ W @ term / (self.n_landmarks * self.d) ** 2
        else:
            return term.T @ W @ term

    def get_Q(
        self, noise: float = None, output_poly: bool = False, use_cliques: list = []
    ):
        if noise is None:
            noise = NOISE

        if self.y_ is None:
            self.y_ = np.zeros((self.n_landmarks, self.d))
            n_angles = self.d * (self.d - 1) // 2
            theta = self.theta[: self.d + n_angles]
            R, t = get_C_r_from_theta(theta, self.d)
            for i in range(self.n_landmarks):
                pi = self.landmarks[i]
                # ui = deepcopy(pi) #R @ pi + t
                ui = R @ pi + t
                ui /= ui[self.d - 1]

                if i < self.n_outliers:
                    # generate random unit vector inside the FOV cone
                    # tan(a/2)*t3 >= sqrt(t1**2 + t2**2) or t3 >= 1

                    # randomly sample a vector
                    if np.tan(FOV / 2) * ui[self.d - 1] < np.sqrt(
                        np.sum(ui[: self.d - 1] ** 2)
                    ):
                        print("warning: inlier not in FOV!!")

                    success = False
                    for _ in range(N_TRYS):
                        ui_test = deepcopy(ui)
                        ui_test[: self.d - 1] += np.random.normal(
                            scale=NOISE_OUT, loc=0, size=self.d - 1
                        )
                        # ui_test[: self.d - 1] += np.random.uniform(
                        #    low=-NOISE_OUT, high=NOISE_OUT, size=self.d - 1
                        # )
                        if np.tan(FOV / 2) * ui_test[self.d - 1] >= np.sqrt(
                            np.sum(ui_test[: self.d - 1] ** 2)
                        ):
                            success = True
                            ui = ui_test
                            break
                    if not success:
                        raise ValueError("did not find valid outlier ui")
                else:
                    ui[: self.d - 1] += np.random.normal(
                        scale=noise, loc=0, size=self.d - 1
                    )
                assert ui[self.d - 1] == 1.0
                ui /= np.linalg.norm(ui)
                self.y_[i] = ui

        Q = self.get_Q_from_y(self.y_, output_poly=output_poly, use_cliques=use_cliques)
        return Q, self.y_

    def get_Q_from_y(self, y, output_poly: bool = False, use_cliques: list = []):
        """
        every cost term can be written as
        (1 + wi)/b**2  [l x'] Qi [l; x] / norm + 1 - wi
        = [l x'] Qi/b**2 [l; x] /norm + wi * [l x']Qi/b**2[l;x] / norm + 1 - wi

        cost term:
        (Rpi + t) (I - uiui') (Rpi + t)
        """
        Q = PolyMatrix(symmetric=True)
        if NORMALIZE:
            norm = (self.n_landmarks * self.d) ** 2

        if len(use_cliques):
            js = use_cliques
        else:
            js = list(range(self.n_landmarks))

        for i in js:
            pi = self.landmarks[i]
            ui = y[i]
            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]  # I, pi x I
            Wi = np.eye(self.d) - np.outer(ui, ui)
            Qi = Pi.T @ Wi @ Pi  # "t,t, t,c, c,c: Wi, Wi @ kron, kron.T @ Wi @ kron
            if NORMALIZE:
                Qi /= norm

            if self.robust:
                Qi /= self.beta**2
                # last two terms, should not be affected by norm
                Q["h", "h"] += 1
                Q["h", f"w_{i}"] += -0.5
                if self.level == "xwT":
                    # Q[f"z_{i}", "x"] += 0.5 * Qi
                    Q[f"z_{i}", "t"] += 0.5 * Qi[:, : self.d]
                    Q[f"z_{i}", "c"] += 0.5 * Qi[:, self.d :]
                    # Q["x", "x"] += Qi
                    Q["t", "t"] += Qi[: self.d, : self.d]
                    Q["t", "c"] += Qi[: self.d, self.d :]
                    Q["c", "c"] += Qi[self.d :, self.d :]
                elif self.level == "xxT":
                    Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]
                    # Q["x", "x"] += Qi
                    Q["t", "t"] += Qi[: self.d, : self.d]
                    Q["t", "c"] += Qi[: self.d, self.d :]
                    Q["c", "c"] += Qi[self.d :, self.d :]
            else:
                # Q["x", "x"] += Qi
                Q["t", "t"] += Qi[: self.d, : self.d]
                Q["t", "c"] += Qi[: self.d, self.d :]
                Q["c", "c"] += Qi[self.d :, self.d :]
        if output_poly:
            return 0.5 * Q
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse

    def __repr__(self):
        appendix = "_robust" if self.robust else ""
        return f"mono_{self.d}d_{self.level}_{self.param_level}{appendix}"
