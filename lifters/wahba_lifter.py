import matplotlib
import matplotlib.pylab as plt

import autograd.numpy as np

from lifters.robust_pose_lifter import RobustPoseLifter
from utils.geometry import get_C_r_from_theta

N_TRYS = 10
NOISE = 1e-2  # inlier noise
NOISE_OUT = 1.0  # outlier noise

# TODO(FD) for some reason this is not required as opposed to what is stated in Heng's paper
# and it currently breaks tightness (might be a bug in my implementation though)
USE_INEQ = False

NORMALIZE = False


class WahbaLifter(RobustPoseLifter):
    def h_list(self, t):
        """
        We want to inforce that
        - norm(t) <= 10 (default)
        as constraints h_j(t)<=0
        """
        default = super().h_list(t)
        return default

    def generate_random_setup(self):
        """Generate a new random setup. This is called once and defines the toy problem to be tightened."""
        self.theta  # makes sure to generate theta
        self.landmarks = np.random.normal(
            loc=0, scale=1.0, size=(self.n_landmarks, self.d)
        )
        self.parameters = np.r_[1.0, self.landmarks.flatten()]
        return

    def get_random_position(self):
        return np.random.uniform(
            -0.5 * self.MAX_DIST ** (1 / self.d),
            0.5 * self.MAX_DIST ** (1 / self.d),
            size=self.d,
        )

    def get_B_known(self):
        """Get inequality constraints of the form x.T @ B @ x <= 0"""
        if not USE_INEQ:
            return []

        default = super().get_B_known()
        return default

    def term_in_norm(self, R, t, pi, ui):
        return R @ pi + t - ui

    def residual_sq(self, R, t, pi, ui):
        # TODO: can easily extend below to matrix-weighted
        W = np.eye(self.d)
        res_sq = (R @ pi + t - ui).T @ W @ (R @ pi + t - ui)
        if NORMALIZE:
            return res_sq / (self.n_landmarks * self.d) ** 2
        return res_sq

    def sample_y(self, i, R, t, noise=NOISE):
        if noise is None:
            noise = NOISE
        pi = self.landmarks[i]
        # ui = deepcopy(pi) #R @ pi + t
        ui = R @ pi + t
        if i < self.n_outliers:
            ui += np.random.normal(scale=NOISE_OUT, loc=0, size=self.d)
        else:
            ui += np.random.normal(scale=noise, loc=0, size=self.d)
        return ui

    def get_Q(
        self, noise: float = None, output_poly: bool = False, use_cliques: list = []
    ):
        if self.y_ is None:
            n_angles = self.d * (self.d - 1) // 2
            theta = self.theta[: self.d + n_angles]
            self.y_ = np.empty((self.n_landmarks, self.d))
            R, t = get_C_r_from_theta(theta, self.d)
            for i in range(self.n_landmarks):
                valid_measurement = False
                while not valid_measurement:
                    y_i = self.sample_y(i, R, t, noise=noise)
                    residual = self.residual_sq(R, t, self.landmarks[i], y_i)
                    if i < self.n_outliers:
                        valid_measurement = residual > self.beta
                    else:
                        valid_measurement = residual < self.beta
                self.y_[i] = y_i
        Q = self.get_Q_from_y(self.y_, output_poly=output_poly, use_cliques=use_cliques)
        return Q, self.y_

    def get_Q_from_y(self, y, output_poly: bool = False, use_cliques: list = []):
        """
        every cost term can be written as
        (1 + wi)/b^2  r^2(x, zi) + (1 - wi)

        residual term:
        (Rpi + t - ui).T Wi (Rpi + t - ui) =
        [t', vec(R)'] @ [I (pi x I)]' @ Wi @ [I (pi x I)] @ [t ; vec(R)]
        ------x'-----   -----Pi'-----
        - 2 [t', vec(R)'] @ [I (pi x I)]' Wi @ ui
            -----x'------   ---------Pi_xl--------
        + ui.T @ Wi @ ui
        -----Pi_ll------
        """

        if len(use_cliques):
            js = use_cliques
        else:
            js = list(range(self.n_landmarks))

        from poly_matrix.poly_matrix import PolyMatrix

        Q = PolyMatrix(symmetric=True)
        if NORMALIZE:
            norm = (self.n_landmarks * self.d) ** 2

        Wi = np.eye(self.d)
        for i in js:
            pi = self.landmarks[i]
            ui = y[i]
            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]

            Pi_ll = ui.T @ Wi @ ui
            Pi_xl = -(Pi.T @ Wi @ ui)[:, None]
            Qi = Pi.T @ Wi @ Pi
            if NORMALIZE:
                Pi_ll /= norm
                Pi_xl /= norm
                Qi /= norm

            if self.robust:
                Qi /= self.beta**2
                Pi_ll /= self.beta**2
                Pi_xl /= self.beta**2
                # Q["x", "x"] += Qi
                Q["t", "t"] += Qi[: self.d, : self.d]
                Q["t", "c"] += Qi[: self.d, self.d :]
                Q["c", "c"] += Qi[self.d :, self.d :]

                # Q["x", "h"] += Pi_xl
                Q["t", "h"] += Pi_xl[: self.d, :]
                Q["c", "h"] += Pi_xl[self.d :, :]
                Q["h", "h"] += 1 + Pi_ll  # 1 from (1 - wi), Pi_ll from first term.
                Q["h", f"w_{i}"] += -0.5  # from (1 - wi), 0.5 cause on off-diagonal
                if self.level == "xwT":
                    # Q[f"z_{i}", "x"] += 0.5 * Qi
                    Q[f"z_{i}", "t"] += 0.5 * Qi[:, : self.d]
                    Q[f"z_{i}", "c"] += 0.5 * Qi[:, self.d :]

                    Q["h", f"w_{i}"] += 0.5 * Pi_ll  # from first term

                    Q[f"z_{i}", "h"] += Pi_xl
                elif self.level == "xxT":
                    Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]

                    # Q["x", f"w_{i}"] += Pi_xl
                    Q["t", f"w_{i}"] += Pi_xl[: self.d, :]
                    Q["c", f"w_{i}"] += Pi_xl[self.d :, :]

                    Q["h", f"w_{i}"] += 0.5 * Pi_ll
            else:
                # Q["x", "x"] += Qi
                Q["t", "t"] += Qi[: self.d, : self.d]
                Q["t", "c"] += Qi[: self.d, self.d :]
                Q["c", "c"] += Qi[self.d :, self.d :]

                # Q["x", "h"] += Pi_xl
                Q["t", "h"] += Pi_xl[: self.d, :]
                Q["c", "h"] += Pi_xl[self.d :, :]
                Q["h", "h"] += Pi_ll  # on diagonal
        if output_poly:
            return 0.5 * Q
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse

    def __repr__(self):
        appendix = "_robust" if self.robust else ""
        return f"wahba_{self.d}d_{self.level}_{self.param_level}{appendix}"
