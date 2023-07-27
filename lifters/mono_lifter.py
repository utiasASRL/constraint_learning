from copy import deepcopy

import matplotlib
import matplotlib.pylab as plt

matplotlib.use("TkAgg")
plt.ion()

import numpy as np

from lifters.state_lifter import StateLifter
from utils.geometry import (
    get_C_r_from_theta,
    get_C_r_from_xtheta,
    get_xtheta_from_C_r,
    get_xtheta_from_theta,
)

NOISE = 1e-1


class MonoLifter(StateLifter):
    LEVELS = ["no", "xwT", "xxT"]
    PARAM_LEVELS = ["no"]
    VARIALBE_LIST = [
        ["l", "x"],
        ["l", "x", "w_0"],
        ["l", "x", "z_0"],
        ["l", "x", "w_0", "w_1"],
        ["l", "x", "w_0", "z_0"],
        ["l", "x", "z_0", "z_1"],
    ]
    LEVEL_NAMES = {"no": "no", "xwT": "x kron w", "xxT": "x kron x"}

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
    def __init__(
        self, level="no", param_level="no", d=2, n_landmarks=3, variable_list=None
    ):
        """
        :param level:
            - xwT: x kron w
            - xxT: x kron x
        """
        self.beta = 1.0
        self.n_landmarks = n_landmarks
        super().__init__(
            level=level, param_level=param_level, d=d, variable_list=variable_list
        )

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        n = self.d**2 + self.d
        var_dict = {"l": 1, "x": n}
        var_dict.update({f"w_{i}": 1 for i in range(self.n_landmarks)})
        if self.level == "xwT":
            var_dict.update({f"z_{i}": n for i in range(self.n_landmarks)})
        elif self.level == "xxT":
            var_dict.update({"z_0": n**2})
        return var_dict

    def get_level_dims(self, n=1):
        """Return the dimension of the chosen lifting level, for n parameters"""
        return

    def generate_random_setup(self):
        """Generate a new random setup. This is called once and defines the toy problem to be tightened."""
        self.landmarks = np.random.rand(self.n_landmarks, self.d)
        self.parameters = np.r_[1.0, self.landmarks.flatten()]
        return

    def generate_random_theta(self):
        """Generate a random new feasible point, this is the ground truth."""
        from utils.geometry import generate_random_pose

        t = generate_random_pose(d=self.d)
        w = np.random.choice([-1, 1], size=self.n_landmarks)
        return np.r_[t, w]

    def sample_theta(self):
        """Sample a new feasible theta."""
        return self.generate_random_theta()

    def sample_parameters(self, x=None):
        """Sample new parameters, given x."""
        if self.param_level == "no":
            return [1.0]
        else:
            raise NotImplementedError("no parameters implement yet for mono.")

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        """Get the lifted vector x given theta and parameters."""
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict.keys()

        theta_here = theta[: -self.n_landmarks]
        if (self.d == 2) and len(theta_here) == 6:  # x, y, vec(C)
            R, t = get_C_r_from_xtheta(theta_here, self.d)
        elif (self.d == 2) and len(theta_here) == 3:  # x, y, alpha
            R, t = get_C_r_from_theta(theta_here, self.d)
        elif (self.d == 3) and len(theta_here) == 12:  # x, y, z, vec(C)
            R, t = get_C_r_from_xtheta(theta_here, self.d)
        elif (self.d == 3) and len(theta_here) == 6:  # x, y, z, alpha
            R, t = get_C_r_from_theta(theta_here, self.d)

        if self.param_level != "no":
            landmarks = np.array(parameters[1:]).reshape((self.n_landmarks, self.d))
        else:
            landmarks = self.landmarks

        x_data = []
        for key in var_subset:
            if key == "l":
                x_data.append(1.0)
            elif key == "x":
                x_data += list(get_xtheta_from_C_r(R, t))
            elif "w" in key:
                j = int(key.split("_")[-1])
                w_j = theta[-self.n_landmarks + j]
                x_data.append(w_j)

        if self.level == "no":
            pass
        elif self.level == "xxT":
            if "z_0" in var_subset:
                x_vec = list(get_xtheta_from_C_r(R, t))
                x_data += list(np.kron(x_vec, x_vec))
        elif self.level == "xwT":
            for key in var_subset:
                if "z" in key:
                    j = int(key.split("_")[-1])
                    w_j = theta[-self.n_landmarks + j]
                    x_vec = get_xtheta_from_C_r(R, t)
                    x_data += list(x_vec * w_j)
        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_parameters(self, var_subset=None) -> list:
        """Get the current paratmers given the (fixed) setup."""
        return self.extract_parameters(self, var_subset, self.landmarks)

    def residual(self, x, ui, pi):
        try:
            R, t = get_C_r_from_theta(x, self.d)
        except:
            R, t = get_C_r_from_xtheta(x, self.d)
        W = np.eye(self.d) - np.outer(ui, ui)
        return (R @ pi + t).T @ W @ (R @ pi + t)

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        n_rot = self.d * (self.d - 1) // 2
        theta_x = deepcopy(self.theta[: self.d + n_rot])
        theta_x += np.random.normal(size=theta_x.shape, scale=delta)
        theta_w = self.theta[self.d + n_rot :]
        return np.r_[get_xtheta_from_theta(theta_x, self.d), theta_w]

    def get_cost(self, theta, y):
        x = theta[: -self.n_landmarks]
        w = theta[-self.n_landmarks :]
        assert np.all(w**2 == 1.0)
        cost = 0
        for i in range(self.n_landmarks):
            assert abs(np.linalg.norm(y[i]) - 1.0) < 1e-10
            res = self.residual(x, y[i], self.landmarks[i])
            cost += (1 + w[i]) / self.beta**2 * res + 1 - w[i]
        return 0.5 * cost

    def get_grad(self, t, y):
        raise NotImplementedError("get_grad not implement yet")

    def get_hess(self, t, y):
        raise NotImplementedError("get_hess not implement yet")

    def get_Q(self, noise: float = None):
        if noise is None:
            noise = NOISE
        from poly_matrix.poly_matrix import PolyMatrix

        Q = PolyMatrix(symmetric=True)

        y = np.empty((self.n_landmarks, self.d))
        theta = self.theta[: -self.n_landmarks]
        R, t = get_C_r_from_theta(theta, self.d)
        for i in range(self.n_landmarks):
            pi = self.landmarks[i]
            # ui = deepcopy(pi) #R @ pi + t
            ui = R @ pi + t
            ui /= np.linalg.norm(ui)
            ui += np.random.normal(scale=noise, size=self.d)
            ui /= np.linalg.norm(ui)
            y[i] = ui

            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]
            Wi = np.eye(self.d) - np.outer(ui, ui)
            Qi = Pi.T @ Wi @ Pi / self.beta**2

            Q["l", "l"] += 1
            Q["l", f"w_{i}"] += -0.5
            if self.level == "xwT":
                Q[f"z_{i}", "x"] += 0.5 * Qi
                Q[f"x", "x"] += Qi
            elif self.level == "xxT":
                Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]
                Q[f"x", "x"] += Qi
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse, y

    def local_solver(self, t0, y, verbose=False, solver_kwargs={}):
        import pymanopt
        from pymanopt.optimizers import (
            SteepestDescent,
            TrustRegions,
            ParticleSwarm,
            ConjugateGradient,
        )
        from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product

        # We assume that we know w! If we wanted to solve for w too we would need
        # IRLS or similar. Since we just care about getting the global solution
        # with a local sovler that's not necessary.
        w = self.theta[-self.n_landmarks :]

        manifold = Product((SpecialOrthogonalGroup(self.d, k=1), Euclidean(self.d)))

        @pymanopt.function.autograd(manifold)
        def cost(R, t):
            cost = 0
            for i in range(self.n_landmarks):
                Wi = np.eye(self.d) - np.outer(y[i], y[i])
                pi_cam = R @ self.landmarks[i] + t
                residual = pi_cam.T @ Wi @ pi_cam
                cost += (1 + w[i]) / self.beta**2 * residual + 1 - w[i]
            return 0.5 * cost

        @pymanopt.function.autograd(manifold)
        def euclidean_gradient(R, t):
            grad_R = np.zeros(R.shape)
            grad_t = np.zeros(t.shape)
            for i in range(self.n_landmarks):
                Wi = np.eye(self.d) - np.outer(y[i], y[i])
                # residual = (R @ pi + t).T @ Wi @ (R @ pi + t)
                pi_cam = R @ self.landmarks[i] + t
                grad_R += (
                    2
                    * w[i]
                    / self.beta**2
                    * np.outer(Wi.T @ pi_cam, self.landmarks[i])
                )
                grad_t += 2 * w[i] / self.beta**2 * Wi.T @ pi_cam
            return grad_R, grad_t

        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient  #
        )
        # optimizer = SteepestDescent()  # slow
        # optimizer = TrustRegions()  # fast
        # optimizer = ParticleSwarm() # super slow
        optimizer = ConjugateGradient(min_gradient_norm=1e-7)  # very fast

        R0, t0 = get_C_r_from_xtheta(t0[: -self.n_landmarks], self.d)
        res = optimizer.run(problem, initial_point=(R0, t0))
        R, t = res.point
        theta_hat = np.r_[get_xtheta_from_C_r(R, t), w]

        return theta_hat, res.stopping_criterion, res.cost

    def __repr__(self):
        return f"mono_{self.d}d_{self.level}_{self.param_level}"
