from copy import deepcopy

import matplotlib
import matplotlib.pylab as plt

matplotlib.use("TkAgg")
plt.ion()

import numpy as np

from lifters.state_lifter import StateLifter
from utils.geometry import get_C_r_from_theta, get_C_r_from_xtheta

NOISE = 1e-3

class MonoLifter(StateLifter):
    LEVELS = ["no", "xwT", "xxT"] 
    PARAM_LEVELS = ["no"]
    VARIALBE_LIST = [
        ["l", "x"],
        ["l", "x", "w_0"],
        ["l", "x", "w_0", "w_1"],
        ["l", "x", "w_0", "w_1", "w_2"],
    ]

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
    def __init__(self, level="no", param_level="no", d=2, n_landmarks=3, variable_list=None):
        """
        :param level:
            - xwT: x kron w 
            - xxT: x kron x
        """
        self.beta = 1.0
        self.n_landmarks = n_landmarks
        super().__init__(level=level, param_level=param_level, d=d, variable_list=variable_list)

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        n = self.d**2 + self.d
        var_dict = {"l": 1, "x": n}
        var_dict.update({f"w_{i}": 1 for i in range(self.n_landmarks)})
        if self.level == "xwT":
            var_dict.update({f"z_{i}": n for i in range(self.n_landmarks)})
        elif self.level == "xxT":
            var_dict.update({"z": n**2})
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
        """Generate a random new feasible point, this is the ground truth. """
        from utils.geometry import generate_random_pose
        x = generate_random_pose(d=self.d) # t, r
        t = np.random.choice([-1, 1], size=self.n_landmarks)
        return np.r_[x, t]

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

        # TODO(FD) below is a bit hacky, these two variables should not both be called theta.
        theta_here = theta[:-self.n_landmarks]

        # theta is either (x, y, alpha) or (x, y, z, a1, a2, a3)
        if len(theta_here) in [3, 6]:
            R, t = get_C_r_from_theta(theta_here, self.d)
        # theta is (x, y, z, C.flatten()), technically this should be called xtheta!
        elif len(theta_here) == 12:
            R, t = get_C_r_from_xtheta(theta_here, self.d)
        else:
            raise ValueError(theta_here)

        if self.param_level != "no":
            landmarks = np.array(parameters[1:]).reshape((self.n_landmarks, self.d))
        else:
            landmarks = self.landmarks

        x_data = []
        for key in var_subset:
            if key == "l":
                x_data.append(1.0)
            elif key == "x":
                x_data += list(t) + list(R.flatten("F"))  # column-wise flatten
            elif "w" in key:
                j = int(key.split("_")[-1])
                w_j = theta[-self.n_landmarks + j]
                x_data.append(w_j)

        if self.level == "no":
            pass
        elif self.level == "xxT":
            if "z" in var_subset:
                x_vec = list(t) + list(R.flatten("F"))
                x_data += list(np.kron(x_vec, x_vec))
        elif self.level == "xwT":
            for key in var_subset:
                if "z" in key:
                    j = int(key.split("_")[-1])
                    w_j = theta[-self.n_landmarks + j]
                    x_vec = np.r_[t, R.flatten("F")]
                    x_data += list(x_vec * w_j)
        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_parameters(self, var_subset=None) -> list:
        """Get the current paratmers given the (fixed) setup."""
        return self.extract_parameters(self, var_subset, self.landmarks)

    def residual(self, x, ui, pi):
        R, t = get_C_r_from_theta(x, self.d)
        W = np.eye(self.d) - np.outer(ui, ui)
        return (R @ pi + t).T @ W @ (R @ pi + t)

    def get_cost(self, theta, y):
        x = theta[:-self.n_landmarks]
        w = theta[-self.n_landmarks:]
        assert np.all(w**2 == 1.0)
        cost = 0
        for i in range(self.n_landmarks):
            assert abs(np.linalg.norm(y[i]) - 1.0) < 1e-10
            cost += (1 + w[i]) / self.beta**2 * self.residual(x, y[i], self.landmarks[i]) + 1 - w[i]
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
        theta = self.theta[:-self.n_landmarks]
        R, t = get_C_r_from_theta(theta, self.d)
        for i in range(self.n_landmarks):
            pi = self.landmarks[i]
            #ui = deepcopy(pi) #R @ pi + t
            ui = R @ pi + t
            ui /= np.linalg.norm(ui)
            ui += np.random.normal(scale=noise, size=self.d)
            ui /= np.linalg.norm(ui)
            y[i] = ui

            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]
            Wi = np.eye(self.d) - np.outer(ui, ui)
            Qi = Pi.T @ Wi @ Pi  / self.beta**2

            Q["l", "l"] += 1 
            Q["l", f"w_{i}"] += - 0.5
            if self.level == "xwT":
                Q[f"z_{i}", "x"] += 0.5 * Qi 
                Q[f"x", "x"] += Qi
            elif self.level == "xxT":
                Q["z", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]
                Q[f"x", "x"] += Qi
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict) 
        return Q_sparse, y

    def local_solver(self, t0, y, verbose=False, solver_kwargs={}):
        raise NotImplementedError("no local solver for mono yet.")

    def __repr__(self):
        return f"mono_{self.d}d_{self.level}_{self.param_level}"