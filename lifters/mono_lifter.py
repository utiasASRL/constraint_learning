from copy import deepcopy

import matplotlib
import matplotlib.pylab as plt

matplotlib.use("TkAgg")
plt.ion()

import autograd.numpy as np

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.geometry import (
    get_C_r_from_theta,
    get_C_r_from_xtheta,
    get_xtheta_from_C_r,
    get_xtheta_from_theta,
)

NOISE = 1e-1
N_OUTLIERS = 0

MAX_DIST = 2.0 # maximum distance of camera from landmarks
FOV = np.pi / 2 # camera field of view

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7,
    max_iterations=10000,
    min_step_size=1e-8,
    verbosity=1
)

# TODO(FD) we need to add a penalty here, otherwise the local solution is not good.
# However, the penalty results in inequality constraints etc. and that's not easy to deal with.
ADD_PENALTY = True
PENALTY_RHO = 10
PENALTY_U = 1e-3

def h_list(t):
    """
        We want to inforce that 
        - tan(a/2)*t3 >= sqrt(t1**2 + t2**2) or t3 >= 1
        - norm(t) <= 10 
        as constraints h_j(t)<=0
    """
    return [
        np.sqrt(np.sum(t[:-1]**2)) - np.tan(FOV/2)*t[-1],
        #-t[-1]+1,
        np.sqrt(np.sum(t**2)) - MAX_DIST
    ]

def penalty(t, rho=PENALTY_RHO, u=PENALTY_U):
    try:
        return np.sum([rho * u * np.log10(1 + np.exp(hi / u)) for hi in h_list(t)])
    except RuntimeWarning:
        PENALTY_U *= 0.1
        u = PENALTY_U
        return np.sum([rho * u * np.log10(1 + np.exp(hi / u)) for hi in h_list(t)])
        

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
        self, level="no", param_level="no", d=2, n_landmarks=3, variable_list=None, robust=False
    ):
        """
        :param level:
            - xwT: x kron w
            - xxT: x kron x
        """
        self.beta = 1.0
        self.n_landmarks = n_landmarks

        self.robust = robust
        self.level = level
        if variable_list == "all":
            variable_list = self.get_all_variables()

        if not robust:
            assert level == "no"
        super().__init__(
            level=level, param_level=param_level, d=d, variable_list=variable_list
        )

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        n = self.d**2 + self.d
        var_dict = {"l": 1, "x": n}
        if not self.robust:
            return var_dict
        var_dict.update({f"w_{i}": 1 for i in range(self.n_landmarks)})
        if self.level == "xwT":
            var_dict.update({f"z_{i}": n for i in range(self.n_landmarks)})
        elif self.level == "xxT":
            var_dict.update({"z_0": n**2})
        return var_dict

    def get_all_variables(self):
        all_variables = ["l", "x"] 
        if self.robust:
            all_variables += [f"w_{i}" for i in range(self.n_landmarks)]
        if self.level == "xxT":
            all_variables.append("z_0")
        elif self.level == "xwT":
            all_variables += [f"z_{i}" for i in range(self.n_landmarks)]
        variable_list = [all_variables]
        return variable_list

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

        # generate a random pose that is looking at world centre (where landmarks are)
        pc_cw = np.random.rand(self.d)*0.1
        pc_cw[self.d - 1] = np.random.uniform(1, MAX_DIST)

        n_angles = self.d * (self.d - 1) // 2
        angles = np.random.uniform(0, 2*np.pi, size=n_angles)
        if self.robust:
            w = [-1] * N_OUTLIERS + [1.0] * (self.n_landmarks - N_OUTLIERS)
            return np.r_[pc_cw, angles, w]
        return np.r_[pc_cw, angles]

    def sample_theta(self):
        """Sample a new feasible theta."""
        theta = self.generate_random_theta()
        if self.robust:
            w = np.random.choice([-1, 1], size=self.n_landmarks)#[-1] * N_OUTLIERS + [1.0] * (self.n_landmarks - N_OUTLIERS)
            theta[-len(w):] = w
        return theta

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

        if self.robust:
            theta_here = theta[: -self.n_landmarks]
        else:
            theta_here = theta

        if (self.d == 2) and len(theta_here) == 6:  # x, y, vec(C)
            R, t = get_C_r_from_xtheta(theta_here, self.d)
        elif (self.d == 2) and len(theta_here) == 3:  # x, y, alpha
            R, t = get_C_r_from_theta(theta_here, self.d)
        elif (self.d == 3) and len(theta_here) == 12:  # x, y, z, vec(C)
            R, t = get_C_r_from_xtheta(theta_here, self.d)
        elif (self.d == 3) and len(theta_here) == 6:  # x, y, z, alpha
            R, t = get_C_r_from_theta(theta_here, self.d)
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
        return self.extract_parameters(var_subset, self.landmarks)

    def get_A_known(self, output_poly=False):
        A_list = []
        if self.robust:
            for i in range(self.n_landmarks):
                Ai = PolyMatrix(symmetric=True)
                Ai["l", "l"] = -1.0
                Ai[f"w_{i}", f"w_{i}"] = 1.0
                if output_poly:
                    A_list.append(Ai)
                else:
                    A_list.append(Ai.get_matrix(self.var_dict))
        return A_list

    def get_B_known(self):
        """ Get inequality constraints of the form x.T @ B @ x >= 0"""
        dim_x = self.d + self.d**2

        # enforce that norm(t) <= MAX_DIST 
        B1 = PolyMatrix(symmetric=True)
        constraint = np.zeros((dim_x, dim_x))
        constraint[range(self.d), range(self.d)] = 1.0
        B1["l", "l"] = -MAX_DIST
        B1["x", "x"] = constraint

        # enforce that tan(FOV/2)*t3 >= sqrt(t1**2 + t2**2)
        # tan**2(FOV/2)*t**2 - t1**2 - t2**2 >= 0
        B3 = PolyMatrix(symmetric=True)
        constraint = np.zeros((dim_x, dim_x))
        constraint[range(self.d-1), range(self.d-1)] = 1.0
        constraint[self.d-1, self.d-1] = -np.tan(FOV / 2)**2
        B3["x", "x"] = constraint 
        
        # t3 >= 0
        constraint = np.zeros(dim_x)
        constraint[self.d-1] = -1
        B2 = PolyMatrix(symmetric=True)
        B2["l", "x"] = constraint[None, :]
        return [B1.get_matrix(self.var_dict), B2.get_matrix(self.var_dict), B3.get_matrix(self.var_dict)]

        # simplified: enforce that t[2] >= 1
        B2 = PolyMatrix(symmetric=True)
        constraint = np.zeros(dim_x)
        constraint[self.d-1] = -1.0
        B2["l","l"] = 1.0
        B2["l","x"] = constraint[None, :]
        return [B1.get_matrix(self.var_dict), B2.get_matrix(self.var_dict)]


    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        if self.robust:
            n_rot = self.d * (self.d - 1) // 2
            theta_x = deepcopy(self.theta[: self.d + n_rot])
            theta_x += np.random.normal(size=theta_x.shape, scale=delta)
            theta_w = self.theta[self.d + n_rot :]
            return np.r_[get_xtheta_from_theta(theta_x, self.d), theta_w]
        else:
            theta_x = deepcopy(self.theta)
            theta_x += np.random.normal(size=theta_x.shape, scale=delta)
            return get_xtheta_from_theta(theta_x, self.d)
            

    def residual(self, R, t, pi, ui):
        W = np.eye(self.d) - np.outer(ui, ui)
        return (R @ pi + t).T @ W @ (R @ pi + t)

    def get_cost(self, theta, y):
        if self.robust:
            x = theta[: -self.n_landmarks]
            w = theta[-self.n_landmarks :]
            assert np.all(w**2 == 1.0)
        else:
            x = theta

        try:
            R, t = get_C_r_from_theta(x, self.d)
        except:
            R, t = get_C_r_from_xtheta(x, self.d)

        cost = 0
        for i in range(self.n_landmarks):
            assert abs(np.linalg.norm(y[i]) - 1.0) < 1e-10
            res = self.residual(R, t, self.landmarks[i], y[i])
            if self.robust:
                cost += (1 + w[i]) / self.beta**2 * res + 1 - w[i]
            else:
                cost += res
        return 0.5 * cost 

    def get_grad(self, t, y):
        raise NotImplementedError("get_grad not implement yet")

    def get_hess(self, t, y):
        raise NotImplementedError("get_hess not implement yet")

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
            ui /= ui[self.d-1]

            ui[:self.d - 1] += np.random.normal(scale=noise, loc=0, size=self.d-1)
            assert ui[self.d-1] == 1.0
            ui /= np.linalg.norm(ui)
            y[i] = ui

        Q = self.get_Q_from_y(y)
        return Q, y

    def get_Q_from_y(self, y):
        from poly_matrix.poly_matrix import PolyMatrix

        Q = PolyMatrix(symmetric=True)

        for i in range(self.n_landmarks):
            pi = self.landmarks[i]
            ui = y[i]
            Pi = np.c_[np.eye(self.d), np.kron(pi, np.eye(self.d))]
            Wi = np.eye(self.d) - np.outer(ui, ui)
            Qi = Pi.T @ Wi @ Pi / self.beta**2
            if self.robust:
                Qi /= self.beta**2
                Q["l", "l"] += 1
                Q["l", f"w_{i}"] += -0.5
                if self.level == "xwT":
                    Q[f"z_{i}", "x"] += 0.5 * Qi
                    Q[f"x", "x"] += Qi
                elif self.level == "xxT":
                    Q["z_0", f"w_{i}"] += 0.5 * Qi.flatten()[:, None]
                    Q[f"x", "x"] += Qi
            else:
                Q[f"x", "x"] += Qi
        Q_sparse = 0.5 * Q.get_matrix(variables=self.var_dict)
        return Q_sparse

    def local_solver(self, t0, y, verbose=False, method=METHOD, solver_kwargs=SOLVER_KWARGS):
        import pymanopt
        from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product
        if method == "CG":
            from pymanopt.optimizers import ConjugateGradient as Optimizer # fastest
        elif method == "SD":
            from pymanopt.optimizers import SteepestDescent as Optimizer # slow
        elif method == "TR":
            from pymanopt.optimizers import TrustRegions as Optimizer # okay
        else:
            raise ValueError(method)
        
        if verbose:
            solver_kwargs["verbosity"] = 2 

        # We assume that we know w! If we wanted to solve for w too we would need
        # IRLS or similar. Since we just care about getting the global solution
        # with a local sovler that's not necessary.
        w = self.theta[-self.n_landmarks :]

        manifold = Product((SpecialOrthogonalGroup(self.d, k=1), Euclidean(self.d)))

        @pymanopt.function.autograd(manifold)
        def cost(R, t):
            cost = 0
            for i in range(self.n_landmarks):
                residual = self.residual(R, t, self.landmarks[i], y[i])
                if self.robust:
                    cost += (1 + w[i]) / self.beta**2 * residual + 1 - w[i]
                else:
                    cost += residual
            if ADD_PENALTY: 
                return 0.5 * cost + penalty(t)
            else:
                return 0.5 * cost 

        @pymanopt.function.autograd(manifold)
        def euclidean_gradient(R, t):
            grad_R = np.zeros(R.shape)
            grad_t = np.zeros(t.shape)
            for i in range(self.n_landmarks):
                Wi = np.eye(self.d) - np.outer(y[i], y[i])
                # residual = (R @ pi + t).T @ Wi @ (R @ pi + t)
                pi_cam = R @ self.landmarks[i] + t
                if self.robust:
                    grad_R += (
                        2
                        * w[i]
                        / self.beta**2
                        * np.outer(Wi.T @ pi_cam, self.landmarks[i])
                    )
                    grad_t += 2 * w[i] / self.beta**2 * Wi.T @ pi_cam
                else:
                    grad_R += np.outer(Wi.T @ pi_cam, self.landmarks[i])
                    grad_t += Wi.T @ pi_cam
            return grad_R, grad_t

        if ADD_PENALTY:
            euclidean_gradient = None        
        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient  #
        )
        optimizer = Optimizer(**solver_kwargs)

        R_0, t_0 = get_C_r_from_xtheta(t0[: self.d + self.d**2], self.d)
        res = optimizer.run(problem, initial_point=(R_0, t_0))
        R, t = res.point
        if self.robust:
            theta_hat = np.r_[get_xtheta_from_C_r(R, t), w]
        else:
            theta_hat = get_xtheta_from_C_r(R, t)

        cost_penalized = res.cost
        if ADD_PENALTY:
            pen = penalty(t)
            if abs(res.cost) > 1e-10:
                assert abs(pen)/res.cost <= 1e-1, (pen, res.cost)
            cost_penalized -= pen
        return theta_hat, res.stopping_criterion, cost_penalized

    def __repr__(self):
        appendix = "_robust" if self.robust else ""
        return f"mono_{self.d}d_{self.level}_{self.param_level}{appendix}"