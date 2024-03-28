from abc import ABC, abstractmethod
from copy import deepcopy

import autograd.numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.spatial.transform import Rotation as R

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.geometry import get_C_r_from_theta, get_noisy_pose, get_theta_from_C_r

N_TRYS = 10

METHOD = "CG"
SOLVER_KWARGS = dict(
    min_gradient_norm=1e-7, max_iterations=10000, min_step_size=1e-8, verbosity=1
)

# TODO(FD) we need to add a penalty here, otherwise the local solution is not good.
# However, the penalty results in inequality constraints etc. and that's not easy to deal with.
PENALTY_RHO = 10
PENALTY_U = 1e-3

# the cutoff parameter of least squares. If residuals are >= BETA, they are considered outliers.
BETA = 0.1


class RobustPoseLifter(StateLifter, ABC):
    LEVELS = ["no", "xwT", "xxT"]
    PARAM_LEVELS = ["no"]
    LEVEL_NAMES = {"no": "no", "xwT": "x kron w", "xxT": "x kron x"}
    MAX_DIST = 10.0  # maximum of norm of t.

    @property
    def VARIABLE_LIST(self):
        if not self.robust:
            return [["h", "t", "c"]]
        else:
            base = ["h", "t", "c"]
            return [
                base,
                base + ["w_0"],
                base + ["z_0"],
                base + ["w_0", "w_1"],
                base + ["w_0", "z_0"],
                base + ["z_0", "z_1"],
                # base + ["w_0", "w_1", "z_0"],
                # base + ["w_0", "w_1", "z_0", "z_1"],
            ]

    # Add any parameters here that describe the problem (e.g. number of landmarks etc.)
    def __init__(
        self,
        n_outliers,
        level="no",
        param_level="no",
        d=2,
        n_landmarks=3,
        variable_list=None,
        robust=False,
        beta=BETA,
    ):
        """
        :param level:
            - xwT: x kron w
            - xxT: x kron x
        """
        self.beta = beta
        self.n_landmarks = n_landmarks

        self.robust = robust
        self.level = level
        if variable_list == "all":
            variable_list = self.get_all_variables()
        # elif variable_list is None:
        #    self.variable_list = self.VARIABLE_LIST

        if not robust:
            assert level == "no"
        super().__init__(
            level=level,
            param_level=param_level,
            d=d,
            variable_list=variable_list,
            n_outliers=n_outliers,
            robust=robust,
        )

    def penalty(self, t, rho=PENALTY_RHO, u=PENALTY_U):
        try:
            return np.sum(
                [rho * u * np.log10(1 + np.exp(hi / u)) for hi in self.h_list(t)]
            )
        except RuntimeWarning:
            PENALTY_U *= 0.1
            u = PENALTY_U
            return np.sum(
                [rho * u * np.log10(1 + np.exp(hi / u)) for hi in self.h_list(t)]
            )

    @property
    def var_dict(self):
        """Return key,size pairs of all variables."""
        n = self.d**2 + self.d
        var_dict = {"h": 1, "t": self.d, "c": self.d**2}
        if not self.robust:
            return var_dict
        var_dict.update({f"w_{i}": 1 for i in range(self.n_landmarks)})
        if self.level == "xwT":
            var_dict.update({f"z_{i}": n for i in range(self.n_landmarks)})
        elif self.level == "xxT":
            var_dict.update({"z_0": n**2})
        return var_dict

    def get_all_variables(self):
        all_variables = ["h", "t", "c"]
        if self.robust:
            all_variables += [f"w_{i}" for i in range(self.n_landmarks)]
        if self.level == "xxT":
            all_variables.append("z_0")
        elif self.level == "xwT":
            all_variables += [f"z_{i}" for i in range(self.n_landmarks)]
        variable_list = [all_variables]
        return variable_list

    def sample_theta(self):
        """Sample a new feasible theta."""
        theta = self.generate_random_theta()
        if self.robust:
            w = np.random.choice([-1, 1], size=self.n_landmarks)
            theta[-len(w) :] = w
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

        R, t = get_C_r_from_theta(theta_here, self.d)

        x_data = []
        for key in var_subset:
            if key == "h":
                x_data.append(1.0)
            elif key == "t":
                x_data += list(t)
            elif key == "c":
                x_data += list(R.flatten("F"))
            elif "w" in key:
                j = int(key.split("_")[-1])
                w_j = theta[-self.n_landmarks + j]
                x_data.append(w_j)

        if self.level == "no":
            pass
        elif self.level == "xxT":
            if "z_0" in var_subset:
                x_vec = list(get_theta_from_C_r(R, t))
                x_data += list(np.kron(x_vec, x_vec).flatten())
        elif self.level == "xwT":
            for key in var_subset:
                if "z" in key:
                    j = int(key.split("_")[-1])
                    w_j = theta[-self.n_landmarks + j]
                    x_vec = get_theta_from_C_r(R, t)
                    x_data += list(x_vec * w_j)
        dim_x = self.get_dim_x(var_subset=var_subset)
        assert len(x_data) == dim_x
        return np.array(x_data)

    def get_parameters(self, var_subset=None) -> list:
        """Get the current paratmers given the (fixed) setup."""
        return self.extract_parameters(var_subset, self.landmarks)

    def generate_random_theta(self):
        """Generate a random new feasible point, this is the ground truth."""

        # generate a random pose that is looking at world centre (where landmarks are)
        success = False
        i = 0

        while not success:
            pc_cw = self.get_random_position()
            success = np.all(np.array(self.h_list(pc_cw)) <= 0)
            if success:
                break
            i += 1
            if i >= N_TRYS:
                raise ValueError("didn't find valid initialization")

        n_angles = self.d * (self.d - 1) // 2
        if self.d == 2:
            angle = np.random.uniform(0, 2 * np.pi)
            C = R.from_euler("z", angle).as_matrix()[:2, :2]
        else:
            C = R.random().as_matrix()
        theta_x = get_theta_from_C_r(C, pc_cw)
        if self.robust:
            # we always assume the first elements correspond to outliers
            # and the last elements to inliers.
            w = [-1] * self.n_outliers + [1.0] * (self.n_landmarks - self.n_outliers)
            return np.r_[theta_x, w]
        return theta_x

    def get_error(self, theta_hat):
        from utils.geometry import get_pose_errors_from_theta

        theta_hat_pose = theta_hat[: self.d + self.d**2]
        theta_gt_pose = self.theta[: self.d + self.d**2]
        return get_pose_errors_from_theta(theta_hat_pose, theta_gt_pose, self.d)

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        if self.robust:
            theta = deepcopy(self.theta[: self.d + self.d**2])
            C, r = get_C_r_from_theta(theta)
            theta_noisy = get_noisy_pose(C, r, delta=delta)
            theta_w = self.theta[self.d + self.d**2 :]
            return np.r_[theta_noisy, theta_w]
        else:
            C, r = get_C_r_from_theta(self.theta)
            theta_noisy = get_noisy_pose(C, r, delta=delta)
            return theta_noisy

    def get_cost(self, theta, y):
        if self.robust:
            x = theta[: -self.n_landmarks]
            w = theta[-self.n_landmarks :]
            assert np.all(w**2 == 1.0)
        else:
            x = theta

        R, t = get_C_r_from_theta(x, self.d)

        cost = 0
        for i in range(self.n_landmarks):
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

    def local_solver(
        self, t0, y, verbose=False, method=METHOD, solver_kwargs=SOLVER_KWARGS
    ):
        import pymanopt
        from pymanopt.manifolds import Euclidean, Product, SpecialOrthogonalGroup

        if method == "CG":
            from pymanopt.optimizers import ConjugateGradient as Optimizer  # fastest
        elif method == "SD":
            from pymanopt.optimizers import SteepestDescent as Optimizer  # slow
        elif method == "TR":
            from pymanopt.optimizers import TrustRegions as Optimizer  # okay
        else:
            raise ValueError(method)

        if verbose:
            solver_kwargs["verbosity"] = 2

        # We assume that we know w! If we wanted to solve for w too we would need
        # IRLS or similar. Since we just care about getting the global solution
        # with a local sovler that's not necessary.
        if self.robust:
            w = self.theta[-self.n_landmarks :]

        manifold = Product((SpecialOrthogonalGroup(self.d, k=1), Euclidean(self.d)))

        @pymanopt.function.autograd(manifold)
        def cost(R, t):
            cost = 0
            for i in range(self.n_landmarks):
                residual = self.residual_sq(R, t, self.landmarks[i], y[i])
                if self.robust:
                    cost += (1 + w[i]) / self.beta**2 * residual + 1 - w[i]
                else:
                    cost += residual
            return 0.5 * cost + self.penalty(t)

        @pymanopt.function.autograd(manifold)
        def euclidean_gradient(R, t):
            grad_R = np.zeros(R.shape)
            grad_t = np.zeros(t.shape)
            for i in range(self.n_landmarks):
                Wi = np.eye(self.d) - np.outer(y[i], y[i])
                # residual = (R @ pi + t).T @ Wi @ (R @ pi + t)
                term = self.term_in_norm(R, t, self.landmarks[i], y[i])
                if self.robust:
                    grad_R += (
                        2
                        * w[i]
                        / self.beta**2
                        * np.outer(Wi.T @ term, self.landmarks[i])
                    )
                    grad_t += 2 * w[i] / self.beta**2 * Wi.T @ term
                else:
                    grad_R += np.outer(Wi.T @ term, self.landmarks[i])
                    grad_t += Wi.T @ term
            return grad_R, grad_t

        euclidean_gradient = None  # set to None
        problem = pymanopt.Problem(
            manifold, cost, euclidean_gradient=euclidean_gradient  #
        )
        optimizer = Optimizer(**solver_kwargs)

        R_0, t_0 = get_C_r_from_theta(t0[: self.d + self.d**2], self.d)
        res = optimizer.run(problem, initial_point=(R_0, t_0))
        R, t = res.point

        print("local solver sanity check:")
        print("final penalty:", self.penalty(t))
        for i in range(self.n_landmarks):
            residual = self.residual_sq(R, t, self.landmarks[i], y[i])
            if i < self.n_outliers:
                print(f"outlier residual: {residual:.4e}")
                assert (
                    residual > self.beta
                ), f"outlier residual too small: {residual} <= {self.beta}"
            else:
                print(f"inlier residual: {residual:.4e}")
                assert (
                    residual <= self.beta
                ), f"inlier residual too large: {residual} > {self.beta}"

        if self.robust:
            theta_hat = np.r_[get_theta_from_C_r(R, t), w]
        else:
            theta_hat = get_theta_from_C_r(R, t)

        cost_penalized = res.cost
        if self.robust:
            pen = self.penalty(t)
            if abs(res.cost) > 1e-10:
                assert abs(pen) / res.cost <= 1e-1, (pen, res.cost)
            cost_penalized -= pen

        success = ("min step_size" in res.stopping_criterion) or (
            "min grad norm" in res.stopping_criterion
        )
        info = {
            "success": success,
            "msg": res.stopping_criterion,
        }
        if success:
            return theta_hat, info, cost_penalized
        else:
            return None, info, cost_penalized

    def test_and_add(self, A_list, Ai, output_poly):
        x = self.get_x()
        Ai_sparse = Ai.get_matrix(self.var_dict)
        err = x.T @ Ai_sparse @ x
        assert abs(err) <= 1e-10, err
        if output_poly:
            A_list.append(Ai)
        else:
            A_list.append(Ai_sparse)

    def get_A_known(self, var_dict=None, output_poly=False):
        A_list = []
        if var_dict is None:
            var_dict = self.var_dict

        if "c" in var_dict:
            # enforce diagonal
            for i in range(self.d):
                Ei = np.zeros((self.d, self.d))
                Ei[i, i] = 1.0
                constraint = np.kron(Ei, np.eye(self.d))
                Ai = PolyMatrix(symmetric=True)
                Ai["c", "c"] = constraint
                Ai["h", "h"] = -1
                self.test_and_add(A_list, Ai, output_poly=output_poly)

            # enforce off-diagonal
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    Ei = np.zeros((self.d, self.d))
                    Ei[i, j] = 1.0
                    Ei[j, i] = 1.0
                    constraint = np.kron(Ei, np.eye(self.d))
                    Ai = PolyMatrix(symmetric=True)
                    Ai["c", "c"] = constraint
                    self.test_and_add(A_list, Ai, output_poly=output_poly)
        if self.robust:
            for key in var_dict:
                if "w" in key:
                    i = key.split("_")[-1]
                    Ai = PolyMatrix(symmetric=True)
                    Ai["h", "h"] = -1.0
                    Ai[f"w_{i}", f"w_{i}"] = 1.0
                    self.test_and_add(A_list, Ai, output_poly=output_poly)

                    # below doesn't hold: w_i*w_j = += 1
                    # for key_other in [k for k in var_dict if (k.startswith("w") and (k!= key))]:
                    #    Ai = PolyMatrix(symmetric=True)
                    #    Ai["h", "h"] = -1.0
                    #    Ai[key, key_other] = 0.5
                    #    self.test_and_add(A_list, Ai, output_poly=output_poly)

                if "z" in key:
                    if self.level == "xwT":
                        i = key.split("_")[-1]
                        """ each z_i equals x * w_i"""

                        for j in range(self.d):
                            Ai = PolyMatrix(symmetric=True)
                            constraint = np.zeros((self.d + self.d**2))
                            constraint[j] = 1.0
                            Ai["h", f"z_{i}"] = constraint[None, :]
                            constraint = np.zeros((self.d))
                            constraint[j] = -1.0
                            Ai[f"t", f"w_{i}"] = constraint[:, None]
                            self.test_and_add(A_list, Ai, output_poly=output_poly)

                        for j in range(self.d**2):
                            Ai = PolyMatrix(symmetric=True)
                            constraint = np.zeros((self.d + self.d**2))
                            constraint[self.d + j] = 1.0
                            Ai["h", f"z_{i}"] = constraint[None, :]
                            constraint = np.zeros((self.d**2))
                            constraint[j] = -1.0
                            Ai[f"c", f"w_{i}"] = constraint[:, None]
                            self.test_and_add(A_list, Ai, output_poly=output_poly)
        return A_list

    def get_B_known(self):
        """Get inequality constraints of the form x.T @ B @ x <= 0.
        By default, we always add ||t|| <= MAX_DIST
        """
        B1 = PolyMatrix(symmetric=True)
        B1["h", "h"] = -self.MAX_DIST
        B1["t", "t"] = np.eye(self.d)
        return [B1.get_matrix(self.var_dict)]

    @abstractmethod
    def h_list(self, t):
        """
        Any inequality constraints to enforce, returned as a list [h_1(t), h_2(t), ...]
        We use the convention h_i(t) <= 0.

        By default, we always add |t| <= MAX_DIST
        """
        return [np.sqrt(np.sum(t[: self.d] ** 2)) - self.MAX_DIST]

    @abstractmethod
    def get_random_position(self):
        """Generate a new random position. Orientation angles will be drawn uniformly from [0, pi]."""
        return None

    @abstractmethod
    def generate_random_setup(self):
        """Generate a new random setup. This is called once and defines the toy problem to be tightened."""
        return

    @abstractmethod
    def term_in_norm(self, R, t, pi, ui):
        return

    @abstractmethod
    def residual_sq(self, R, t, pi, ui):
        return

    @abstractmethod
    def get_Q(self, noise: float = None):
        return

    @abstractmethod
    def get_Q_from_y(self, y):
        return

    @abstractmethod
    def __repr__(self):
        return
