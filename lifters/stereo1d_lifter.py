import numpy as np

from lifters.state_lifter import StateLifter


class Stereo1DLifter(StateLifter):
    PARAM_LEVELS = ["no", "p"]

    def __init__(self, n_landmarks, param_level="no"):
        self.n_landmarks = n_landmarks
        self.d = 1
        self.W = 1.0
        super().__init__(param_level=param_level)

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    def generate_random_setup(self):
        self.landmarks = np.random.rand(self.n_landmarks)
        self.parameters = self.get_parameters()

    def generate_random_theta(self):
        self.theta = self.sample_theta()

    def sample_theta(self):
        x_try = np.random.rand(1)
        counter = 0
        while np.min(np.abs(x_try - self.landmarks)) <= 1e-3:
            x_try = np.random.rand(1)
            if counter >= 1000:
                print("Warning: couldn't find valid setup")
                return
        return x_try

    def sample_parameters(self, theta=None):
        if self.param_level == "p":
            parameters_ = np.r_[1.0, np.random.rand(self.n_landmarks)]
        elif self.param_level == "no":
            parameters_ = [1.0]
        return parameters_

    def get_parameters(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        indices = self.get_variable_indices(var_subset)
        if self.param_level == "p":
            return np.r_[1.0, self.landmarks[indices]]
        elif self.param_level == "no":
            return np.array([1.0])

    def get_x(self, theta=None, parameters=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters

        if var_subset is None:
            var_subset = self.var_dict.keys()

        x_data = []
        for key in var_subset:
            if key == "h":
                x_data.append(1.0)
            elif key == "x":
                x_data.append(float(theta[0]))
            elif "z" in key:
                idx = int(key.split("_")[-1])
                if self.param_level == "p":
                    x_data.append(float(1 / (theta[0] - parameters[idx + 1])))
                elif self.param_level == "no":
                    x_data.append(float(1 / (theta[0] - self.landmarks[idx])))
            else:
                raise ValueError("unknown key in get_x", key)
        return np.array(x_data)

    def get_p(self, parameters=None, var_subset=None):
        """
        :param parameters: list of all parameters
        :param var_subset: subset of variables tat we care about (will extract corresponding parameters)
        """
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        if self.param_level == "no":
            return np.array([1.0])

        landmarks = self.get_variable_indices(var_subset)
        if len(landmarks):
            sub_p = np.r_[1.0, parameters[1:][landmarks]]
            if self.param_level == "p":
                return sub_p
            else:
                raise ValueError(self.param_level)
        else:
            return np.array([1.0])

    @property
    def var_dict(self):
        vars = ["h", "x"] + [f"z_{j}" for j in range(self.n_landmarks)]
        return {v: 1 for v in vars}

    def get_param_idx_dict(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        param_dict_ = {"h": 0}
        if self.param_level == "no":
            return param_dict_

        indices = self.get_variable_indices(var_subset)
        for n in indices:
            param_dict_[f"p_{n}"] = n + 1
        return param_dict_

    def get_Q(self, noise: float = 1e-3) -> tuple:
        from poly_matrix.least_squares_problem import LeastSquaresProblem

        y = 1 / (self.theta - self.landmarks) + np.random.normal(
            scale=noise, loc=0, size=self.n_landmarks
        )

        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({"h": -y[j], f"z_{j}": 1})
        return ls_problem.get_Q().get_matrix(self.var_dict), y

    def get_A_known(self, add_known_redundant=False):
        from poly_matrix.poly_matrix import PolyMatrix

        # if self.add_parameters:
        #    raise ValueError("can't extract known matrices yet when using parameters.")

        A_known = []

        # enforce that z_j = 1/(x - a_j) <=> 1 - z_j*x + a_j*z_j = 0
        for j in range(self.n_landmarks):
            A = PolyMatrix()
            A["h", f"z_{j}"] = 0.5 * self.landmarks[j]
            A["x", f"z_{j}"] = -0.5
            A["h", "h"] = 1.0
            A_known.append(A.get_matrix(variables=self.var_dict))

        if not add_known_redundant:
            return A_known

        # add known redundant constraints:
        # enforce that z_j - z_i = (a_j - a_i) * z_j * z_i
        for i in range(self.n_landmarks):
            for j in range(i + 1, self.n_landmarks):
                A = PolyMatrix()
                A["h", f"z_{j}"] = 1
                A["h", f"z_{i}"] = -1
                A[f"z_{i}", f"z_{j}"] = self.landmarks[i] - self.landmarks[j]
                A_known.append(A.get_matrix(variables=self.var_dict))
        return A_known

    def get_cost(self, t, y):
        W = self.W
        return np.sum((y - (1 / (t - self.landmarks))) ** 2)

    def local_solver(
        self, t_init, y, num_iters=100, eps=1e-5, W=None, verbose=False, **kwargs
    ):
        a = self.landmarks
        x_op = t_init
        for i in range(num_iters):
            u = y - (1 / (x_op - a))
            if verbose:
                print(f"cost {i}", np.sum(u**2))
            du = 1 / ((x_op - a) ** 2)
            if np.linalg.norm(du) > 1e-10:
                dx = -np.sum(u * du) / np.sum(du * du)
                x_op = x_op + dx
                if np.abs(dx) < eps:
                    msg = f"converged dx after {i} it"
                    return x_op, msg, self.get_cost(x_op, y)
            else:
                msg = f"converged in du after {i} it"
                return x_op, msg, self.get_cost(x_op, y)
        return None, "didn't converge", None

    def __repr__(self):
        return f"stereo1d_{self.param_level}"
