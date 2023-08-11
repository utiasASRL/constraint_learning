import numpy as np

from lifters.state_lifter import StateLifter


class Stereo1DSLAMLifter(StateLifter):
    LEVELS = ["no", "za", "all"]

    def __init__(self, n_landmarks, level="no"):
        self.n_landmarks = n_landmarks
        self.d = 1
        self.W = 1.0
        self.theta_ = None
        self.level = level
        if self.level == "no":
            L = 0
        elif self.level == "za":
            L = self.n_landmarks**2
        elif self.level == "all":
            # z*a  + t*a + t*z + y_i_j * ak
            # y_i_j  ut_i  vt_i  + w_i_j_k
            L = (
                self.n_landmarks**2
                + 2 * self.n_landmarks
                + self.n_landmarks**2 * self.n_landmarks
            )
        super().__init__(theta_shape=(1 + n_landmarks,), M=n_landmarks, L=L)

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    def generate_random_setup(self):
        pass

    def sample_theta(self):
        # for SLAM, we also want to regenerate the landmarks!
        # self.generate_random_setup()

        theta = np.random.rand(1)
        landmarks = np.random.rand(self.n_landmarks)
        counter = 0
        while np.min(np.abs(theta - landmarks)) <= 1e-3:
            theta = np.random.rand(1)
            landmarks = np.random.rand(self.n_landmarks)
            counter += 1
            if counter >= 1000:
                print("Warning: couldn't find valid setup")
                return
        return np.r_[theta, landmarks]

    def get_x(self, theta=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict.keys()

        x, *landmarks = theta

        x_data = []
        for key in var_subset:
            if key == "h":
                x_data.append(1.0)
            elif key == "x":
                x_data.append(float(x))
            elif "a" in key:
                idx = int(key.split("_")[-1])
                x_data.append(float(landmarks[idx]))
            elif "z" in key:
                idx = int(key.split("_")[-1])
                zi = float(1 / (x - landmarks[idx]))
                x_data.append(zi)
            elif "y" in key:
                i, j = list(map(int, key.split("_")[-2:]))
                zi = float(1 / (x - landmarks[i]))
                aj = landmarks[j]
                x_data.append(zi * aj)
            elif "u" in key:
                idx = int(key.split("_")[-1])
                x_data.append(float(x * landmarks[idx]))
            elif "v" in key:
                idx = int(key.split("_")[-1])
                zi = float(1 / (x - landmarks[idx]))
                x_data.append(float(x * zi))
            elif "w" in key:
                i, j, k = list(map(int, key.split("_")[-3:]))
                zi = float(1 / (x - landmarks[i]))
                aj = landmarks[j]
                ak = landmarks[k]
                x_data.append(zi * aj * ak)
            else:
                raise ValueError("unknown key in get_x", key)
        return np.array(x_data)

    @property
    def var_dict(self):
        import itertools

        vars = ["h", "x"]
        vars += [f"a_{j}" for j in range(self.n_landmarks)]
        vars += [f"z_{j}" for j in range(self.n_landmarks)]
        if self.level == "za":
            vars += [
                f"y_{i}_{j}"
                for i, j in itertools.product(
                    range(self.n_landmarks), range(self.n_landmarks)
                )
            ]
        elif self.level == "all":
            vars += [
                f"y_{i}_{j}"
                for i, j in itertools.product(
                    range(self.n_landmarks), range(self.n_landmarks)
                )
            ]
            vars += [f"u_{i}" for i in range(self.n_landmarks)]
            vars += [f"v_{i}" for i in range(self.n_landmarks)]
            vars += [
                f"w_{i}_{j}_{k}"
                for i, j, k in itertools.product(
                    range(self.n_landmarks),
                    range(self.n_landmarks),
                    range(self.n_landmarks),
                )
            ]
        return {v: 1 for v in vars}

    def get_grad(self, t, y):
        raise NotImplementedError("get_grad not implement yet")

    def get_Q(self, noise: float = 1e-3) -> tuple:
        from poly_matrix.least_squares_problem import LeastSquaresProblem

        x, *landmarks = self.theta

        y = 1 / (x - landmarks) + np.random.normal(
            scale=noise, loc=0, size=self.n_landmarks
        )

        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({"h": -y[j], f"z_{j}": 1})
        return ls_problem.get_Q().get_matrix(self.var_dict), y

    def get_A_known(self, add_known_redundant=False):
        from poly_matrix.poly_matrix import PolyMatrix

        A_known = []

        # enforce that z_j = 1/(x - a_j) <=> 1 - z_j*x + a_j*z_j = 0
        for j in range(self.n_landmarks):
            A = PolyMatrix()
            A[f"a_{j}", f"z_{j}"] = 0.5
            A["x", f"z_{j}"] = -0.5
            A["h", "h"] = 1.0
            A_known.append(A.get_matrix(variables=self.var_dict))

        if True:  # not add_known_redundant:
            return A_known

        # TODO(FD) below constraint is not quadratic anymore!
        # add known redundant constraints:
        # enforce that z_j - z_i = (a_j - a_i) * z_j * z_i
        for i in range(self.n_landmarks):
            for j in range(i + 1, self.n_landmarks):
                A = PolyMatrix()
                A["h", f"z_{j}"] = 1
                A["h", f"z_{i}"] = -1
                A[f"z_{i}", f"z_{j}"] = self.landmarks[i] - landmarks[j]
                A_known.append(A.get_matrix(variables=self.var_dict))
        return A_known

    def get_cost(self, x, y):
        t, *landmarks = x
        W = self.W
        return np.sum((y - (1 / (t - landmarks))) ** 2)

    def local_solver(
        self, t_init, y, num_iters=100, eps=1e-5, W=None, verbose=False, **kwargs
    ):
        x_op, *landmarks = t_init
        for i in range(num_iters):
            u = y - (1 / (x_op - landmarks))
            if verbose:
                print(f"cost {i}", np.sum(u**2))
            du = 1 / ((x_op - landmarks) ** 2)
            if np.linalg.norm(du) > 1e-10:
                dx = -np.sum(u * du) / np.sum(du * du)
                x_op = x_op + dx
                t_op = np.r_[x_op, landmarks]
                if np.abs(dx) < eps:
                    msg = f"converged in dx after {i} it"
                    return t_op, msg, self.get_cost(t_op, y)
            else:
                msg = f"converged in du after {i} it"
                return t_op, msg, self.get_cost(t_op, y)
        return None, "didn't converge", None

    def __repr__(self):
        return "stereo1d_slam"
