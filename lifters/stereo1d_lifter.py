import numpy as np

from lifters.state_lifter import StateLifter


class Stereo1DLifter(StateLifter):
    def __init__(self, n_landmarks):
        self.n_landmarks = n_landmarks
        self.d = 1
        self.W = 1.0
        self.theta_ = None
        super().__init__(theta_shape=(1,), M=n_landmarks)

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.sample_theta()
        return self.theta_

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        self.landmarks = np.random.rand(self.n_landmarks)

    def sample_theta(self):
        x_try = np.random.rand(1)
        counter = 0
        while np.min(np.abs(x_try - self.landmarks)) <= 1e-3:
            x_try = np.random.rand(1)
            if counter >= 1000:
                print("Warning: couldn't find valid setup")
                return
        return x_try

    def get_x(self, theta=None, var_subset=None):
        """
        :param var_subset: list of variables to include in x vector. Set to None for all.
        """
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.var_dict.keys()

        x_data = []
        for key in var_subset:
            if key == "l":
                x_data.append(1.0)
            elif key == "x":
                x_data.append(float(theta))
            elif "z" in key:
                idx = int(key.split("_")[-1])
                x_data.append(float(1 / (theta - self.landmarks[idx])))
            else:
                raise ValueError("unknown key in get_x", key)
        return np.array(x_data)

    @property
    def var_dict(self):
        vars = ["l", "x"] + [f"z_{j}" for j in range(self.n_landmarks)]
        return {v: 1 for v in vars}

    def get_grad(self, t, y):
        raise NotImplementedError("get_grad not implement yet")

    def get_Q(self, noise: float = 1e-3) -> tuple:
        from poly_matrix.least_squares_problem import LeastSquaresProblem

        y = 1 / (self.theta - self.landmarks) + np.random.normal(
            scale=noise, loc=0, size=self.n_landmarks
        )

        ls_problem = LeastSquaresProblem()
        for j in range(len(y)):
            ls_problem.add_residual({"l": -y[j], f"z_{j}": 1})
        return ls_problem.get_Q().get_matrix(self.var_dict), y

    def get_A_known(self, add_known_redundant=False):
        from poly_matrix.poly_matrix import PolyMatrix

        A_known = []

        # enforce that z_j = 1/(x - a_j) == 1 <=> 1 - z_j*x + a_j*z_j = 0
        for j in range(self.n_landmarks):
            A = PolyMatrix()
            A["l", f"z_{j}"] = 0.5 * self.landmarks[j]
            A["x", f"z_{j}"] = -0.5
            A["l", "l"] = 1.0
            A_known.append(A.get_matrix(variables=self.var_dict))

        if not add_known_redundant:
            return A_known

        # add known redundant constraints:
        # enforce that z_j - z_i = (a_j - a_i) * z_j * z_i
        for i in range(self.n_landmarks):
            for j in range(i + 1, self.n_landmarks):
                A = PolyMatrix()
                A["l", f"z_{j}"] = 1
                A["l", f"z_{i}"] = -1
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
        return "stereo1d"
