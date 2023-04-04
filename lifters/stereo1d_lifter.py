import numpy as np

from lifters.state_lifter import StateLifter


class Stereo1DLifter(StateLifter):
    def __init__(self, n_landmarks):
        self.n_landmarks = n_landmarks
        self.d = 1
        self.W = 1.0
        super().__init__(theta_shape=(1,), M=n_landmarks)

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        self.landmarks = np.random.rand(self.n_landmarks)

    def generate_random_unknowns(self, replace=True):
        x_try = np.random.rand(1)
        counter = 0
        while np.min(np.abs(x_try - self.landmarks)) <= 1e-10:
            x_try = np.random.rand(1)
            if counter >= 1000:
                print("Warning: couldn't find valid setup")
                return
        if replace:
            self.unknowns = x_try
        return x_try

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()

        x = self.unknowns
        a = self.landmarks
        x_data = [1] + list(x)
        x_data += [float(1 / (x - ai)) for ai in a]
        return np.array(x_data)

    def get_var_dict(self):
        vars = ["l", "x"] + [f"z{i}" for i in range(self.n_landmarks)]
        return {v: 1 for v in vars}

    def get_Q(self, noise=1e-3) -> tuple:
        from poly_matrix.poly_matrix import PolyMatrix

        x = self.unknowns
        a = self.landmarks

        y = 1 / (x - a) + np.random.normal(scale=noise, loc=0, size=len(a))
        Q = PolyMatrix()
        for j in range(len(a)):
            Q["l", "l"] += y[j] ** 2
            Q["l", f"z{j}"] += -y[j]
            Q[f"z{j}", f"z{j}"] += 1
        return Q.toarray(self.get_var_dict()), y

    def get_A_known(self):
        from poly_matrix.poly_matrix import PolyMatrix

        A_known = []

        var_dict = self.get_var_dict()
        for j in range(self.n_landmarks):
            A = PolyMatrix()
            A["l", f"z{j}"] = 0.5 * self.landmarks[j]
            A["x", f"z{j}"] = -0.5
            A["l", "l"] = 1.0
            A_known.append(A.toarray(variables=var_dict))
        return A_known

    def get_cost(self, t, y):
        W = self.W
        a = self.landmarks
        return np.sum((y - (1 / (t - a))) ** 2)

    def local_solver(self, t_init, y, num_iters=100, eps=1e-5, W=None, verbose=False):
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
                    return x_op, msg, Stereo1DLifter.get_cost(a, y, x_op)
            else:
                msg = f"converged in du after {i} it"
                return x_op, msg, Stereo1DLifter.get_cost(a, y, x_op)
        return None, "didn't converge", None

    def __repr__(self):
        return "stereo1d"
