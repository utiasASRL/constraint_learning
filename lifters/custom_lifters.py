import numpy as np

from lifters.state_lifter import StateLifter

def change_dimensions(a, y, x):
    p_w = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    y_mat = np.c_[[*y]]  # N x 2
    return p_w[:, :, None], y_mat[:, :, None], x[:, None]


def get_T(t: np.ndarray):
    """
    :param t: is composed of x, y, alpha
    """
    if len(t) == 6:
        T = np.zeros((3, 3))
        T[-1, -1] = 1.0
        T[:2, 0] = t[:2]
        T[:2, 1] = t[2:4]
        T[:2, -1] = t[-2:]
    else:
        raise NotImplementedError
    return T


def get_rot_matrix(rot):
    from scipy.spatial.transform import Rotation as R

    """ return the desired parameterization """
    if np.ndim(rot) <= 1:
        if np.ndim(rot) == 1:
            rot = float(rot)
        r = R.from_euler("z", rot)
        return r.as_matrix()[:2, :2]
    elif len(rot) == 2:
        r = R.from_euler("xyz", rot)
        return r.as_matrix()


class RangeOnlyLifter(StateLifter):
    def __init__(self, n_positions, d):
        self.n_positions = n_positions
        self.d = d
        super().__init__(theta_shape=(self.n_positions, d), M=n_positions)

    def generate_random_unknowns(self):
        self.unknowns = np.random.rand(self.n_positions, self.d)

    def get_theta(self):
        return self.unknowns

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()

        norms = np.linalg.norm(theta, axis=1) ** 2
        t = theta.flatten("C")  # generates t1, t2, ... with t1=[x1,y1]
        np.testing.assert_allclose(t[:2], theta[0, :])

        x = np.r_[1, t, norms]
        assert len(x) == self.N + self.M + 1
        return x


class Poly4Lifter(StateLifter):
    def __init__(self):
        super().__init__(theta_shape=(1,), M=1)

    def generate_random_unknowns(self):
        self.unknowns = np.random.rand(1)

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return np.r_[1, theta, theta**2]


class Poly6Lifter(StateLifter):
    def __init__(self):
        super().__init__(theta_shape=(1,), M=2)

    def generate_random_unknowns(self):
        self.unknowns = np.random.rand(1)

    def get_theta(self):
        return np.r_[self.unknowns]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        return np.r_[1, theta, theta**2, theta**3]


class PoseLandmarkLifter(StateLifter):
    def __init__(self, n_landmarks, n_poses, d):
        self.d = d
        self.n_poses = n_poses
        self.n_landmarks = n_landmarks

        # translation component
        N = (n_poses + n_landmarks) * d

        # number of paramters of rotation
        self.n_rot = int(d * (d - 1) / 2)
        N += n_poses * d**2

        M = n_poses * n_landmarks * d
        super().__init__(theta_shape=(N,), M=M)

    def generate_random_setup(self):
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_unknowns(self, ax=None):
        positions = np.random.rand(self.n_poses, self.d)
        angles = np.random.rand(self.n_poses, self.n_rot)
        matrices = [get_rot_matrix(angle) for angle in angles]
        if ax is not None:
            ax.scatter(*positions.T)

        self.unknowns = [positions, matrices]

    def get_theta(self):
        positions, matrices = self.unknowns
        r_vecs = np.array([mat.flatten("C") for mat in matrices]).flatten()
        return np.r_[positions.flatten("C"), r_vecs, self.landmarks.flatten("C")]

    def get_x(self, theta=None):
        """
        theta is tuple with elements (positions, matrices, landmarks)
        """
        import itertools

        if theta is None:
            theta = self.get_theta()

        x_data = [1] + list(theta)
        positions, matrices = self.unknowns

        # create substitutions of type:
        # y1_01 = C_01.T @ y0_01
        #       = rotation_1.T @ landmark_1
        for l, p in itertools.product(range(self.n_landmarks), range(self.n_poses)):
            w = matrices[p].T @ self.landmarks[l]
            x_data += list(w)
        x = np.array(x_data)
        assert len(x) == self.dim_X()
        return x


class Stereo1DLifter(StateLifter):
    def __init__(self, n_landmarks):
        self.n_landmarks = n_landmarks
        super().__init__(theta_shape=(1,), M=n_landmarks)

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        self.landmarks = np.random.rand(self.n_landmarks)

    def generate_random_unknowns(self):
        x_try = np.random.rand(1)
        counter = 0
        while np.min(np.abs(x_try - self.landmarks)) <= 1e-10:
            x_try = np.random.rand(1)
            if counter >= 1000:
                print("Warning: couldn't find valid setup")
                return
        self.unknowns = x_try

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

    def get_Q(self, noise=1e-3) -> np.ndarray:
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

    @staticmethod
    def get_cost(a, y, x):
        return np.sum((y - (1 / (x - a))) ** 2)

    @staticmethod
    def local_solver(a, y, x_init, num_iters=100, eps=1e-5, verbose=False):
        x_op = x_init
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
                    return x_op, msg
            else:
                msg = f"converged in du after {i} it"
                return x_op, msg
        return None, "didn't converge"


class Stereo2DLifter(StateLifter):
    def __init__(self, n_landmarks, level=0):
        self.level = level
        self.n_landmarks = n_landmarks
        self.d = 2
        M = self.n_landmarks * self.d
        if level > 0:
            M += self.n_landmarks * self.d
        super().__init__(theta_shape=(6,), M=M)

    def generate_random_setup(self):
        # important!! we can only resample x, the landmarks have to stay the same
        # [(x, y, alpha), (landmarks)]
        self.landmarks = np.random.rand(self.n_landmarks, self.d)

    def generate_random_unknowns(self):
        self.unknowns = np.r_[np.random.rand(2), np.random.rand() * 2 * np.pi]

    def get_theta(self):
        x, y, alpha = self.unknowns
        C = get_rot_matrix(alpha)
        C = get_rot_matrix(alpha)
        return np.r_[C[:, 0], C[:, 1], x, y]

    def get_x(self, theta=None):
        if theta is None:
            theta = self.get_theta()
        elif len(theta) == 3:  # theta actually contains unknowns
            x, y, alpha = theta
            C = get_rot_matrix(alpha)
            r = np.r_[x, y]
            theta = np.r_[C[:, 0], C[:, 1], x, y]

        C = np.c_[theta[:2], theta[2:4]]
        r = theta[4:]
        x_data = [1] + list(theta)

        higher_data = []
        for j in range(self.n_landmarks):
            pj = self.landmarks[j, :]
            zj = C[1, :] @ pj + r[1]
            u = 1 / zj * np.r_[C[0, :] @ pj + r[0], 1]
            x_data += list(u)

            if self.level > 0:
                higher_data += list(u**2)

        x_data += higher_data
        return np.array(x_data)

    def get_var_dict(self):
        var_dict = {"l": 1}
        var_dict["x"] = 6
        var_dict.update({f"z{i}": 2 for i in range(self.n_landmarks)})
        if self.level > 0:
            var_dict.update({f"y{i}": 2 for i in range(self.n_landmarks)})
        return var_dict

    def get_Q(self, noise: float = 1e-3) -> tuple:
        from poly_matrix.poly_matrix import PolyMatrix
        from stereo2d_problem import M as M_mat

        T = get_T(self.get_theta())

        y = []
        for j in range(self.n_landmarks):
            y_gt = T @ np.r_[self.landmarks[j, :], 1.0]
            y_gt /= y_gt[1]
            y_gt = M_mat @ y_gt
            y.append(y_gt + np.random.normal(loc=0, scale=noise))

        M_tilde = M_mat[:, [0, 2]]

        Q = PolyMatrix()
        M_tilde_sq = M_tilde.T @ M_tilde
        for j in range(len(y)):
            Q["l", "l"] += np.linalg.norm(y[j] - M_mat[:, 1]) ** 2
            Q[f"z{j}", "l"] += -(y[j] - M_mat[:, 1]).T @ M_tilde
            Q[f"z{j}", f"z{j}"] += M_tilde_sq
            # Q[f"y{j}", "l"] = 0.5 * np.diag(M_tilde_sq)
        return Q.toarray(self.get_var_dict()), y

    @staticmethod
    def get_inits(n_inits):
        return np.c_[
            np.random.rand(n_inits),
            np.random.rand(n_inits),
            2 * np.pi * np.random.rand(n_inits),
        ]

    @staticmethod
    def get_cost(a, y, x):
        from stereo2d_problem import _cost

        p_w, y, phi = change_dimensions(a, y, x)
        cost = _cost(p_w=p_w, y=y, phi=phi, W=None)[0, 0]
        return cost

    @staticmethod
    def local_solver(a, y, x_init, verbose=False):
        from stereo2d_problem import local_solver

        p_w, y, init_phi = change_dimensions(a, y, x_init)
        success, phi_hat, cost = local_solver(
            p_w=p_w, y=y, W=None, init_phi=init_phi, log=verbose
        )
        if success:
            return phi_hat.flatten(), "converged"
        else:
            return None, "didn't converge"
