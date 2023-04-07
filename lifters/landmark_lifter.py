import numpy as np

from lifters.state_lifter import StateLifter
from utils import get_rot_matrix


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

    def sample_feasible(self, ax=None):
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
        assert len(x) == self.dim_x
        return x
