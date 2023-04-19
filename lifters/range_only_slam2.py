import numpy as np

from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
from poly_matrix.least_squares_problem import LeastSquaresProblem


class RangeOnlySLAM2Lifter(RangeOnlySLAM1Lifter):
    """Range-only SLAM, version 1

    Uses substitutions e_ik = ||a_k - t_i||
    """

    def __init__(self, n_positions, n_landmarks, d, edges=None):
        super().__init__(n_positions, n_landmarks, d, edges=edges)

    @property
    def M(self):
        return len(self.edges)

    @property
    def sub_var_dict(self):
        var_dict = {}
        var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
        return var_dict

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        positions, landmarks = self.get_positions_and_landmarks(theta)

        x_data = [[1]]
        x_data += [list(theta)]
        x_data += [
            [np.linalg.norm(landmarks[k] - positions[n]) ** 2] for n, k in self.edges
        ]
        x = np.concatenate(x_data, axis=0)
        assert len(x) == self.N + self.M + 1
        return x

    def get_Q_from_y(self, y):
        self.ls_problem = LeastSquaresProblem()
        for n, k in self.edges:
            self.ls_problem.add_residual({"l": y[n, k], f"e{n}{k}": -1})
        # fix Gauge freedom
        if self.remove_gauge == "cost":
            I = np.eye(self.d)
            for d in range(self.d):
                self.ls_problem.add_residual({"a0": I[d].reshape((1, -1))})
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

    def get_A_known(self):
        from poly_matrix.poly_matrix import PolyMatrix

        A_list = []
        for n, k in self.edges:
            A = PolyMatrix()
            A[f"x{n}", f"x{n}"] = np.eye(self.d)
            if self.remove_gauge == "hard":
                if k > 0:
                    A[f"a{k}", f"a{k}"] = np.eye(self.var_dict[f"a{k}"])
                    A[f"x{n}", f"a{k}"] = -np.eye(self.d)[:, : self.var_dict[f"a{k}"]]
            else:
                A[f"a{k}", f"a{k}"] = np.eye(self.d)
                A[f"x{n}", f"a{k}"] = -np.eye(self.d)
            A["l", f"e{n}{k}"] = -0.5
            A_list.append(A.get_matrix(self.var_dict))
        return A_list

    def get_J_lifting(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)

        J_lifting = np.zeros((self.M, self.N))
        for i, (n, k) in enumerate(self.edges):
            delta = landmarks[k] - positions[n]

            # grad w.r.t. position
            # d/dp_n|| ak - p_n || = -2(ak - p_n)
            J_lifting[i, n * self.d : (n + 1) * self.d] = -2 * delta

            if self.remove_gauge == "hard":
                # grad w.r.t. landmark d/d_pn ||a_k - p_n || = 2(ak - pn)
                self.fill_depending_on_k(J_lifting[i], k, 2 * delta)
            else:
                start = self.n_positions * self.d + k * self.d
                J_lifting[i, start : start + self.d] = 2 * delta
        return J_lifting

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for j, (n, k) in enumerate(self.edges):
            hessian = np.zeros((self.N, self.N))
            i = n * self.d

            # diagonal for position
            hessian[range(i, i + self.d), range(i, i + self.d)] = 2.0
            if self.remove_gauge == "hard":
                # diagonal for landmark
                self.fill_hessian_depending_on_k(hessian, k, val=2.0)  # along i
                # diagonal for position - landmark
                self.fill_hessian_depending_on_k(
                    hessian, k, fix_i=range(i, i + self.d), val=-2.0
                )
            else:
                j = (self.n_positions + k) * self.d
                hessian[range(i, i + self.d), range(i, i + self.d)] = 2
                # diagonal for landmark
                hessian[range(j, j + self.d), range(j, j + self.d)] = 2
                # off-diagonal position - landmark
                hessian[range(i, i + self.d), range(j, j + self.d)] = -2
                hessian[range(j, j + self.d), range(i, i + self.d)] = -2
            hessians.append(hessian)
        return hessians

    def __repr__(self):
        return f"rangeonlyslam2-{self.d}d"


if __name__ == "__main__":
    lifter = RangeOnlySLAM2Lifter(n_positions=3, n_landmarks=4, d=2)
    lifter.run(n_dual=1)
