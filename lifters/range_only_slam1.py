import matplotlib.pylab as plt
import numpy as np

from lifters.range_only_lifters import RangeOnlyLifter
from poly_matrix.least_squares_problem import LeastSquaresProblem


class RangeOnlySLAM1Lifter(RangeOnlyLifter):
    """Range-only SLAM, version 1

    Uses substitution tau_i=||t_i||^2, alpha_k=||a_k||^2, e_ik = a_k @ t_i
    """

    def __init__(self, n_positions, n_landmarks, d, edges=None, remove_gauge="hard"):
        super().__init__(
            n_positions, n_landmarks, d, edges=edges, remove_gauge=remove_gauge
        )

    @property
    def M(self):
        return self.n_positions + self.n_landmarks + len(self.edges)

    @property
    def N(self):
        if self.remove_gauge == "hard":
            if self.d == 2:
                # a1_x, a2, a3, ...
                return self.n_positions * self.d + (self.n_landmarks - 2) * self.d + 1
            else:
                # a1_x, a2_x, a2_y, a3, a4, ...
                return self.n_positions * self.d + (self.n_landmarks - 3) * self.d + 3
        else:
            return (self.n_positions + self.n_landmarks) * self.d

    def get_base_var_dict(self):
        var_dict = {"l": 1}
        var_dict.update({f"x{n}": self.d for n in range(self.n_positions)})
        if self.remove_gauge == "hard":
            # for 2d, a1_x, a2_x, a2_y, ...
            # for 3d, a1_x, a2_x, a2_y, a3_x, a3_y, a3_z, ...
            var_dict.update({f"a{k}": k for k in range(1, self.d)})
            var_dict.update({f"a{k}": self.d for k in range(self.d, self.n_landmarks)})
        else:
            var_dict.update({f"a{k}": self.d for k in range(self.n_landmarks)})
        return var_dict

    @property
    def var_dict(self):
        var_dict = self.get_base_var_dict()
        var_dict.update({f"tau{n}": 1 for n in range(self.n_positions)})
        var_dict.update({f"alpha{k}": 1 for k in range(self.n_landmarks)})
        var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
        return var_dict

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        positions, landmarks = self.get_positions_and_landmarks(theta)

        x_data = [[1]]
        x_data += [list(theta)]
        x_data += [[np.linalg.norm(p) ** 2] for p in positions]
        x_data += [[np.linalg.norm(a) ** 2] for a in landmarks]
        x_data += [[landmarks[k] @ positions[n]] for n, k in self.edges]

        x = np.concatenate(x_data, axis=0)
        assert len(x) == self.N + self.M + 1
        return x

    def generate_random_setup(self):
        self.generate_random_landmarks()

    def sample_theta(self):
        positions = self.sample_random_positions()
        # landmarks = self.sample_random_landmarks()
        return self.get_theta(positions, self.landmarks)

    def get_Q_from_y(self, y):
        self.ls_problem = LeastSquaresProblem()
        for n, k in self.edges:
            self.ls_problem.add_residual(
                # d_nk**2 - ||t_n||**2 + 2t_n@a_k - ||a_k||**2
                #   l         tau_n        e_nk        alpha_k
                {"l": y[n, k], f"tau{n}": -1, f"alpha{k}": -1, f"e{n}{k}": 2}
            )
        # fix Gauge freedom
        if self.remove_gauge == "cost":
            I = np.eye(self.d)
            for d in range(self.d):
                self.ls_problem.add_residual({"a0": I[d].reshape((1, -1))})
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

    def get_A_known(self):
        from poly_matrix.poly_matrix import PolyMatrix

        A_list = []
        for n in range(self.n_positions):
            A = PolyMatrix()
            A[f"x{n}", f"x{n}"] = np.eye(self.d)
            A["l", f"tau{n}"] = -0.5
            A_list.append(A.get_matrix(self.var_dict))
        for k in range(self.n_landmarks):
            A = PolyMatrix()
            if self.remove_gauge == "hard":
                if k > 0:
                    A[f"a{k}", f"a{k}"] = np.eye(self.var_dict[f"a{k}"])
            else:
                A[f"a{k}", f"a{k}"] = np.eye(self.d)
            A["l", f"alpha{k}"] = -0.5
            A_list.append(A.get_matrix(self.var_dict))
        for n, k in self.edges:
            A = PolyMatrix()
            if self.remove_gauge == "hard":
                if k > 0:
                    A[f"x{n}", f"a{k}"] = np.eye(self.d)[:, : self.var_dict[f"a{k}"]]
            else:
                A[f"x{n}", f"a{k}"] = np.eye(self.d)
            A["l", f"e{n}{k}"] = -1
            A_list.append(A.get_matrix(self.var_dict))
        return A_list

    def get_cost(self, t, y):
        # fix Gauge freedom
        cost = super().get_cost(t, y)
        if self.remove_gauge == "cost":
            cost += np.linalg.norm(self.landmarks[0]) ** 2
        return cost

    def sample_feasible(self):
        return self.sample_random_positions()

    def fill_depending_on_k(self, row, k, vec):
        """Because of Gauge freedom removal,
        the first columns of the landmark-based part of J
        are incomplete.
        """
        if k == 0:
            return
        elif k == 1:
            i = self.n_positions * self.d
            row[i] = vec[0]
        elif k == 2:
            i = self.n_positions * self.d + 1
            row[i : i + 2] = vec[:2]
        elif k >= 3:
            i = self.n_positions * self.d + 3 + (k - 3) * self.d
            row[i : i + self.d] = vec

    def get_J_lifting(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)

        J_lifting = np.zeros((self.M, self.N))
        counter = 0
        for n in range(self.n_positions):
            i = n * self.d
            J_lifting[counter, i : i + self.d] = 2 * positions[n]
            counter += 1
        for k in range(self.n_landmarks):
            if self.remove_gauge == "hard":
                self.fill_depending_on_k(J_lifting[counter, :], k, 2 * landmarks[k, :])
            else:
                i = (self.n_positions + k) * self.d
                J_lifting[counter, i : i + self.d] = 2 * landmarks[k]
            counter += 1
        for n, k in self.edges:
            i = n * self.d
            J_lifting[counter, i : i + self.d] = landmarks[k]
            if self.remove_gauge == "hard":
                self.fill_depending_on_k(J_lifting[counter, :], k, positions[n])
            else:
                i = (self.n_positions + k) * self.d
                J_lifting[counter, i : i + self.d] = positions[n]
            counter += 1
        return J_lifting

    def fill_hessian_depending_on_k(self, hessian, k, fix_i=None, val=2.0):
        if k == 0:  # and (fix_i is None):
            return  # no Hessian!
        # elif k == 0:
        #    pass

        elif k == 1:
            i = self.n_positions * self.d
            if fix_i is None:
                hessian[i, i] = val  # a1_x
            else:
                hessian[fix_i[0], i] = val
                hessian[i, fix_i[0]] = val
        elif k == 2:
            i = self.n_positions * self.d + 1
            var_i = range(i, i + 2)
            if fix_i is None:
                hessian[var_i, var_i] = val  # a2_x, a2_y
            else:
                hessian[fix_i[:2], var_i] = val  # a2_x, a2_y
                hessian[var_i, fix_i[:2]] = val  # a2_x, a2_y
        elif k >= 3:
            i = self.n_positions * self.d + 3 + (k - 3) * self.d
            var_i = range(i, i + self.d)
            if fix_i is None:
                hessian[var_i, var_i] = val  # a3_x, a3_y, a3_z
            else:
                hessian[fix_i, var_i] = val
                hessian[var_i, fix_i] = val

    def get_hess_lifting(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        for j in range(self.M):
            hessian = np.zeros((self.N, self.N))

            # Hessian of || tau_j ||^2:  2 * I
            if j < self.n_positions:
                i = j * self.d
                hessian[range(i, i + self.d), range(i, i + self.d)] = 2

            # Hessian of || alpha_j ||^2:  2 * I
            elif j < self.n_landmarks + self.n_positions:
                if self.remove_gauge == "hard":
                    k = j - self.n_positions
                    self.fill_hessian_depending_on_k(hessian, k)
                else:
                    i = j * self.d
                    hessian[i : i + self.d, i : i + self.d] = 2.0

            # Hessian of alpha_j@tau_j:  tau_j or alpha_j
            else:
                n, k = self.edges[j - self.n_landmarks - self.n_positions]
                if self.remove_gauge == "hard":
                    i = n * self.d
                    self.fill_hessian_depending_on_k(
                        hessian, k, fix_i=range(i, i + self.d), val=1.0
                    )
                else:
                    i = n * self.d
                    j = (self.n_positions + k) * self.d
                    hessian[range(i, i + self.d), range(j, j + self.d)] = 1
                    hessian[range(j, j + self.d), range(i, i + self.d)] = 1
            hessians.append(hessian)
        return hessians

    def __repr__(self):
        return f"rangeonlyslam1-{self.d}d"


if __name__ == "__main__":
    lifter = RangeOnlySLAM1Lifter(n_positions=3, n_landmarks=4, d=2)
    lifter.run(n_dual=1, noise=0.1, plot=True)
