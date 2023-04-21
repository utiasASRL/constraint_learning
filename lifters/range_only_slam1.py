import itertools

import matplotlib.pylab as plt
import numpy as np

from lifters.range_only_lifters import RangeOnlyLifter
from poly_matrix.least_squares_problem import LeastSquaresProblem
from poly_matrix.poly_matrix import PolyMatrix


def vkron(a, b):
    """
    if a is 1xN, make it Nx1
    if b is NxN leave it
    """
    if (np.ndim(a) == 2) and (a.shape[0] == 1):
        a = a.T
    elif np.ndim(a) == 1:
        a = a.reshape((-1, 1))
    if (np.ndim(b) == 2) and (b.shape[0] == 1):
        b = b.T
    elif np.ndim(b) == 1:
        b = b.reshape((-1, 1))
    return np.kron(a, b)


class RangeOnlySLAM1Lifter(RangeOnlyLifter):
    """Range-only SLAM, version 1

    Uses substitution tau_i=||t_i||^2, alpha_k=||a_k||^2, e_ik = a_k @ t_i
    """

    LEVELS = [
        "inner",
        "outer",
    ]

    def __init__(
        self,
        n_positions,
        n_landmarks,
        d,
        edges=None,
        remove_gauge="hard",
        resample_landmarks=False,
        level="inner",
    ):
        self.level = level
        self.resample_landmarks = resample_landmarks
        super().__init__(
            n_positions, n_landmarks, d, edges=edges, remove_gauge=remove_gauge
        )

    @property
    def M(self):
        M = self.n_positions + self.n_landmarks + len(self.edges)
        if self.level == "outer":
            M *= self.d**2
        return M

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

    @property
    def base_var_dict(self):
        var_dict = {}
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
    def sub_var_dict(self):
        var_dict = {}
        if self.level == "inner":
            var_dict.update({f"tau{n}": 1 for n in range(self.n_positions)})
            var_dict.update({f"alpha{k}": 1 for k in range(self.n_landmarks)})
            var_dict.update({f"e{n}{k}": 1 for n, k in self.edges})
        elif self.level == "outer":
            dim = self.d**2
            var_dict.update({f"tau{n}": dim for n in range(self.n_positions)})
            var_dict.update({f"alpha{k}": dim for k in range(self.n_landmarks)})
            var_dict.update({f"e{n}{k}": dim for n, k in self.edges})
        return var_dict

    @property
    def var_dict(self):
        var_dict = {"l": 1}
        var_dict.update(self.base_var_dict)
        var_dict.update(self.sub_var_dict)
        return var_dict

    def get_x(self, theta=None):
        if theta is None:
            theta = self.theta
        positions, landmarks = self.get_positions_and_landmarks(theta)

        x_data = [[1]]
        x_data += [list(theta)]
        if self.level == "inner":
            x_data += [[np.linalg.norm(p) ** 2] for p in positions]
            x_data += [[np.linalg.norm(a) ** 2] for a in landmarks]
            x_data += [[landmarks[k] @ positions[n]] for n, k in self.edges]
        elif self.level == "outer":
            x_data += [list(np.kron(p, p)) for p in positions]
            x_data += [list(np.kron(a, a)) for a in landmarks]
            x_data += [list(np.kron(positions[n], landmarks[k])) for n, k in self.edges]
        x = np.concatenate(x_data, axis=0)
        assert len(x) == self.N + self.M + 1
        return x

    def generate_random_setup(self):
        self.generate_random_landmarks()

    def sample_theta(self):
        positions = self.sample_random_positions()
        if self.resample_landmarks:
            landmarks = self.sample_random_landmarks()
        else:
            landmarks = self.landmarks
        return self.get_theta(positions, landmarks)

    def get_Q_from_y(self, y):
        self.ls_problem = LeastSquaresProblem()
        if self.level == "outer":
            I = np.eye(self.d).flatten().reshape((1, -1))
        for n, k in self.edges:
            if self.level == "inner":
                # d_nk**2 - ||t_n||**2 + 2t_n@a_k - ||a_k||**2
                #   l         tau_n        e_nk        alpha_k
                self.ls_problem.add_residual(
                    {"l": y[n, k], f"tau{n}": -1, f"alpha{k}": -1, f"e{n}{k}": 2}
                )
            else:
                # d_nk**2 - ||t_n||**2 + 2t_n@a_k - ||a_k||**2
                #   l       -I @ tau_n  +2I @ e_nk  -I @ alpha_k
                self.ls_problem.add_residual(
                    {
                        "l": y[n, k],
                        f"tau{n}": -I,
                        f"alpha{k}": -I,
                        f"e{n}{k}": 2 * I,
                    }
                )
        # fix Gauge freedom
        if self.remove_gauge == "cost":
            I = np.eye(self.d)
            for d in range(self.d):
                self.ls_problem.add_residual({"a0": I[d].reshape((1, -1))})
        return self.ls_problem.get_Q().get_matrix(self.var_dict)

    def get_A_known(self):
        A_list = []
        for n in range(self.n_positions):
            if self.level == "inner":
                A = PolyMatrix()
                A[f"x{n}", f"x{n}"] = np.eye(self.d)
                A["l", f"tau{n}"] = -0.5
                A_list.append(A.get_matrix(self.var_dict))
            else:
                for j, k in itertools.product(range(self.d), range(self.d)):
                    X = np.zeros((self.d, self.d))
                    x = np.zeros(self.d**2)
                    A = PolyMatrix()
                    X[j, k] += 1.0
                    X[k, j] += 1.0
                    x[j * self.d + k] += -1.0
                    A[f"x{n}", f"x{n}"] += X
                    A["l", f"tau{n}"] += x.reshape((1, -1))
                    A_list.append(A.get_matrix(self.var_dict))

        if self.level == "outer":
            # TODO(FD) implement the other known matrices
            return A_list

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

    def fill_depending_on_k(self, row, counter, k, vec):
        """Because of Gauge freedom removal,
        the first columns of the landmark-based part of J
        are incomplete.
        """
        if k == 0:
            return counter

        # below is equivalent to row[counter, f"a{k}"] = min(k, 3)
        # keeping it like this for better readability
        elif k == 1:
            # i = self.n_positions * self.d
            # row[i] = vec[0]
            row[counter, f"a{k}"] = vec[0]
            return counter + 1
        elif k == 2:
            # i = self.n_positions * self.d + 1
            # row[i : i + 2] = vec[:2]
            row[counter, f"a{k}"] = vec[:2][None, :]
            return counter + 1
        elif k >= 3:
            # i = self.n_positions * self.d + 3 + (k - 3) * self.d
            # row[i : i + self.d] = vec
            row[counter, f"a{k}"] = vec[None, :]
            return counter + 1

    def get_J_lifting(self, t):
        if self.level == "inner":
            return self.get_J_lifting_inner(t)
        elif self.level == "outer":
            return self.get_J_lifting_outer(t)

    def get_J_lifting_outer(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)
        J_lifting = PolyMatrix(symmetric=False)
        for n in range(self.n_positions):
            # below is d/d(p_n) (p_n kron p_n)
            # fmt:off
            J_lifting[f"tau{n}", f"x{n}"] = ( 
                vkron(positions[n], np.eye(self.d)) 
              + vkron(np.eye(self.d), positions[n])
            )
            # fmt:on
        for k in range(self.n_landmarks):
            if self.remove_gauge == "hard":
                if k == 0:
                    continue
                # below is d/d(a_k) (a_k kron a_k)
                dim = self.var_dict[f"a{k}"]
                I = np.zeros((self.d, self.d))
                I[range(dim), range(dim)] = 1.0
                ak = landmarks[k]
                # fmt:off
                J_lifting[f"alpha{k}", f"a{k}"] = (
                    vkron(ak, I) + 
                    vkron(I, ak)
                )[:, :dim]
                # fmt:on
            else:
                ak = landmarks[k]
                # fmt:off
                J_lifting[f"alpha{k}", f"a{k}"] = (
                    vkron(ak, np.eye(self.d)) + 
                    vkron(np.eye(self.d), ak)
                )
                # fmt:on
        for n, k in self.edges:
            # d/d(p_n) (p_n kron a_n)
            J_lifting[f"e{n}{k}", f"x{n}"] = vkron(np.eye(self.d), self.landmarks[k])
            if self.remove_gauge == "hard":
                if k == 0:
                    continue
                dim = self.var_dict[f"a{k}"]
                # d/d(a_k) (p_n kron a_k)
                # if a_n has k < d elements:
                #
                # p_n = [x y z] a_k = [a b]
                # p_n kron a_k = [xa xb ya yb za zb]
                # J = [x
                #        x
                #          0
                #      y
                #        y
                #          0
                #      z
                #        z
                #          0]
                I = np.zeros((self.d, self.d))
                I[range(dim), range(dim)] = 1.0
                J_lifting[f"e{n}{k}", f"a{k}"] = vkron(
                    self.positions[n],
                    I,
                )[:, :dim]
            else:
                J_lifting[f"e{n}{k}", f"a{k}"] = vkron(
                    self.positions[n],
                    np.eye(self.d),
                )
        return J_lifting.get_matrix((self.sub_var_dict, self.base_var_dict))

    def get_J_lifting_inner(self, t):
        positions, landmarks = self.get_positions_and_landmarks(t)
        J_lifting = PolyMatrix(symmetric=False)
        for n in range(self.n_positions):
            J_lifting[f"tau{n}", f"x{n}"] = 2 * positions[n].reshape((1, -1))
        for k in range(self.n_landmarks):
            if self.remove_gauge == "hard":
                if k == 0:
                    continue
                J_lifting[f"alpha{k}", f"a{k}"] = (
                    2 * landmarks[k, : self.var_dict[f"a{k}"]]
                ).reshape((1, -1))
            else:
                J_lifting[f"alpha{k}", f"a{k}"] = 2 * landmarks[k].reshape((1, -1))
        for n, k in self.edges:
            J_lifting[f"e{n}{k}", f"x{n}"] = landmarks[k].reshape((1, -1))
            if self.remove_gauge == "hard":
                if k == 0:
                    continue
                J_lifting[f"e{n}{k}", f"a{k}"] = positions[
                    n, : self.var_dict[f"a{k}"]
                ].reshape((1, -1))
            else:
                J_lifting[f"e{n}{k}", f"a{k}"] = positions[n].reshape((1, -1))
        return J_lifting.get_matrix((self.sub_var_dict, self.base_var_dict))

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
        if self.level == "inner":
            return self.get_hess_lifting_inner(t)
        else:
            raise NotImplementedError(self.level)

    def get_hess_lifting_inner(self, t):
        """return list of the hessians of the M lifting functions."""
        hessians = []
        # Hessian of || tau_j ||^2:  2 * I
        for n in range(self.n_positions):
            hessian = PolyMatrix()
            hessian[f"x{n}", f"x{n}"] = 2 * np.eye(self.d)
            hessians.append(hessian.get_matrix(self.base_var_dict))
        # Hessian of || alpha_k ||^2:  2 * I
        for k in range(self.n_landmarks):
            hessian = PolyMatrix()
            if k > 0:
                hessian[f"a{k}", f"a{k}"] = 2 * np.eye(self.var_dict[f"a{k}"])
            hessians.append(hessian.get_matrix(self.base_var_dict))
        # Hessian of alpha_j@tau_j:  tau_j or alpha_j
        for n, k in self.edges:
            hessian = PolyMatrix()
            hessian_old = np.zeros((self.N, self.N))
            if self.remove_gauge == "hard":
                # old implementation
                i = n * self.d
                self.fill_hessian_depending_on_k(
                    hessian_old, k, fix_i=range(i, i + self.d), val=1.0
                )

                # new
                if k > 0:
                    hessian[f"x{n}", f"a{k}"] = np.eye(self.d)[
                        :, : self.var_dict[f"a{k}"]
                    ]
            else:
                # old implementation
                i = n * self.d
                j = (self.n_positions + k) * self.d
                hessian_old[range(i, i + self.d), range(j, j + self.d)] = 1
                hessian_old[range(j, j + self.d), range(i, i + self.d)] = 1

                # new
                hessian[f"x{n}", f"a{k}"] = np.eye(self.d)

            hessian_new = hessian.get_matrix(self.base_var_dict).toarray()
            np.testing.assert_allclose(hessian_old, hessian_new)

            hessians.append(hessian.get_matrix(self.base_var_dict))
        assert len(hessians) == self.M
        return hessians

    def __repr__(self):
        return f"rangeonlyslam1-{self.d}d"


if __name__ == "__main__":
    lifter = RangeOnlySLAM1Lifter(
        n_positions=3, n_landmarks=4, d=2, resample_landmarks=True, level="outer"
    )
    lifter.run(n_dual=1, noise=0.1, plot=True)

    lifter = RangeOnlySLAM1Lifter(
        n_positions=3, n_landmarks=4, d=2, resample_landmarks=False
    )
    lifter.run(n_dual=1, noise=0.1, plot=True)

    lifter = RangeOnlySLAM1Lifter(
        n_positions=3, n_landmarks=4, d=2, resample_landmarks=True
    )
    lifter.run(n_dual=1, noise=0.1, plot=True)
