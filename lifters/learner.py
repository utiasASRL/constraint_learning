import numpy as np
import matplotlib.pylab as plt

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix

MAX_VARS = 3
NOISE = 1e-5
SEED = 5

ADJUST = True
TOL_REL_GAP = 1e-3


def increases_rank(mat, new_row):
    # TODO(FD) below is not the most efficient way of checking lin. indep.
    new_row = new_row.flatten()
    if mat is None:
        return True
    mat_test = np.vstack([mat, new_row[None, :]])
    new_rank = np.linalg.matrix_rank(mat_test)

    # if the new matrix is not full row-rank then the new row was
    # actually linealy dependent.
    if new_rank == mat_test.shape[0]:
        return True
    return False


class Learner(object):
    VARIABLES = [f"z_{i}" for i in range(MAX_VARS)]

    def __init__(self, lifter: StateLifter):
        self.lifter = lifter
        self.variables = ["l", "x"]
        self.var_counter = -1
        self.poly_matrices = []

    def get_a_current(self, var_dict=None):
        if var_dict is None:
            var_dict = self.var_dict
        if len(self.poly_matrices):
            return np.vstack(
                [
                    self.lifter.get_vec(p.get_matrix(var_dict))
                    for p in self.poly_matrices
                ]
            )
        return None

    def get_A_current(self, var_dict=None):
        if var_dict is None:
            var_dict = self.var_dict
        return [p.get_matrix(var_dict) for p in self.poly_matrices]

    def is_tight(self):
        return False
        from solvers.common import find_local_minimum, solve_sdp_cvxpy
        from solvers.sparse import solve_lambda

        A_list = self.get_A_current(self.lifter.var_dict)
        A_b_list_all = self.lifter.get_A_b_list(A_list)

        np.random.seed(SEED)
        Q, y = self.lifter.get_Q(noise=NOISE)
        qcqp_that, qcqp_cost = find_local_minimum(self.lifter, y=y, verbose=False)
        xhat = self.lifter.get_x(qcqp_that)

        # compute lambas by solving dual problem
        X, info = solve_sdp_cvxpy(
            Q, A_b_list_all, adjust=ADJUST
        )  # , rho_hat=qcqp_cost)
        if info["cost"] is None:
            print("Warning: is problem infeasible?")
            return False
        else:
            if abs(qcqp_cost - info["cost"]) / qcqp_cost > TOL_REL_GAP:
                print("Warning: no strong duality?")
                print(f"qcqp cost: {qcqp_cost}")
                print(f"dual cost: {info['cost']}")
                return False
            else:
                print("Strong duality!")
                return True

        # compute lamdas using optimization
        __, lamdas = solve_lambda(Q, A_b_list_all, xhat, force_first=1)
        if lamdas is None:
            print("Warning: problem doesn't have feasible solution!")
            return False

    @property
    def var_dict(self):
        return {k: self.lifter.var_dict[k] for k in self.variables}

    def update_variables(self):
        # add new variable to the list of variables to study
        self.var_counter += 1
        if self.var_counter >= len(self.VARIABLES):
            return False
        self.variables.append(self.VARIABLES[self.var_counter])
        return True

    def learn_patterns(self, use_known=False):
        Y = self.lifter.generate_Y(var_subset=self.variables)

        a_current = self.get_a_current()

        if use_known:
            A_current = self.get_A_current()
            basis_new, S = self.lifter.get_basis(Y, A_known=A_current)
        else:
            basis_new, S = self.lifter.get_basis(Y)

        corank = basis_new.shape[0]

        print(f"{self.variables}: {corank} learned matrices found")
        if corank > 0:
            StateLifter.test_S_cutoff(S, corank)

        new_patterns = []
        for bi_sub in basis_new:
            # check if this newly learned pattern is linearly independent of previous patterns.
            bi_sub[np.abs(bi_sub) < 1e-10] = 0.0
            bi, bi_poly = self.lifter.zero_pad_subvector(
                bi_sub, var_subset=self.var_dict, target_subset=self.var_dict
            )

            ai = self.lifter.get_reduced_a(bi, var_subset=self.var_dict)

            # if so, add it to the new patterns.
            if increases_rank(a_current, ai):
                new_patterns.append(bi_poly)
        return new_patterns

    def augment_patterns(self, new_patterns):
        new_poly_rows = self.lifter.augment_basis_list(new_patterns)

        # for each new augmented row, check if it increases the current rank.
        var_dict = {l: 1 for l in self.lifter.get_label_list(self.variables)}
        for new_poly_row in new_poly_rows:
            bi = new_poly_row.get_matrix((["l"], var_dict))
            ai = self.lifter.get_reduced_a(bi, self.variables)

            a_current = self.get_a_current()
            if increases_rank(a_current, ai):
                Ai_sparse = self.lifter.get_mat(ai, var_dict=self.var_dict)
                Ai, __ = PolyMatrix.init_from_sparse(Ai_sparse, self.var_dict)
                self.poly_matrices.append(Ai)

    def run(self):
        from utils.plotting_tools import plot_basis
        from lifters.plotting_tools import plot_matrices

        plot_rows = []
        plot_row_labels = []
        while not self.is_tight():
            if not self.update_variables():
                print("no more variables to add.")
                break

            new_patterns = self.learn_patterns(use_known=False)
            if len(new_patterns) == 0:
                print("new variables didn't have any effect.")
                continue

            plot_rows += new_patterns
            plot_row_labels += [
                f"{self.variables}:{i}" for i in range(len(new_patterns))
            ]

            self.augment_patterns(new_patterns)

        patterns_poly = PolyMatrix.init_from_row_list(
            plot_rows, row_labels=plot_row_labels
        )
        fig, ax = plot_basis(patterns_poly, self.lifter)

        fig, ax = plot_matrices(
            A_list=[p.get_matrix(self.var_dict) for p in self.poly_matrices],
            colorbar=False,
            vmin=-1,
            vmax=1,
        )


if __name__ == "__main__":
    from stereo1d_lifter import Stereo1DLifter

    lifter = Stereo1DLifter(n_landmarks=MAX_VARS, add_parameters=True)
    learner = Learner(lifter=lifter)
    learner.run()
    plt.show()
    print("done")
