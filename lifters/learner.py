import numpy as np
import matplotlib.pylab as plt

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.plotting_tools import savefig

MAX_VARS = 2
N_LANDMARKS = 3
NOISE = 1e-1
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
    """
    Class to incrementally learn and augment constraint patterns until we reach tightness.
    """

    VARIABLES = ["x"] + [f"z_{i}" for i in range(MAX_VARS)]

    def __init__(self, lifter: StateLifter):
        self.lifter = lifter
        self.var_counter = -1

        self.mat_vars = ["l"]

        # b_vector contains the learned "patterns" of the form:
        # {variable_tuple: [list of learned b-vectors for this variable subset]}
        self.b_vectors = {}

        # A_matrices contains the generated Poly matrices (induced from the patterns)
        self.A_matrices = {}

        # list of dual costs
        self.dual_costs = []
        self.solver_vars = None

    @property
    def mat_var_dict(self):
        return {k: self.lifter.var_dict[k] for k in self.mat_vars}

    @property
    def row_var_dict(self):
        return self.lifter.var_dict_all(self.mat_vars)

    def get_a_current(self, target_mat_var_dict=None):
        a_list = [
            self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict))
            for A_poly in self.A_matrices.values()
        ]
        if len(a_list):
            return np.vstack(a_list)
        return None

    def get_A_current_from_b(self, target_mat_var_dict=None):
        if target_mat_var_dict is None:
            target_mat_var_dict = self.mat_var_dict

        if len(self.b_vectors):
            A_current = []
            for mat_var_dict, b_list in self.b_vectors.items():
                for bi_sub in b_list:
                    bi = self.lifter.zero_pad_subvector(
                        bi_sub,
                        var_subset=mat_var_dict,
                        target_subset=target_mat_var_dict,
                    ).flatten()
                    ai = self.lifter.get_reduced_a(bi, target_mat_var_dict)
                    Ai_sparse = self.lifter.get_mat(ai, target_mat_var_dict)
                    Ai, __ = PolyMatrix.init_from_sparse(Ai_sparse, target_mat_var_dict)
                    A_current.append(Ai)
            return A_current
        return None

    def get_a_current_from_b(self, target_mat_var_dict=None):
        if len(self.b_vectors):
            A_current = self.get_A_current(target_mat_var_dict)
            return np.vstack(
                [
                    self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict))
                    for A_poly in A_current
                ]
            )
        return None

    def get_b_current(self, target_mat_var_dict=None):
        if target_mat_var_dict is None:
            target_mat_var_dict = self.mat_var_dict

        bs = []
        for mat_var_dict, b_list in self.b_vectors.items():
            for bi in b_list:
                bi = self.lifter.zero_pad_subvector(
                    bi, mat_var_dict, target_mat_var_dict
                )
                bs.append(bi)
        if len(bs):
            return np.vstack(bs)
        return None

    def is_tight(self):
        from solvers.common import find_local_minimum, solve_sdp_cvxpy
        from solvers.sparse import solve_lambda

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict)
            for A_poly in self.A_matrices.values()
        ]
        A_b_list_all = self.lifter.get_A_b_list(A_list)

        if self.solver_vars is None:
            np.random.seed(SEED)
            Q, y = self.lifter.get_Q(noise=NOISE)
            qcqp_that, qcqp_cost = find_local_minimum(self.lifter, y=y, verbose=False)
            xhat = self.lifter.get_x(qcqp_that)
            self.solver_vars = dict(Q=Q, y=y, qcqp_cost=qcqp_cost, xhat=xhat)

        # compute lambas by solving dual problem
        X, info = solve_sdp_cvxpy(
            self.solver_vars["Q"], A_b_list_all, adjust=ADJUST
        )  # , rho_hat=qcqp_cost)
        self.dual_costs.append(info["cost"])
        if info["cost"] is None:
            print("Warning: is problem infeasible?")
            return False
        else:
            if (
                abs(self.solver_vars["qcqp_cost"] - info["cost"])
                / self.solver_vars["qcqp_cost"]
                > TOL_REL_GAP
            ):
                print(
                    f"no strong duality yet: qcqp cost={self.solver_vars['qcqp_cost']:.2e}, dual cost={info['cost']:.2e}"
                )
                return False
            else:
                print("achieved strong duality")
                return True

        # compute lamdas using optimization
        __, lamdas = solve_lambda(
            self.solver_vars["Q"], A_b_list_all, self.solver_vars["xhat"], force_first=1
        )
        if lamdas is None:
            print("Warning: problem doesn't have feasible solution!")
            return False

    def update_variables(self):
        # add new variable to the list of variables to study
        self.var_counter += 1
        if self.var_counter >= len(self.VARIABLES):
            return False
        self.mat_vars.append(self.VARIABLES[self.var_counter])
        return True

    def learn_patterns(self, use_known=False):
        Y = self.lifter.generate_Y(var_subset=self.mat_vars)

        basis_current = self.get_b_current()
        if use_known and basis_current is not None:
            Y = np.vstack([Y, basis_current])

        basis_new, S = self.lifter.get_basis(Y)
        corank = basis_new.shape[0]

        print(f"{self.mat_vars}: {corank} learned matrices found")
        if corank > 0:
            StateLifter.test_S_cutoff(S, corank)

        new_patterns = []
        for i, bi_sub in enumerate(basis_new):
            # check if this newly learned pattern is linearly independent of previous patterns.
            bi_sub[np.abs(bi_sub) < 1e-10] = 0.0

            if increases_rank(basis_current, bi_sub):
                print(f"b{i} is a valid pattern")
                new_patterns.append(bi_sub)
            else:
                print(f"b{i} is linearly dependent")
        self.b_vectors[tuple(self.mat_vars)] = new_patterns
        return new_patterns

    def apply_patterns(self):
        key = tuple(self.mat_vars)
        if not key in self.b_vectors:
            print(f"no patterns learned yet for {key}.")
            return

        new_patterns = self.b_vectors[key]

        for i, new_pattern in enumerate(new_patterns):
            new_poly_rows = self.lifter.augment_basis_list(
                [new_pattern], self.mat_var_dict, n_landmarks=self.lifter.n_landmarks
            )

            # For each new augmented row, check if it increases the current rank.
            # We operate on the full set of landmarks here, and we bring all of the
            # constraints to A form before checking ranks.
            all_dict = self.lifter.var_dict_all(self.lifter.var_dict)
            for j, new_poly_row in enumerate(new_poly_rows):
                bi = new_poly_row.get_matrix((["l"], all_dict))
                ai = self.lifter.get_reduced_a(bi, self.lifter.var_dict)

                a_current = self.get_a_current(self.lifter.var_dict)
                if increases_rank(a_current, ai):
                    Ai_sparse = self.lifter.get_mat(ai, var_dict=self.lifter.var_dict)
                    Ai, __ = PolyMatrix.init_from_sparse(
                        Ai_sparse, self.lifter.var_dict
                    )

                    # name = f"[{','.join(self.mat_vars)}]:{i}"
                    name = f"{self.mat_vars[-1]}:b{i}-{j}"
                    self.A_matrices[name] = Ai

    def run(self, fname_root=""):
        from utils.plotting_tools import plot_basis
        from lifters.plotting_tools import plot_matrices

        plot_rows = []
        plot_row_labels = []

        while not self.is_tight():
            if not self.update_variables():
                print("no more variables to add")
                break

            new_patterns = self.learn_patterns(use_known=True)
            if len(new_patterns) == 0:
                print("new variables didn't have any effect")
                continue

            self.apply_patterns()

            plot_rows += [
                self.lifter.convert_b_to_poly(bi, self.mat_vars) for bi in new_patterns
            ]
            plot_row_labels += [
                f"{self.mat_vars}:b{i}" for i in range(len(new_patterns))
            ]

        patterns_poly = PolyMatrix.init_from_row_list(
            plot_rows, row_labels=plot_row_labels
        )
        fig, ax = plot_basis(patterns_poly, self.lifter, var_subset=self.mat_vars)
        if fname_root != "":
            savefig(fig, fname_root + "_patterns.png")

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict)
            for A_poly in self.A_matrices.values()
        ]

        fig, ax = plot_matrices(
            A_list=A_list,
            colorbar=False,
            vmin=-1,
            vmax=1,
            nticks=3,
            names=list(self.A_matrices.keys()),
        )
        if fname_root != "":
            savefig(fig, fname_root + "_matrices.png")

        fig, ax = plt.subplots()
        xticks = range(len(self.dual_costs))
        ax.semilogy(xticks, self.dual_costs)
        ax.set_xticks(xticks, self.mat_vars[-len(xticks) :])
        ax.axhline(self.solver_vars["qcqp_cost"], color="k", ls=":")
        if fname_root != "":
            savefig(fig, fname_root + "_tightness.png")


if __name__ == "__main__":
    # from stereo2d_lifter import Stereo2DLifter
    # lifter = Stereo2DLifter(n_landmarks=N_LANDMARKS, add_parameters=True)

    from stereo1d_lifter import Stereo1DLifter

    lifter = Stereo1DLifter(n_landmarks=N_LANDMARKS, add_parameters=True)

    learner = Learner(lifter=lifter)
    fname_root = f"_results/{lifter}"
    learner.run(fname_root=fname_root)
    plt.show()
    print("done")
