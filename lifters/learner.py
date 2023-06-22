import numpy as np
import pandas as pd

from utils.plotting_tools import import_plt

plt = import_plt()

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.plotting_tools import savefig

NOISE = 1e-2
SEED = 5

ADJUST = True
TOL_REL_GAP = 1e-3

# threshold for SVD
EPS_SVD = 1e-5


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

    def __init__(self, lifter: StateLifter, variable_list: list):
        self.lifter = lifter
        self.variable_iter = iter(variable_list)

        self.mat_vars = ["l"]

        # b_tuples contains the learned "patterns" of the form:
        # ((i, mat_vars), <i-th learned vector for these mat_vars variables>)
        self.b_tuples = []
        self.b_current_ = None
        # PolyMatrix summarizing all valid patterns (for plotting and debugging mostly)
        self.patterns_poly_ = None

        # A_matrices contains the generated Poly matrices (induced from the patterns),
        # elements are of the form: (name, <poly matrix>)
        self.A_matrices = []
        self.a_current_ = None

        # list of dual costs
        self.ranks = []
        self.dual_costs = []
        self.solver_vars = None

    @property
    def mat_var_dict(self):
        return {k: self.lifter.var_dict[k] for k in self.mat_vars}

    @property
    def row_var_dict(self):
        return self.lifter.var_dict_all(self.mat_vars)

    @property
    def patterns_poly(self):
        if self.patterns_poly_ is None:
            self.patterns_poly_ = self.generate_patterns_poly(
                factor_out_parameters=True
            )
        return self.patterns_poly_

    def get_a_current(self):
        target_mat_var_dict = self.lifter.var_dict
        if self.a_current_ is None:
            a_list = [
                self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict))
                for __, A_poly in self.A_matrices
            ]
            if len(a_list):
                self.a_current_ = np.vstack(a_list)
        elif self.a_current_.shape[0] < len(self.A_matrices):
            a_list = [
                self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict))
                for __, A_poly in self.A_matrices[self.a_current_.shape[0] :]
            ]
            self.a_current_ = np.vstack([self.a_current_] + a_list)
        return self.a_current_

    def get_b_current(self, target_mat_var_dict=None):
        if target_mat_var_dict is None:
            target_mat_var_dict = self.mat_var_dict

        # if self.b_current_ is None:
        b_list = []
        for (i, mat_vars), bi in self.b_tuples:
            bi = self.lifter.zero_pad_subvector(bi, mat_vars, target_mat_var_dict)
            b_list.append(bi)
        if len(b_list):
            self.b_current_ = np.vstack(b_list)

        # TODO: below doesn't work because b keeps changing size in each iteration.
        # we can make it work by adding blocks according to the new variable set to consider.
        # elif self.b_current_.shape[0] < len(self.b_tuples):
        #    b_list = []
        #    for (i, mat_var_dict), bi in self.b_tuples[self.b_current_.shape[0] :]:
        #        bi = self.lifter.zero_pad_subvector(
        #            bi, mat_var_dict, target_mat_var_dict
        #        )
        #        b_list.append(bi)
        #    if len(b_list):
        #        self.b_current_ = np.vstack([self.b_current_] + b_list)
        return self.b_current_

    def duality_gap_is_zero(self, dual_cost, verbose=False):
        res = (
            abs(self.solver_vars["qcqp_cost"] - dual_cost)
            / self.solver_vars["qcqp_cost"]
            < TOL_REL_GAP
        )
        if not verbose:
            return res
        elif res:
            print(
                f"achieved strong duality: qcqp cost={self.solver_vars['qcqp_cost']:.2e}, dual cost={dual_cost:.2e}"
            )
        elif not res:
            print(
                f"no strong duality yet: qcqp cost={self.solver_vars['qcqp_cost']:.2e}, dual cost={dual_cost:.2e}"
            )
        return res

    def is_tight(self, verbose=False):
        A_list = [
            A_poly.get_matrix(self.lifter.var_dict) for __, A_poly in self.A_matrices
        ]
        A_b_list_all = self.lifter.get_A_b_list(A_list)
        X, info = self._test_tightness(A_b_list_all, verbose=True)
        self.dual_costs.append(info["cost"])
        if info["cost"] is None:
            self.ranks.append(np.zeros(A_list[0].shape[0]))
            print("Warning: is problem infeasible?")
            max_error, bad_list = self.lifter.test_constraints(A_list)
            print("Maximum error:", max_error)
            return False
        else:
            eigs = np.linalg.eigvalsh(X)[::-1]
            self.ranks.append(eigs)
            return self.duality_gap_is_zero(info["cost"], verbose=verbose)

    def generate_minimal_subset(self, reorder=False):
        from solvers.sparse import solve_lambda

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict) for __, A_poly in self.A_matrices
        ]
        A_b_list_all = self.lifter.get_A_b_list(A_list)

        # find the importance of each constraint
        if reorder:
            __, lamdas = solve_lambda(
                self.solver_vars["Q"],
                A_b_list_all,
                self.solver_vars["xhat"],
                force_first=1,
            )
            if lamdas is None:
                print("Warning: problem doesn't have feasible solution!")
                return False

            # order constraints by importance
            sorted_idx = np.argsort(np.abs(lamdas[1:]))[::-1]
        else:
            sorted_idx = range(len(self.A_matrices))
        A_b_list = [(self.lifter.get_A0(), 1.0)]

        dual_costs = []
        tightness_counter = 0
        for i, idx in enumerate(sorted_idx):
            Ai_sparse = self.A_matrices[idx][1].get_matrix(self.lifter.var_dict)
            A_b_list += [(Ai_sparse, 0.0)]
            X, info = self._test_tightness(A_b_list, verbose=False)
            dual_cost = info["cost"]
            # dual_cost = 1e-10
            dual_costs.append(dual_cost)
            if dual_cost is None:
                print(f"{i}/{len(sorted_idx)}: solver error")
            elif self.duality_gap_is_zero(dual_cost):
                print(f"{i}/{len(sorted_idx)}: tight")
                tightness_counter += 1
            else:
                print(f"{i}/{len(sorted_idx)}: not tight yet")
            if tightness_counter > 10:
                break

        plt.figure()
        plt.axhline(self.solver_vars["qcqp_cost"], color="k")
        plt.scatter(range(len(dual_costs)), dual_costs)
        plt.show()

        # b_tuples constains all of A_matrices in this case, and has the same ordering.
        # TODO(FD): this only works when we don't do incremental learning -- make this more explicit
        # or improve the implementation.
        return [self.b_tuples[idx] for idx in sorted_idx[:i]]

    def _test_tightness(self, A_b_list_all, verbose=False):
        from solvers.common import find_local_minimum, solve_sdp_cvxpy

        if self.solver_vars is None:
            np.random.seed(SEED)
            Q, y = self.lifter.get_Q(noise=NOISE)
            qcqp_that, qcqp_cost = find_local_minimum(self.lifter, y=y, verbose=False)
            xhat = self.lifter.get_x(qcqp_that)
            self.solver_vars = dict(Q=Q, y=y, qcqp_cost=qcqp_cost, xhat=xhat)

        # compute lambas by solving dual problem
        X, info = solve_sdp_cvxpy(
            self.solver_vars["Q"], A_b_list_all, adjust=ADJUST, verbose=verbose
        )  # , rho_hat=qcqp_cost)
        return X, info

    def update_variables(self):
        # add new variable to the list of variables to study
        try:
            self.mat_vars = next(self.variable_iter)
            return True
        except StopIteration:
            return False

    def learn_patterns(self, use_known=False):
        Y = self.lifter.generate_Y(var_subset=self.mat_vars)

        basis_current = self.get_b_current()
        if use_known and basis_current is not None:
            Y = np.vstack([Y, basis_current])

        basis_new, S = self.lifter.get_basis(Y, eps=EPS_SVD)
        corank = basis_new.shape[0]

        if corank > 0:
            StateLifter.test_S_cutoff(S, corank, eps=EPS_SVD)

        new_patterns = []
        for i, bi_sub in enumerate(basis_new):
            # check if this newly learned pattern is linearly independent of previous patterns.
            bi_sub[np.abs(bi_sub) < 1e-10] = 0.0

            # sanity check
            ai = self.lifter.get_reduced_a(bi_sub, var_subset=self.mat_vars)
            Ai_sparse = self.lifter.get_mat(ai, var_dict=self.mat_var_dict)
            Ai, __ = PolyMatrix.init_from_sparse(Ai_sparse, self.lifter.var_dict)

            err, bad_idx = self.lifter.test_constraints([Ai_sparse], errors="print")
            if len(bad_idx):
                print(
                    f"constraint matrix pattern b{i} has high error: {err:.2e}, not adding it to patterns"
                )
                continue

            ai_full = self.lifter.get_vec(Ai_sparse)
            a_current = self.get_a_current()
            if increases_rank(
                a_current, ai_full
            ):  # if increases_rank(basis_current, bi_sub):
                new_patterns.append(bi_sub)

                name = f"{self.mat_vars[-1]}:b{i}"

                self.A_matrices.append((name, Ai))
        print(
            f"{self.mat_vars}: found {len(new_patterns)}/{basis_new.shape[0]} independent patterns"
        )
        return new_patterns

    def apply_patterns(self, new_patterns):
        # list of constraint indices that were not redundant after summing out parameters.
        valid_list = []

        for i, new_pattern in enumerate(new_patterns):
            if (self.lifter.param_level == "no") and any(
                ["z" in var for var in self.mat_vars]
            ):
                # if we did not add parameters, then each learned constraint only applies to
                # one specific landmark, and we can't add any augmented constraints
                # (they will all be invalid and rejected anyways)
                new_poly_rows = [
                    self.lifter.convert_b_to_polyrow(new_pattern, self.mat_vars)
                ]
            else:
                new_poly_rows = self.lifter.augment_basis_list(
                    [new_pattern],
                    self.mat_var_dict,
                    n_landmarks=self.lifter.n_landmarks,
                )

            # For each new augmented row, check if it increases the current rank.
            # We operate on the full set of landmarks here, and we bring all of the
            # constraints to A form before checking ranks.
            counter = 0
            for j, new_poly_row in enumerate(new_poly_rows):
                ai = self.lifter.convert_poly_to_a(new_poly_row, self.lifter.var_dict)
                a_current = self.get_a_current()
                if increases_rank(a_current, ai):
                    Ai_sparse = self.lifter.get_mat(ai, var_dict=self.lifter.var_dict)
                    Ai, __ = PolyMatrix.init_from_sparse(
                        Ai_sparse, self.lifter.var_dict
                    )

                    error, bad_idx = self.lifter.test_constraints(
                        [Ai_sparse], errors="ignore"
                    )
                    if len(bad_idx):
                        print(
                            f"skipping matrix {j} of pattern b{i} because high error {error:.2e}"
                        )
                        continue

                    # name = f"[{','.join(self.mat_vars)}]:{i}"
                    name = f"{self.mat_vars[-1]}:b{i}-{j}"
                    self.A_matrices.append((name, Ai))
                    counter += 1
            if counter > 0:
                # print(f"pattern b{i}: added {counter}/{len(new_poly_rows)} new constraints")
                valid_list.append(i)
            elif counter == 0:
                pass
                # print(f"   pattern b{i}: no new constraints")
        return valid_list

    def run(self, verbose=False):
        while not self.is_tight(verbose=verbose):
            # add one more variable to the list of variables to vary
            if not self.update_variables():
                print("no more variables to add")
                break

            # learn new patterns, orthogonal to the ones found so far.
            new_patterns = self.learn_patterns(use_known=True)
            if len(new_patterns) == 0:
                print("new variables didn't have any effect")
                continue

            # apply the pattern to all landmarks
            self.apply_patterns(new_patterns)
            self.b_tuples += [
                ((i, self.mat_vars), p) for i, p in enumerate(new_patterns)
            ]

    def get_sorted_df(self):
        series = []
        for i in self.patterns_poly.variable_dict_i:
            data = {j: float(val) for j, val in self.patterns_poly.matrix[i].items()}
            series.append(
                pd.Series(
                    data,
                    index=self.patterns_poly.variable_dict_j,
                    dtype="Sparse[float]",
                )
            )
        df = pd.DataFrame(
            series, dtype="Sparse[float]", index=self.patterns_poly.variable_dict_i
        )

        def sort_fun(series):
            # This is a bit complicated because we don't want the order to change
            # because of the values, only isna() should matter.
            # To make this work, we temporarily change the non-nan values to the order in which they appear
            index = pd.MultiIndex.from_product([[0], series.index])
            series.index = index
            scipy_sparse = series.sparse.to_coo()[0]
            # don't start at 0 because it's considered empty by scipy
            scipy_sparse.data = np.arange(1, 1 + scipy_sparse.nnz)
            pd_sparse = pd.Series.sparse.from_coo(scipy_sparse, dense_index=True)
            return pd_sparse

        df.dropna(axis=1, how="all", inplace=True)
        df_sorted = df.sort_values(
            key=sort_fun,
            by=list(df.columns),
            axis=0,
            na_position="last",
            inplace=False,
        )
        return df_sorted

    def generate_patterns_poly(self, b_tuples=None, factor_out_parameters=False):
        if b_tuples is None:
            b_tuples = self.b_tuples

        plot_rows = []
        plot_row_labels = []
        j = -1
        old_mat_vars = ""
        for key, new_pattern in b_tuples:
            i, mat_vars = key
            if factor_out_parameters:
                ai = self.lifter.get_reduced_a(new_pattern, mat_vars)
                poly_row = self.lifter.convert_a_to_polyrow(ai, mat_vars)
            else:
                poly_row = self.lifter.convert_b_to_polyrow(new_pattern, mat_vars)

            plot_rows.append(poly_row)
            if mat_vars != old_mat_vars:
                j += 1
                plot_row_labels.append(f"{j}{mat_vars}:b{i}")
                old_mat_vars = mat_vars
            else:
                plot_row_labels.append(f"{j}:b{i}")

        patterns_poly = PolyMatrix.init_from_row_list(
            plot_rows, row_labels=plot_row_labels
        )

        # make sure variable_dict_j is ordered correctly.
        patterns_poly.variable_dict_j = self.lifter.var_dict_all(
            mat_vars, with_parameters=not factor_out_parameters
        )
        return patterns_poly

    def save_df(self, df, fname_root="", title=""):
        from utils.plotting_tools import plot_basis

        # convert to poly matrix for plotting purposes only.
        poly_matrix = PolyMatrix(symmetric=False)
        for i, row in df.iterrows():
            for k, val in row[~row.isna()].items():
                poly_matrix[i, k] = val

        variables_j = self.lifter.var_dict_all(
            var_subset=self.mat_vars, with_parameters=False
        )
        fig, ax = plot_basis(poly_matrix, variables_j=variables_j, discrete=True)
        ax.set_title(title)

        # below works too, incase above takes too long.
        # from utils.plotting_tools import add_colorbar
        # fig, ax = plt.subplots()
        # h_w_ratio = len(df) / len(df.columns)
        # fig.set_size_inches(10, 10 * h_w_ratio)  # w, h
        # im = ax.pcolormesh(df.values.astype(float))
        # ax.set_xticks(range(1, 1 + len(df.columns)), df.columns, rotation=90)
        # ax.xaxis.tick_top()
        # ax.set_yticks(range(len(df.index)), df.index)
        # add_colorbar(fig, ax, im)

        if fname_root != "":
            savefig(fig, fname_root + "_patterns-sorted.png")
        return fig, ax

    def save_patterns(self, fname_root="", title="", with_parameters=False):
        from utils.plotting_tools import plot_basis

        patterns_poly = self.generate_patterns_poly(
            factor_out_parameters=not with_parameters
        )
        variables_j = self.lifter.var_dict_all(
            self.mat_vars, with_parameters=with_parameters
        )
        fig, ax = plot_basis(patterns_poly, variables_j=variables_j, discrete=True)
        if with_parameters:
            for p in range(1, self.lifter.get_dim_P(self.mat_vars)):
                ax.axvline(p * self.lifter.get_dim_X(self.mat_vars) - 0.5, color="red")

        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_patterns.png")
        return fig, ax

    def save_tightness(self, fname_root, title=""):
        labels = self.mat_vars[-len(self.dual_costs) :]

        fig, ax = plt.subplots()
        xticks = range(len(self.dual_costs))
        ax.semilogy(xticks, self.dual_costs)
        ax.set_xticks(xticks, labels)
        ax.axhline(self.solver_vars["qcqp_cost"], color="k", ls=":")
        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_tightness.png")

        ratios = [e[0] / e[1] for e in self.ranks]
        fig, ax = plt.subplots()
        xticks = range(len(ratios))
        ax.semilogy(xticks, ratios)
        ax.set_xticks(xticks, labels)
        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_ratios.png")

        fig, ax = plt.subplots()
        labels = self.mat_vars[-len(self.ranks) :]
        for eig, label in zip(self.ranks, labels):
            ax.semilogy(eig, label=label)
        ax.legend(loc="upper right")
        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_eigs.png")

    def save_matrices(self, fname_root, title=""):
        from lifters.plotting_tools import plot_matrices

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict) for __, A_poly in self.A_matrices
        ]
        names = [f"{k}\n{i}/{len(A_list)}" for i, (k, __) in enumerate(self.A_matrices)]
        fig, axs = plot_matrices(
            A_list=A_list,
            colorbar=False,
            vmin=-1,
            vmax=1,
            nticks=3,
            names=names,
        )
        fig.suptitle(title)
        if fname_root != "":
            savefig(fig, fname_root + "_matrices.png")


if __name__ == "__main__":
    from stereo1d_lifter import Stereo1DLifter

    max_vars = 2
    n_landmarks = 10
    variable_list = [
        ["l", "x"] + [f"z_{i}" for i in range(j)] for j in range(max_vars + 1)
    ]
    lifter = Stereo1DLifter(n_landmarks=n_landmarks, param_level="p")
    learner = Learner(lifter=lifter, variable_list=variable_list)

    fname_root = f"_results/{lifter}"
    learner.run()
    learner.save_patterns(with_parameters=True, fname_root=fname_root)
    learner.save_tightness(fname_root)
    # learner.save_matrices(fname_root)
    plt.show()
    print("done")
