import time

import numpy as np
import pandas as pd
import scipy.sparse as sp


from utils.plotting_tools import import_plt, add_rectangles, add_colorbar
plt = import_plt()

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.plotting_tools import savefig
from utils.common import increases_rank

NOISE = 1e-2
SEED = 5

ADJUST = True
TOL_REL_GAP = 1e-3
TOL_RANK_ONE = 1e8

# threshold for SVD
EPS_SVD = 1e-5
# threshold to consider matrix element zero
EPS_SPARSE = 1e-9

N_CLEANING_STEPS = 2 # set to 0 for no effect

class Learner(object):
    """
    Class to incrementally learn and augment constraint patterns until we reach tightness.
    """

    def __init__(self, lifter: StateLifter, variable_list: list, apply_patterns:bool=True):
        self.lifter = lifter
        self.variable_iter = iter(variable_list)

        self.apply_patterns_to_others = apply_patterns

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
        self.variable_list = []
        self.solver_vars = None

    @property
    def mat_var_dict(self):
        return {k: self.lifter.var_dict[k] for k in self.mat_vars}

    @property
    def row_var_dict(self):
        return self.lifter.var_dict_row(self.mat_vars)

    @property
    def patterns_poly(self):
        if self.patterns_poly_ is None:
            self.patterns_poly_ = self.generate_patterns_poly(
                factor_out_parameters=True
            )
        return self.patterns_poly_

    def get_a_current(self, sparse=False):

        target_mat_var_dict = self.lifter.var_dict_unroll
        if self.a_current_ is None:
            a_list = [
                self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict), sparse=sparse)
                for __, A_poly in self.A_matrices
            ]
            if len(a_list):
                if sparse:
                    self.a_current_ = sp.vstack(a_list)
                else:
                    self.a_current_ = np.vstack(a_list)
        elif self.a_current_.shape[0] < len(self.A_matrices):
            a_list = [
                self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict), sparse=sparse)
                for __, A_poly in self.A_matrices[self.a_current_.shape[0] :]
            ]
            if len(a_list):
                if sparse:
                    self.a_current_ = sp.vstack([self.a_current_] + a_list)
                else:
                    self.a_current_ = np.vstack([self.a_current_] + a_list)
        return self.a_current_

    def get_b_current(self, target_mat_var_dict=None):
        """
        Extract basis vectors that depend on a subset of the currently used parameters (keys in target_mat_var_dict).

        example:
            - b_tuples contains("l", "x", "z_0"): list of learned constraints for this subset
            - target: ("l", "x")
        """
        if target_mat_var_dict is None:
            target_mat_var_dict = self.lifter.get_var_dict_unroll(self.mat_var_dict)

        # if self.b_current_ is None:
        b_list = []
        for (i, mat_vars), bi in self.b_tuples:
            bi = self.lifter.zero_pad_subvector(bi, mat_vars, target_mat_var_dict)
            if bi is not None:
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

        if res:
            print(f"achieved weak tightness:")
        else:
            print(f"no weak tightness yet:")
        print(f"qcqp cost={self.solver_vars['qcqp_cost']:.4e}, dual cost={dual_cost:.4e}")
        return res

    def is_rank_one(self, eigs, verbose=False):
        res = eigs[0] / eigs[1] > TOL_RANK_ONE
        if not verbose: 
            return res
        if res:
            print("achieved strong tightness:")
        else:
            print("no strong tightness yet:")
        print(f"first two eigenvalues: {eigs[0]:.2e}, {eigs[1]:.2e}, ratio:{eigs[0] / eigs[1]:.2e}")
        return res

    def is_tight(self, verbose=False):
        A_list = [
            A_poly.get_matrix(self.lifter.var_dict_unroll) for __, A_poly in self.A_matrices
        ]
        A_b_list_all = self.lifter.get_A_b_list(A_list)
        X, info = self._test_tightness(A_b_list_all, verbose=True)
        self.dual_costs.append(info["cost"])
        self.variable_list.append(self.mat_vars)
        if info["cost"] is None:
            self.ranks.append(np.zeros(A_list[0].shape[0]))
            print("Warning: is problem infeasible?")
            max_error, bad_list = self.lifter.test_constraints(A_list, errors="print")
            print("Maximum error:", max_error)
            return False
        else:
            eigs = np.linalg.eigvalsh(X)[::-1]
            self.ranks.append(eigs)

            weak_tightness = self.duality_gap_is_zero(info["cost"], verbose=verbose)
            strong_tightness = self.is_rank_one(eigs, verbose=verbose)
            return strong_tightness

    def generate_minimal_subset(self, reorder=False, ax_cost=None, ax_eigs=None, tightness="rank"):
        from solvers.sparse import solve_lambda
        from matplotlib.ticker import MaxNLocator

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict_unroll) for __, A_poly in self.A_matrices
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
                return range(len(self.A_matrices))

            # order constraints by importance
            sorted_idx = np.argsort(np.abs(lamdas[1:]))[::-1]
        else:
            sorted_idx = range(len(self.A_matrices))
        A_b_list = [(self.lifter.get_A0(), 1.0)]

        minimal_indices = []
        dual_costs = []
        ranks = []
        tightness_counter = 0
        rank_idx = None
        cost_idx = None
        for i, idx in enumerate(sorted_idx):

            name, Ai_poly = self.A_matrices[idx]
            Ai_sparse = Ai_poly.get_matrix(self.lifter.var_dict_unroll)

            A_b_list += [(Ai_sparse, 0.0)]
            X, info = self._test_tightness(A_b_list, verbose=False)
            dual_cost = info["cost"]
            # dual_cost = 1e-10
            dual_costs.append(dual_cost)
            if dual_cost is None:
                ranks.append(np.zeros(Ai_sparse.shape[0]))
                print(f"{i}/{len(sorted_idx)}: solver error")
                continue

            eigs = np.linalg.eigvalsh(X)[::-1]
            ranks.append(eigs)
            if self.duality_gap_is_zero(dual_cost):
                if cost_idx is None:
                    cost_idx = i
                    print(f"{i}/{len(sorted_idx)}: cost-tight")
                if tightness == "cost":
                    tightness_counter += 1
            else:
                pass
                #print(f"{i}/{len(sorted_idx)}: not cost-tight yet")

            if self.is_rank_one(eigs):
                if rank_idx is None:
                    rank_idx = i
                    print(f"{i}/{len(sorted_idx)}: rank-tight")
                if tightness == "rank":
                    tightness_counter += 1
            else:
                pass
                #print(f"{i}/{len(sorted_idx)}: not rank-tight yet")

            # add all necessary constraints to the list.
            if tightness_counter <= 1:
                minimal_indices.append(idx)

            if tightness_counter > 10:
                break

        if ax_cost is not None:
            ax_cost.semilogy(range(len(dual_costs)), dual_costs, marker="o")
            ax_cost.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_cost.set_xlabel("number of added constraints")
            ax_cost.set_ylabel("cost")
            ax_cost.grid(True)

        if ax_eigs is not None:
            cmap = plt.get_cmap("viridis", len(ranks))
            for i, eig in enumerate(ranks):
                label = None
                color=cmap(i)
                if i == len(ranks) // 2:
                    label="..."
                if i == 0:
                    label=f"{i+1}"
                if i == len(ranks) - 1:
                    label=f"{i+1}"
                if i == cost_idx: 
                    label=f"{i+1}: cost-tight"
                    color="red"
                if i == rank_idx: 
                    label=f"{i+1}: rank-tight"
                    color="black"
                ax_eigs.semilogy(eig, color=color, label=label)

            # make sure these two are in foreground
            if cost_idx is not None:
                ax_eigs.semilogy(ranks[cost_idx], color="red")
            if rank_idx is not None:
                ax_eigs.semilogy(ranks[rank_idx], color="black")
            ax_eigs.set_xlabel("index")
            ax_eigs.set_ylabel("eigenvalue")
            ax_eigs.grid(True)
        return minimal_indices

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

    def learn_patterns(self, use_known=False, plot=False):
        Y = self.lifter.generate_Y(var_subset=self.mat_vars, factor=1.5)

        if use_known:
            basis_current = self.get_b_current()
            if basis_current is not None:
                Y = np.vstack([Y, basis_current])

        if plot:
            fig, ax = plt.subplots()

        print(f"{self.mat_vars}: Y {Y.shape}", end="")
        for i in range(N_CLEANING_STEPS + 1):
            basis_new, S = self.lifter.get_basis(Y, eps_svd=EPS_SVD)
            basis_new[np.abs(basis_new) < EPS_SPARSE] = 0.0
            corank = basis_new.shape[0]
            print(f"found {corank} candidate patterns...", end="")
            if corank > 0:
                StateLifter.test_S_cutoff(S, corank, eps=EPS_SVD)
            bad_idx = self.lifter.clean_Y(basis_new, Y, S, plot=False)

            if plot:
                from lifters.plotting_tools import plot_singular_values
                if len(bad_idx):
                    plot_singular_values(S, eps=EPS_SVD, label=f"run {i}", ax=ax)
                else:
                    plot_singular_values(S, eps=EPS_SVD, ax=ax)

            if len(bad_idx) > 0:
                print(f"deleting {len(bad_idx)} and trying again...", end="")
                Y = np.delete(Y, bad_idx, axis=0)
            else:
                break

        print(f"done, with {basis_new.shape[0]}")
        if basis_new.shape[0]:
            A_matrices_new, patterns_new = self.get_independent_subset(basis_new)
            self.A_matrices = [(f"b{i}:{j}", Aj) for j, Aj in enumerate(A_matrices_new)]
            self.b_tuples = [((i, self.mat_vars), p) for i, p in enumerate(patterns_new)]
        else:
            patterns_new = []
        return patterns_new

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

            ai_full = self.lifter.get_vec(Ai_sparse, sparse=True)
            a_current = self.get_a_current(sparse=True)
            if increases_rank(
                a_current, ai_full
            ):  # if increases_rank(basis_current, bi_sub):
                new_patterns.append(bi_sub)

                name = f"{self.mat_vars[-1]}:b{i}"

                self.A_matrices.append((name, Ai))
        print(f"...{len(new_patterns)} are independent.")
        self.b_tuples += [
            ((i, self.mat_vars), p) for i, p in enumerate(new_patterns)
        ]
        return new_patterns


    def apply_patterns(self, new_patterns):
        new_poly_rows_all = []
        for i, new_pattern in enumerate(new_patterns):
            if False: # need to figure out if we need something here.
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
                new_poly_rows_all += new_poly_rows
        A_matrices_new, __ = self.get_independent_subset(new_poly_rows_all)
        self.A_matrices = [(f"b{i}:{j}", Aj) for j, Aj in enumerate(A_matrices_new)]

    def get_independent_subset(self, new_poly_rows):
        """

        """
        import sparseqr as sqr
        # We operate on the full set of landmarks here, and we bring all of the
        # constraints to A form before checking ranks.
        a_new = None #np.empty((len(new_poly_rows), a_current.shape[1]))

        for j, new_poly_row in enumerate(new_poly_rows):
            if isinstance(new_poly_row, PolyMatrix):
                ai = self.lifter.convert_poly_to_a(new_poly_row, self.lifter.var_dict, sparse=True)
            else:
                ai = self.lifter.get_reduced_a(new_poly_row, self.mat_vars, sparse=True)
                Ai_sparse = self.lifter.get_mat(ai, var_dict=self.mat_var_dict, sparse=True)
                ai = self.lifter.get_vec(Ai_sparse, sparse=True)
            if a_new is None:
                a_new = ai
            else:
                a_new = sp.vstack([a_new, ai])
        
        # find which constraints are lin. dep.
        a_current = self.get_a_current(sparse=True)
        if a_current is None:
            A_vec = a_new.T
        else:
            A_vec = sp.vstack([a_current, a_new]).T

        # Use sparse rank revealing QR
        # We "solve" a least squares problem to get the rank and permutations
        # This is the cheapest way to use sparse QR, since it does not require
        # explicit construction of the Q matrix. We can't do this with qr function
        # because the "just return R" option is not exposed.
        Z,R,E,rank = sqr.rz(A_vec, np.zeros((A_vec.shape[0],1)), tolerance=EPS_SVD)
        # Sort the diagonal values. Note that SuiteSparse uses AMD/METIS ordering 
        # to acheive sparsity.
        r_vals = np.abs(R.diagonal())
        sort_inds = np.argsort(r_vals)[::-1]
        if rank < A_vec.shape[1]:
            # indices of constraints to remove
            print(f"get_independent_subset: keeping {rank}/{len(E)} templates")
            #print("indices that should be kept:", )
        keep_idx = E[sort_inds[:rank]]

        Z,R,E,rank_full = sqr.rz(A_vec[:, keep_idx], np.zeros((A_vec.shape[0],1)), tolerance=EPS_SVD)
        assert rank_full == rank

        # ==== find the pivot elements of matrix r ====
        # sanity check: all elements from previously found basis are lin.indep.
        A_matrices_new = []
        patterns_new = []
        for count, j in enumerate(keep_idx):
            ai = A_vec[:, j].T # 1 x N
            Ai_sparse = self.lifter.get_mat(ai, var_dict=self.lifter.var_dict, sparse=True)
            Ai, __ = PolyMatrix.init_from_sparse(
                Ai_sparse, self.lifter.var_dict, unfold=True
            )
            error, bad_idx = self.lifter.test_constraints(
                [Ai_sparse], errors="ignore"
            )
            if len(bad_idx):
                print(
                    f"skipping matrix {j} of because high error {error:.2e}"
                )
                continue
            A_matrices_new.append(Ai)
            patterns_new.append(self.lifter.convert_a_to_polyrow(ai))
        return A_matrices_new, patterns_new

    def run(self, use_known=True, verbose=False, plot=False):
        times = []
        while 1:
            # add one more variable to the list of variables to vary
            if not self.update_variables():
                print("no more variables to add")
                break

            time_dict = {"variables": self.mat_vars}

            # learn new patterns, orthogonal to the ones found so far.
            t1 = time.time()
            new_patterns = self.learn_patterns(use_known=use_known, plot=plot)
            if len(new_patterns) == 0:
                print("new variables didn't have any effect")
                continue
            ttot = time.time() - t1
            time_dict["learn templates"] = ttot
            print(f"pattern learning:   {ttot:.3f}s")

            # apply the pattern to all landmarks
            if self.apply_patterns_to_others:
                t1 = time.time()
                self.apply_patterns(new_patterns)
                ttot = time.time() - t1
                time_dict["apply templates"] = ttot
                print(f"applying patterns:  {ttot:.3f}s")

            t1 = time.time()
            is_tight = self.is_tight(verbose=verbose)
            ttot = time.time() - t1
            time_dict["check tightness"] = ttot
            print(f"checking tightness: {ttot:.3f}s")
            times.append(time_dict)
            if is_tight: 
                break
        return times
            


    def get_sorted_df(self, patterns_poly=None, add_columns={}):
        def sort_fun_sparsity(series):
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

        if patterns_poly is None:
            patterns_poly = self.patterns_poly

        series = []
        for i, key_i in enumerate(patterns_poly.variable_dict_i):
            data = {j: float(val) for j, val in patterns_poly.matrix[key_i].items()}
            for key, idx_list in add_columns.items():
                try:
                    data[key] = idx_list.index(i)
                except ValueError:
                    data[key] = -1
            series.append(
                pd.Series(
                    data,
                    index=list(patterns_poly.variable_dict_j.keys()) + list(add_columns.keys()),
                    dtype="Sparse[float]",
                )
            )
        df = pd.DataFrame(
            series, dtype="Sparse[float]", index=patterns_poly.variable_dict_i
        )
        df.dropna(axis=1, how="all", inplace=True)

        df_sorted = df.sort_values(
            key=sort_fun_sparsity,
            by=list(df.columns),
            axis=0,
            na_position="last",
            inplace=False,
        )
        df_sorted["order_sparsity"] = range(len(df_sorted))
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
                plot_row_labels.append(f"{j}:b{i}")
                #plot_row_labels.append(f"{j}{mat_vars}:b{i}")
                old_mat_vars = mat_vars
            else:
                plot_row_labels.append(f"{j}:b{i}")

        patterns_poly = PolyMatrix.init_from_row_list(
            plot_rows, row_labels=plot_row_labels
        )

        # make sure variable_dict_j is ordered correctly.
        patterns_poly.variable_dict_j = self.lifter.var_dict_row(
            mat_vars, force_parameters_off=not factor_out_parameters
        )
        return patterns_poly

    def save_sorted_patterns(self, df, fname_root="", title="", drop_zero=False):
        from utils.plotting_tools import plot_basis

        # convert to poly matrix for plotting purposes only.
        poly_matrix = PolyMatrix(symmetric=False)
        keys = set()
        for i, row in df.iterrows():
            for k, val in row[~row.isna()].items():
                poly_matrix[i, k] = val
                keys.add(k)

        variables_j = self.lifter.var_dict_row(
            var_subset=self.mat_vars, force_parameters_off=True
        )
        if drop_zero:
            variables_j = {k:v for k, v in variables_j.items() if k in keys}
        fig, ax = plot_basis(poly_matrix, variables_j=variables_j, discrete=True)
        ax.set_title(title)
        
        if ("required (reordered)" in df.columns):
            from matplotlib.patches import Rectangle
            for i, (__, row) in enumerate(df.iterrows()):
                if row["required (reordered)"] < 0:
                    ax.add_patch(Rectangle((ax.get_xlim()[0], i-0.5), ax.get_xlim()[1]+0.5, 1.0, fc="white", alpha=0.5, lw=0.0))
        ax.set_yticklabels([])
        new_xticks = [f"${lbl.get_text().replace('l-', '')}$" for lbl in ax.get_xticklabels()]
        ax.set_xticklabels(new_xticks)

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
        variables_j = self.lifter.var_dict_row(
            self.mat_vars, force_parameters_off=not with_parameters
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
        labels = self.variable_list

        fig, ax = plt.subplots()
        xticks = range(len(self.dual_costs))
        ax.semilogy(xticks, self.dual_costs, marker="o")
        ax.set_xticks(xticks, labels)
        ax.axhline(self.solver_vars["qcqp_cost"], color="k", ls=":")
        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_tightness.png")

        fig, ax = plt.subplots()
        for eig, label in zip(self.ranks, labels):
            ax.semilogy(eig, label=label)
        ax.legend(loc="upper right")
        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_eigs.png")

        return
        ratios = [e[0] / e[1] for e in self.ranks]
        fig, ax = plt.subplots()
        xticks = range(len(ratios))
        ax.semilogy(xticks, ratios)
        ax.set_xticks(xticks, labels)
        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_ratios.png")

    def save_matrices(self, fname_root, title=""):
        from lifters.plotting_tools import plot_matrices

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict_unroll) for __, A_poly in self.A_matrices
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

    def save_matrices_sparsity(self, A_matrices=None, fname_root="", title=""):
        if A_matrices is None:
            A_matrices = self.A_matrices

        Q = self.solver_vars["Q"].toarray()
        sorted_i = self.lifter.var_dict_unroll
        agg_ii = []
        agg_jj = []
        for i, (name, A_poly) in enumerate(A_matrices):
            A_sparse = A_poly.get_matrix(variables=sorted_i)
            ii, jj  = A_sparse.nonzero()
            agg_ii += list(ii)
            agg_jj += list(jj)
        A_agg = sp.csr_matrix(([1.0]*len(agg_ii), (agg_ii, agg_jj)), A_sparse.shape)

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        im0 = axs[0].matshow(1-A_agg.toarray(), vmin=0, vmax=1, cmap="gray") # 1 (white) is empty, 0 (black) is nonempty
        im1 = axs[1].matshow(Q)

        for ax in axs:
            add_rectangles(ax, self.lifter.var_dict)

        from utils.plotting_tools import add_colorbar
        add_colorbar(fig, axs[1], im1)
        # only for dimensions
        add_colorbar(fig, axs[0], im0, visible=False)
        if fname_root != "":
            savefig(fig, fname_root + "_matrices-sparisty.png")
        return fig, axs

    def save_matrices_poly(self, A_matrices=None, fname_root="", title="", reduced_mode=False):
        if A_matrices is None:
            A_matrices = self.A_matrices

        n_rows = len(A_matrices) // 10 + 1
        n_cols = min(len(A_matrices), 10)
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
        fig.set_size_inches(5*n_cols/n_rows, 5)
        axs = axs.flatten()
        for i, (name, A_poly) in enumerate(A_matrices):
            if reduced_mode:
                sorted_i = sorted(A_poly.variable_dict_i.keys())
            else:
                sorted_i = self.lifter.var_dict_unroll
            from utils.plotting_tools import initialize_discrete_cbar
            A_sparse = A_poly.get_matrix(sorted_i)
            cmap, norm, colorbar_yticks = initialize_discrete_cbar(A_sparse.data)
            im = axs[i].matshow(A_sparse.toarray(), cmap=cmap, norm=norm)
            add_rectangles(axs[i], self.lifter.var_dict)

            cax = add_colorbar(fig, axs[i], im, size=0.1)
            cax.set_yticklabels(colorbar_yticks)
        for ax in axs[i+1:]:
            ax.axis("off")

        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if fname_root != "":
            savefig(fig, fname_root + "_matrices-poly.png")
        return fig, axs


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
    # learner.save_matrices(fname_root)
    plt.show()
    print("done")
