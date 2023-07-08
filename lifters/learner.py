import time
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sparseqr as sqr

from utils.plotting_tools import import_plt, add_rectangles, add_colorbar

plt = import_plt()

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.plotting_tools import savefig
from utils.constraint import Constraint


NOISE_SEED = 5

ADJUST_Q = True  # rescale Q matrix
PRIMAL = False # use primal or dual formulation of SDP. Recommended is False, because of how MOSEK is set up.

TOL_REL_GAP = 1e-3
TOL_RANK_ONE = 1e8


class Learner(object):
    """
    Class to incrementally learn and augment constraint templates until we reach tightness.
    """

    def __init__(
        self, lifter: StateLifter, variable_list: list, apply_templates: bool = True
    ):
        self.lifter = lifter
        self.variable_iter = iter(variable_list)

        self.apply_templates_to_others = apply_templates

        self.mat_vars = ["l"]

        # templates contains the learned "templates" of the form:
        # ((i, mat_vars), <i-th learned vector for these mat_vars variables, PolyRow form>)
        self.b_current_ = None  # current basis formed from b matrices
        self.templates_poly_ = None  # for plotting only: all templats stacked in one

        # A_matrices contains the generated PolyMatrices (induced from the templates)
        self.a_current_ = []  # current basis formed from a matrices

        # new representation, making sure we don't compute the same thing twice.
        self.constraints = []
        self.constraint_index = 0
        self.index_tested = set()

        # list of dual costs
        self.df_tight = None
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
    def templates_poly(self):
        if self.templates_poly_ is None:
            self.templates_poly_ = self.generate_templates_poly(
                factor_out_parameters=True
            )
        return self.templates_poly_

    @property
    def A_matrices(self):
        return [c.A_sparse_ for c in self.constraints]

    def get_a_row_list(self, constraints):
        a_row_list = [constraint.a_full_ for constraint in constraints]
        # a_row_list = [constraint.a_full_ for constraint in constraints]
        return a_row_list

    def get_a_current(self):
        if len(self.a_current_) < len(self.constraints):
            a_new = self.get_a_row_list(self.constraints[len(self.a_current_) :])
            self.a_current_ += a_new
        return self.a_current_

    def get_b_current(self, target_mat_var_dict=None):
        """
        Extract basis vectors that depend on a subset of the currently used parameters (keys in target_mat_var_dict).

        example:
            - templates contains("l", "x", "z_0"): list of learned constraints for this subset
            - target: ("l", "x")
        """
        if target_mat_var_dict is None:
            target_mat_var_dict = self.lifter.get_var_dict_unroll(self.mat_var_dict)

        # if self.b_current_ is None:
        b_list = []
        for constraint in self.constraints:
            bi = constraint.b(self.lifter)
            mat_vars = constraint.mat_var_dict
            i = constraint.index
            bi = self.lifter.zero_pad_subvector(bi, mat_vars, target_mat_var_dict)
            if bi is not None:
                b_list.append(bi)
        if len(b_list):
            self.b_current_ = np.vstack(b_list)
        return self.b_current_

    def check_violation(self, dual_cost):
        primal_cost = self.solver_vars["qcqp_cost"]
        return (dual_cost - primal_cost) / primal_cost > TOL_REL_GAP

    def duality_gap_is_zero(self, dual_cost, verbose=False):
        primal_cost = self.solver_vars["qcqp_cost"]
        res = (primal_cost - dual_cost) / primal_cost < TOL_REL_GAP
        if not verbose:
            return res

        if res:
            print(f"achieved cost tightness:")
        else:
            print(f"no cost tightness yet:")
        print(
            f"qcqp cost={self.solver_vars['qcqp_cost']:.4e}, dual cost={dual_cost:.4e}"
        )
        return res

    def is_rank_one(self, eigs, verbose=False):
        res = eigs[0] / eigs[1] > TOL_RANK_ONE
        if not verbose:
            return res
        if res:
            print("achieved rank tightness:")
        else:
            print("no rank tightness yet:")
        print(
            f"first two eigenvalues: {eigs[0]:.2e}, {eigs[1]:.2e}, ratio:{eigs[0] / eigs[1]:.2e}"
        )
        return res

    def is_tight(self, verbose=False, tightness="rank"):
        A_list = [constraint.A_sparse_ for constraint in self.constraints]
        A_b_list_all = self.lifter.get_A_b_list(A_list)
        X, info = self._test_tightness(A_b_list_all, verbose=True)

        final_cost = np.trace(self.solver_vars["Q"] @ X) 
        if abs(final_cost - info["cost"]) >= 1e-10:
            print(f"Warning: cost is inconsistent: {final_cost:.3e}, {info['cost']:.3e}")

        self.dual_costs.append(info["cost"])
        self.variable_list.append(self.mat_vars)

        if info["cost"] is None:
            self.ranks.append(np.zeros(A_list[0].shape[0]))
            print("Warning: is problem infeasible?")
            max_error, bad_list = self.lifter.test_constraints(A_list, errors="print")
            print("Maximum error:", max_error)
            return False
        elif self.check_violation(info["cost"]):
            self.ranks.append(np.zeros(A_list[0].shape[0]))
            print(f"Dual cost higher than QCQP: {info['cost']:.2e}, {self.solver_vars['qcqp_cost']:.2e}")
            print("Usually this means that MOSEK tolerances are too loose, or that there is a mistake in the constraints.")
            max_error, bad_list = self.lifter.test_constraints(A_list, errors="raise")
            print("Maximum feasibility error at random x:", max_error)

            print("It can also mean that we are not sampling enough of the space close to the true solution.")
            tol = 1e-10
            xhat = self.solver_vars["xhat"]
            max_error = -np.inf
            for Ai in A_list:
                error = xhat.T @ Ai @ xhat
                assert abs(error) < tol, error

                errorX = np.trace(X @ Ai)
                max_error = max(errorX, max_error)
            print(f"Maximum feasibility error at solution x: {max_error}")

            return False
        else:
            eigs = np.linalg.eigvalsh(X)[::-1]
            self.ranks.append(eigs)

            if tightness == "rank":
                tightness_val = self.is_rank_one(eigs, verbose=verbose)
            elif tightness == "cost":
                tightness_val = self.duality_gap_is_zero(info["cost"], verbose=verbose)
            return tightness_val

    def generate_minimal_subset(self, reorder=False, tightness="rank"):
        from solvers.sparse import solve_lambda

        A_list = [constraint.A_sparse_ for constraint in self.constraints]
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
                return None
            print("found valid lamdas")

            # order constraints by importance
            sorted_idx = np.argsort(np.abs(lamdas[1:]))[::-1]
        else:
            lamdas = np.zeros(len(self.constraints))
            sorted_idx = range(len(self.constraints))
        A_b_list = [(self.lifter.get_A0(), 1.0)]

        df_data = []

        minimal_indices = []
        tightness_counter = 0

        rank_idx = None
        cost_idx = None
        new_data = {"lifter": str(self.lifter), "reorder": reorder}
        for i, idx in enumerate(sorted_idx):
            new_data.update({"idx": idx, "lamda": lamdas[idx], "value": self.constraints[idx].value})
            Ai_sparse = A_list[idx]

            A_b_list += [(Ai_sparse, 0.0)]
            X, info = self._test_tightness(A_b_list, verbose=False)

            dual_cost = info["cost"]
            # dual_cost = 1e-10
            new_data["dual cost"] = dual_cost
            if dual_cost is None:
                new_data["eigs"] = np.zeros(Ai_sparse.shape[0])
                print(f"{i}/{len(sorted_idx)}: solver error")
                continue

            if self.duality_gap_is_zero(dual_cost):
                if cost_idx is None:
                    cost_idx = i
                    print(f"{i}/{len(sorted_idx)}: cost-tight")
                new_data["cost_tight"] = True
                if tightness == "cost":
                    tightness_counter += 1
            else:
                new_data["cost_tight"] = False
                print(f"{i}/{len(sorted_idx)}: not cost-tight yet: {dual_cost:.3e}, {self.solver_vars['qcqp_cost']:.3e}")

            eigs = np.linalg.eigvalsh(X)[::-1]
            new_data["eigs"] = eigs
            if self.is_rank_one(eigs):
                if rank_idx is None:
                    rank_idx = i
                    print(f"{i}/{len(sorted_idx)}: rank-tight")
                new_data["rank_tight"] = True
                if tightness == "rank":
                    tightness_counter += 1
            else:
                new_data["rank_tight"] = False
                # print(f"{i}/{len(sorted_idx)}: not rank-tight yet")

            df_data.append(deepcopy(new_data))

            # add all necessary constraints to the list.
            if tightness_counter <= 1:
                minimal_indices.append(idx)

            if tightness_counter > 10:
                break

        if self.df_tight is None:
            self.df_tight = pd.DataFrame(df_data)
        else:
            df_tight = pd.DataFrame(df_data)
            self.df_tight = pd.concat([self.df_tight, df_tight], axis=0)
        return minimal_indices

    def _test_tightness(self, A_b_list_all, verbose=False):
        from solvers.common import find_local_minimum, solve_sdp_cvxpy

        if self.solver_vars is None:
            np.random.seed(NOISE_SEED)
            Q, y = self.lifter.get_Q()
            qcqp_that, qcqp_cost = find_local_minimum(self.lifter, y=y, verbose=False)
            xhat = self.lifter.get_x(qcqp_that)
            self.solver_vars = dict(Q=Q, y=y, qcqp_cost=qcqp_cost, xhat=xhat)

        # compute lambas by solving dual problem
        X, info = solve_sdp_cvxpy(
            self.solver_vars["Q"], A_b_list_all, adjust=ADJUST_Q, verbose=verbose, primal=PRIMAL
        )  # , rho_hat=qcqp_cost)
        return X, info

    def update_variables(self):
        # add new variable to the list of variables to study
        try:
            self.mat_vars = next(self.variable_iter)
            return True
        except StopIteration:
            return False

    def learn_templates(self, use_known=False, plot=False, data_dict=None):
        Y = self.lifter.generate_Y(var_subset=self.mat_vars, factor=1.5)

        if use_known:
            basis_current = self.get_b_current()
            if basis_current is not None:
                Y = np.vstack([Y, basis_current])

        if plot:
            fig, ax = plt.subplots()

        for i in range(self.lifter.N_CLEANING_STEPS + 1):
            basis_new, S = self.lifter.get_basis(Y)
            corank = basis_new.shape[0]
            if corank > 0:
                self.lifter.test_S_cutoff(S, corank)
            bad_idx = self.lifter.clean_Y(basis_new, Y, S, plot=False)

            if plot:
                from lifters.plotting_tools import plot_singular_values

                if len(bad_idx):
                    plot_singular_values(
                        S, eps=self.lifter.EPS_SVD, label=f"run {i}", ax=ax
                    )
                else:
                    plot_singular_values(S, eps=self.lifter.EPS_SVD, ax=ax)

            if len(bad_idx) > 0:
                Y = np.delete(Y, bad_idx, axis=0)
            else:
                break
        if data_dict is not None:
            data_dict["rank Y"] = Y.shape[1] - corank
            data_dict["corank Y"] = corank

        print(f"found {basis_new.shape[0]} templates from data matrix Y {Y.shape} ")
        if basis_new.shape[0]:
            # check if the newly found templates are independent of previous, and add them to the list.
            constraints = [
                Constraint.init_from_b(
                    index=self.constraint_index + i,
                    mat_var_dict=self.mat_var_dict,
                    b=b,
                    lifter=self.lifter,
                )
                for i, b in enumerate(basis_new)
            ]
            self.constraint_index += basis_new.shape[0]
            return self.clean_constraints(
                new_constraints=constraints,
                remove_dependent=True,
                remove_imprecise=False,
            )
        return 0, len(self.constraints)

    # @profile
    def apply_templates(self, reapply_all=False):
        # the new templates are all the ones corresponding to the new matrix variables.
        t1 = time.time()
        new_constraints = []
        for constraint in self.constraints:
            if (not reapply_all) and (constraint.mat_var_dict != self.mat_var_dict):
                continue
            new_templates = self.lifter.augment_basis_list(
                [constraint.polyrow_b_],
                n_landmarks=self.lifter.n_landmarks,
            )
            new_constraints += [
                Constraint.init_from_polyrow_b(
                    index=self.constraint_index + i,
                    polyrow_b=new_template,
                    lifter=self.lifter,
                )
                for i, new_template in enumerate(new_templates)
            ]
            self.constraint_index += len(new_templates)
        print(f"-- time to apply templates: {time.time() - t1:.3f}s")

        # determine which of these constraints are actually independent, after reducing them to ai.
        t1 = time.time()
        n_new, n_all = self.clean_constraints(
            new_constraints=new_constraints,
            remove_dependent=True,
            remove_imprecise=False,
        )
        print(f"-- time to add constraints: {time.time() - t1:.3f}s")
        return n_new, n_all

    # @profile
    def clean_constraints(
        self, new_constraints, remove_dependent=True, remove_imprecise=False
    ):
        """
        This function is used in two different ways.

        First use case: Given the new templates, in b-PolyRow form, we determine which of the templates are actually
        independent to a_current. We only want to augment the independent ones, otherwise we waste computing effort.

        Second use case: After applying the templates to as many variable pairs as we wish, we call this function again,
        to make sure all the matrices going into the SDP are in fact linearly independent.
        """
        n_before = len(self.constraints)
        self.constraints += new_constraints

        if remove_dependent:
            t1 = time.time()
            # find which constraints are lin. dep.
            a_current = self.get_a_current()
            A_vec = sp.vstack(a_current, format="coo").T

            # Use sparse rank revealing QR
            # We "solve" a least squares problem to get the rank and permutations
            # This is the cheapest way to use sparse QR, since it does not require
            # explicit construction of the Q matrix. We can't do this with qr function
            # because the "just return R" option is not exposed.
            Z, R, E, rank = sqr.rz(
                A_vec, np.zeros((A_vec.shape[0], 1)), tolerance=1e-10
            )
            # Sort the diagonal values. Note that SuiteSparse uses AMD/METIS ordering
            # to acheive sparsity.
            r_vals = np.abs(R.diagonal())
            sort_inds = np.argsort(r_vals)[::-1]
            if rank < A_vec.shape[1]:
                print(f"clean_constraints: keeping {rank}/{len(E)}")
            bad_idx = list(E[sort_inds[rank:]])

            good_idx = list(E[sort_inds[:rank]])
            for idx in good_idx:
                self.constraints[idx].value = r_vals[idx]

            # Sanity check, removed because too expensive. It almost always passed anyways.
            # Z, R, E, rank_full = sqr.rz(A_vec[:, keep_idx], np.zeros((A_vec.shape[0],1)), tolerance=1e-10)
            # if rank_full != rank:
            #     print(f"Warning: selected constraints did not pass lin. independence check. Rank is {rank_full}, should be {rank}.")
            print(f"time to find independent {time.time() - t1:.3f}s")

        if remove_dependent:
            if len(bad_idx):
                for idx in sorted(bad_idx)[::-1]:
                    del self.constraints[idx]
                    del a_current[idx]

        if remove_imprecise:
            t1 = time.time()
            error, bad_idx = self.lifter.test_constraints(
                [
                    c.A_sparse_
                    for c in self.constraints
                    if c.index not in self.index_tested
                ],
                errors="ignore",
                n_seeds=2,
            )
            self.index_tested = self.index_tested.union(
                [c.index for c in self.constraints]
            )
            print(f"time to test {time.time() - t1:.3f}s")

            if len(bad_idx):
                print(f"removing {bad_idx} because high error, up to {error:.2e}")
                for idx in list(bad_idx)[::-1]:  # reverse order to not mess up indexing
                    del self.constraints[idx]
                    del a_current[idx]

        self.a_current_ = a_current
        return len(self.constraints) - n_before, len(self.constraints)

    def run(self, use_known=True, verbose=False, plot=False, tightness="rank"):
        data = []
        while 1:
            # add one more variable to the list of variables to vary
            if not self.update_variables():
                print("no more variables to add")
                break

            print(f"======== {self.mat_vars} ========")
            data_dict = {"variables": self.mat_vars}
            data_dict["dim Y"] = self.lifter.get_dim_Y(self.mat_vars)

            print(f"-------- templates learning --------")
            # learn new templates, orthogonal to the ones found so far.
            t1 = time.time()
            n_new, n_all = self.learn_templates(
                use_known=use_known, plot=plot, data_dict=data_dict
            )
            data_dict["n templates"] = n_all
            if n_new == 0:
                print("new variables didn't have any effect")
                continue
            ttot = time.time() - t1
            data_dict["t learn templates"] = ttot
            print(f"time:   {ttot:.3f}s")

            # apply the pattern to all landmarks
            if self.apply_templates_to_others:
                print(f"------- applying templates ---------")
                t1 = time.time()

                n_new, n_all = self.apply_templates()
                data_dict["n templates"] = n_all
                ttot = time.time() - t1
                data_dict["t apply templates"] = ttot
                print(f"time:  {ttot:.3f}s")

            t1 = time.time()
            print(f"-------- checking tightness ----------")
            is_tight = self.is_tight(verbose=verbose, tightness=tightness)
            ttot = time.time() - t1
            data_dict["t check tightness"] = ttot
            print(f"time: {ttot:.3f}s")
            data.append(data_dict)
            if is_tight:
                break
        return data

    def get_sorted_df(self, templates_poly=None, add_columns={}):
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

        if templates_poly is None:
            templates_poly = self.templates_poly

        series = []
        for i, key_i in enumerate(templates_poly.variable_dict_i):
            data = {j: float(val) for j, val in templates_poly.matrix[key_i].items()}
            for key, idx_list in add_columns.items():
                try:
                    data[key] = idx_list.index(i)
                except ValueError:
                    data[key] = -1
            series.append(
                pd.Series(
                    data,
                    index=list(templates_poly.variable_dict_j.keys())
                    + list(add_columns.keys()),
                    dtype="Sparse[float]",
                )
            )
        df = pd.DataFrame(
            series, dtype="Sparse[float]", index=templates_poly.variable_dict_i
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

    def generate_templates_poly(self, constraints=None, factor_out_parameters=False):
        if constraints is None:
            constraints = self.constraints

        plot_rows = []
        plot_row_labels = []
        j = -1
        old_mat_vars = ""
        for constraint in constraints:
            mat_vars = constraint.mat_var_dict
            i = constraint.index
            if factor_out_parameters:
                plot_rows.append(constraint.polyrow_a_)
            else:
                plot_rows.append(constraint.polyrow_b_)
            if mat_vars != old_mat_vars:
                j += 1
                plot_row_labels.append(f"{j}:b{i}")
                # plot_row_labels.append(f"{j}{mat_vars}:b{i}")
                old_mat_vars = mat_vars
            else:
                plot_row_labels.append(f"{j}:b{i}")

        templates_poly = PolyMatrix.init_from_row_list(
            plot_rows, row_labels=plot_row_labels
        )

        # make sure variable_dict_j is ordered correctly.
        templates_poly.variable_dict_j = self.lifter.var_dict_row(
            mat_vars, force_parameters_off=not factor_out_parameters
        )
        return templates_poly

    def save_sorted_templates(self, df, fname_root="", title="", drop_zero=False):
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
            variables_j = {k: v for k, v in variables_j.items() if k in keys}
        fig, ax = plot_basis(poly_matrix, variables_j=variables_j, discrete=True)
        ax.set_title(title)

        if "required (reordered)" in df.columns:
            from matplotlib.patches import Rectangle

            for i, (__, row) in enumerate(df.iterrows()):
                if row["required (reordered)"] < 0:
                    ax.add_patch(
                        Rectangle(
                            (ax.get_xlim()[0], i - 0.5),
                            ax.get_xlim()[1] + 0.5,
                            1.0,
                            fc="white",
                            alpha=0.5,
                            lw=0.0,
                        )
                    )
        ax.set_yticklabels([])
        new_xticks = [
            f"${lbl.get_text().replace('l-', '')}$" for lbl in ax.get_xticklabels()
        ]
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
            savefig(fig, fname_root + "_templates-sorted.png")
        return fig, ax

    def save_templates(self, fname_root="", title="", with_parameters=False):
        from utils.plotting_tools import plot_basis

        templates_poly = self.generate_templates_poly(
            factor_out_parameters=not with_parameters
        )
        variables_j = self.lifter.var_dict_row(
            self.mat_vars, force_parameters_off=not with_parameters
        )
        fig, ax = plot_basis(templates_poly, variables_j=variables_j, discrete=True)
        if with_parameters:
            for p in range(1, self.lifter.get_dim_P(self.mat_vars)):
                ax.axvline(p * self.lifter.get_dim_X(self.mat_vars) - 0.5, color="red")

        ax.set_title(title)
        if fname_root != "":
            savefig(fig, fname_root + "_templates.png")
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

    def save_matrices_sparsity(self, A_matrices=None, fname_root="", title=""):
        if A_matrices is None:
            A_matrices = self.A_matrices

        Q = self.solver_vars["Q"].toarray()
        sorted_i = self.lifter.var_dict_unroll
        agg_ii = []
        agg_jj = []
        for i, A_poly in enumerate(A_matrices):
            A_sparse = A_poly.get_matrix(variables=sorted_i)
            ii, jj = A_sparse.nonzero()
            agg_ii += list(ii)
            agg_jj += list(jj)
        A_agg = sp.csr_matrix(([1.0] * len(agg_ii), (agg_ii, agg_jj)), A_sparse.shape)

        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        im0 = axs[0].matshow(
            1 - A_agg.toarray(), vmin=0, vmax=1, cmap="gray"
        )  # 1 (white) is empty, 0 (black) is nonempty
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

    def save_matrices_poly(
        self, A_matrices=None, fname_root="", title="", reduced_mode=False
    ):
        if A_matrices is None:
            A_matrices = self.A_matrices

        n_rows = len(A_matrices) // 10 + 1
        n_cols = min(len(A_matrices), 10)
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
        fig.set_size_inches(5 * n_cols / n_rows, 5)
        axs = axs.flatten()
        for i, A_poly in enumerate(A_matrices):
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
        for ax in axs[i + 1 :]:
            ax.axis("off")

        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if fname_root != "":
            savefig(fig, fname_root + "_matrices-poly.png")
        return fig, axs


if __name__ == "__main__":
    raise ValueError("don't run this")
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
    learner.save_templates(with_parameters=True, fname_root=fname_root)
    # learner.save_matrices(fname_root)
    plt.show()
    print("done")
