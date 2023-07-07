import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sparseqr as sqr

from utils.plotting_tools import import_plt, add_rectangles, add_colorbar
plt = import_plt()

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix
from utils.plotting_tools import savefig


NOISE_LEVEL = 1e-2
NOISE_SEED = 5

ADJUST_Q = True # rescale Q matrix

TOL_REL_GAP = 1e-3
TOL_RANK_ONE = 1e8

class Learner(object):
    """
    Class to incrementally learn and augment constraint templates until we reach tightness.
    """

    def __init__(self, lifter: StateLifter, variable_list: list, apply_templates:bool=True):
        self.lifter = lifter
        self.variable_iter = iter(variable_list)

        self.apply_templates_to_others = apply_templates

        self.mat_vars = ["l"]

        # templates contains the learned "templates" of the form:
        # ((i, mat_vars), <i-th learned vector for these mat_vars variables, PolyRow form>)
        self.templates = []
        self.b_current_ = None # current basis formed from b matrices
        self.templates_poly_ = None #for plotting only: all templats stacked in one

        # A_matrices contains the generated PolyMatrices (induced from the templates)
        self.A_matrices = []
        self.a_current_ = None # current basis formed from a matrices

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
    def templates_poly(self):
        if self.templates_poly_ is None:
            self.templates_poly_ = self.generate_templates_poly(
                factor_out_parameters=True
            )
        return self.templates_poly_

    def get_a_current(self, sparse=False):
        target_mat_var_dict = self.lifter.var_dict_unroll
        a_list = [
            self.lifter.get_vec(A_poly.get_matrix(target_mat_var_dict), sparse=sparse)
            for A_poly in self.A_matrices
        ]
        if len(a_list):
            if sparse:
                self.a_current_ = sp.vstack(a_list)
            else:
                self.a_current_ = np.vstack(a_list)
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
        for (i, mat_vars), bi in self.templates:
            bi = self.lifter.zero_pad_subvector(bi, mat_vars, target_mat_var_dict)
            if bi is not None:
                b_list.append(bi)
        if len(b_list):
            self.b_current_ = np.vstack(b_list)

        # TODO: below doesn't work because b keeps changing size in each iteration.
        # we can make it work by adding blocks according to the new variable set to consider.
        # elif self.b_current_.shape[0] < len(self.templates):
        #    b_list = []
        #    for (i, mat_var_dict), bi in self.templates[self.b_current_.shape[0] :]:
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

    def is_tight(self, verbose=False, tightness="rank"):
        A_list = [
            A_poly.get_matrix(self.lifter.var_dict_unroll) for  A_poly in self.A_matrices
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

            if tightness == "rank":
                tightness_val = self.is_rank_one(eigs, verbose=verbose)
            elif tightness == "cost":
                tightness_val = self.duality_gap_is_zero(info["cost"], verbose=verbose)
            return tightness_val

    def generate_minimal_subset(self, reorder=False, ax_cost=None, ax_eigs=None, tightness="rank"):
        from solvers.sparse import solve_lambda
        from matplotlib.ticker import MaxNLocator

        A_list = [
            A_poly.get_matrix(self.lifter.var_dict_unroll) for A_poly in self.A_matrices
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
            np.random.seed(NOISE_SEED)
            Q, y = self.lifter.get_Q(noise=NOISE_LEVEL)
            qcqp_that, qcqp_cost = find_local_minimum(self.lifter, y=y, verbose=False)
            xhat = self.lifter.get_x(qcqp_that)
            self.solver_vars = dict(Q=Q, y=y, qcqp_cost=qcqp_cost, xhat=xhat)

        # compute lambas by solving dual problem
        X, info = solve_sdp_cvxpy(
            self.solver_vars["Q"], A_b_list_all, adjust=ADJUST_Q, verbose=verbose
        )  # , rho_hat=qcqp_cost)
        return X, info

    def update_variables(self):
        # add new variable to the list of variables to study
        try:
            self.mat_vars = next(self.variable_iter)
            return True
        except StopIteration:
            return False

    def learn_templates(self, use_known=False, plot=False):
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
                    plot_singular_values(S, eps=self.lifter.EPS_SVD, label=f"run {i}", ax=ax)
                else:
                    plot_singular_values(S, eps=self.lifter.EPS_SVD, ax=ax)

            if len(bad_idx) > 0:
                Y = np.delete(Y, bad_idx, axis=0)
            else:
                break

        print(f"found {basis_new.shape[0]} templates from data matrix Y {Y.shape} ")
        if basis_new.shape[0]:
            # check if the newly found templates are independent of previous, and add them to the list. 
            return self.add_new_constraints(basis_new, remove_dependent=True)
        return 0
            


    @profile
    def apply_templates(self):
        # the new templates are all the ones corresponding to the new matrix variables.
        t1 = time.time()
        new_templates = [template[1] for template in self.templates if template[0][1] == self.mat_vars]
        new_constraints = []
        for new_template in new_templates:
            new_templates = self.lifter.augment_basis_list(
                [new_template],
                self.mat_var_dict,
                n_landmarks=self.lifter.n_landmarks,
            )
            new_constraints += new_templates
        print(f"-- time to apply templates: {time.time() - t1:.3f}s")

        # determine which of these constraints are actually independent, after reducing them to ai.  
        if len(new_constraints):
            t1 = time.time()
            n_new = self.add_new_constraints(new_constraints, remove_dependent=True)
            print(f"-- time to add constraints: {time.time() - t1:.3f}s")
            return n_new
        return 0

    @profile
    def add_new_constraints(self, new_templates, remove_dependent=True):
        """
        This function is used in two different ways. 
        
        First use case: Given the new tempaltes, in b-PolyRow form, we determine which of the templates are actually
        independent to a_current. We only want to augment the independent ones, otherwise we waste computing effort. 
        
        Second use case: After applying the templates to as many variable pairs as we wish, we call this function again, 
        to make sure all the matrices going into the SDP are in fact linearly independent. 
        
        """
        
        n_before = len(self.A_matrices)

        if remove_dependent:
            a_new = None #np.empty((len(new_templates), a_current.shape[1]))

        t1 = time.time()
        for j, new_template in enumerate(new_templates):

            # if we deal with a PolyMatrix, then it is the pattern augmented to all variables. 
            if isinstance(new_template, PolyMatrix):
                Ai = self.lifter.convert_polyrow_to_Apoly(new_template)

            else: # below is for newly found templates, which are bi vectors.
                Ai = self.lifter.convert_b_to_Apoly(new_template, self.mat_var_dict)
            self.A_matrices.append(Ai)
            self.templates.append(((j, self.mat_vars), new_template))
        print(f"time to convert {time.time() - t1:.3f}s")

        if not remove_dependent:
            keep_idx = range(len(self.A_matrices))
        else:
            t1 = time.time()
            # find which constraints are lin. dep.
            A_vec = self.get_a_current(sparse=True).T

            # Use sparse rank revealing QR
            # We "solve" a least squares problem to get the rank and permutations
            # This is the cheapest way to use sparse QR, since it does not require
            # explicit construction of the Q matrix. We can't do this with qr function
            # because the "just return R" option is not exposed.
            Z, R, E, rank = sqr.rz(A_vec, np.zeros((A_vec.shape[0],1)), tolerance=1e-10)
            # Sort the diagonal values. Note that SuiteSparse uses AMD/METIS ordering 
            # to acheive sparsity.
            r_vals = np.abs(R.diagonal())
            sort_inds = np.argsort(r_vals)[::-1]
            if rank < A_vec.shape[1]:
                print(f"get_independent_subset: keeping {rank}/{len(E)} templates")
            keep_idx = list(E[sort_inds[:rank]])

            # Sanity check, removed because too expensive. It almost always passed anyways.
            # Z, R, E, rank_full = sqr.rz(A_vec[:, keep_idx], np.zeros((A_vec.shape[0],1)), tolerance=1e-10)
            # if rank_full != rank:
            #     print(f"Warning: selected constraints did not pass lin. independence check. Rank is {rank_full}, should be {rank}.")
            print(f"time to find independent {time.time() - t1:.3f}s")

        t1 = time.time()
        self.A_matrices = [self.A_matrices[i] for i in keep_idx]
        self.templates = [self.templates[i] for i in keep_idx]

        t1 = time.time()
        error, bad_idx = self.lifter.test_constraints(self.A_matrices, errors="ignore", n_seeds=2)
        print(f"time to test {time.time() - t1:.3f}s")
        if len(bad_idx):
            print(f"removing {bad_idx} because high error, up to {error:.2e}")

            for idx in list(bad_idx)[::-1]: # reverse order to not mess up indexing
                del self.A_matrices[idx]
                del self.templates[idx]
        return len(self.A_matrices) - n_before

    @profile
    def run(self, use_known=True, verbose=False, plot=False, tightness="rank"):
        times = []
        while 1:

            # add one more variable to the list of variables to vary
            if not self.update_variables():
                print("no more variables to add")
                break

            print(f"======== {self.mat_vars} ========")
            time_dict = {"variables": self.mat_vars}

            print(f"-------- templates learning --------")
            # learn new templates, orthogonal to the ones found so far.
            t1 = time.time()
            n_new = self.learn_templates(use_known=use_known, plot=plot)
            if n_new == 0:
                print("new variables didn't have any effect")
                continue
            ttot = time.time() - t1
            time_dict["learn templates"] = ttot
            print(f"time:   {ttot:.3f}s")

            # apply the pattern to all landmarks
            if self.apply_templates_to_others:
                print(f"------- applying templates ---------")
                t1 = time.time()
                
                self.apply_templates()
                ttot = time.time() - t1
                time_dict["apply templates"] = ttot
                print(f"time:  {ttot:.3f}s")

            t1 = time.time()
            print(f"-------- checking tightness ----------")
            is_tight = self.is_tight(verbose=verbose, tightness=tightness)
            ttot = time.time() - t1
            time_dict["check tightness"] = ttot
            print(f"time: {ttot:.3f}s")
            times.append(time_dict)
            if is_tight: 
                break
        return times
            


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
                    index=list(templates_poly.variable_dict_j.keys()) + list(add_columns.keys()),
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

    def generate_templates_poly(self, templates=None, factor_out_parameters=False):
        if templates is None:
            templates = self.templates

        plot_rows = []
        plot_row_labels = []
        j = -1
        old_mat_vars = ""
        for key, new_pattern in templates:
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
    learner.save_templates(with_parameters=True, fname_root=fname_root)
    # learner.save_matrices(fname_root)
    plt.show()
    print("done")
