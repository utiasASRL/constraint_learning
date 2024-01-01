import time
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cert_tools.linalg_tools import find_dependent_columns, rank_project
from poly_matrix.poly_matrix import PolyMatrix

from lifters.state_lifter import StateLifter
from utils.constraint import Constraint
from utils.plotting_tools import import_plt, add_rectangles, add_colorbar
from utils.plotting_tools import savefig, plot_singular_values
from solvers.common import find_local_minimum
from solvers.common import solve_certificate
from solvers.common import solve_sdp_cvxpy
from solvers.sparse import solve_lambda
from solvers.sparse import bisection, brute_force


plt = import_plt()


# parameter of SDP solver
TOL = 1e-10

NOISE_SEED = 5

ADJUST_Q = True  # rescale Q matrix
PRIMAL = False  # use primal or dual formulation of SDP. Recommended is False, because of how MOSEK is set up.

FACTOR = 1.2  # oversampling factor.

TOL_REL_GAP = 1e-3
TOL_RANK_ONE = 1e7

PLOT_MAX_MATRICES = 10  # set to np.inf to plot all individual matrices.

USE_KNOWN = True
GLOBAL_THRESH = 1e-3

METHOD_NULL = "qrp"  # use svd or qp for comparison only, otherwise leave it at qrp


class Learner(object):
    """
    Class to incrementally learn and augment constraint templates until we reach tightness.
    """

    def __init__(
        self,
        lifter: StateLifter,
        variable_list: list,
        apply_templates: bool = True,
        noise: float = None,
        n_inits: int = 10,
        use_known: bool = USE_KNOWN,
    ):
        self.use_known = use_known
        self.noise = noise
        self.lifter = lifter
        self.variable_iter = iter(variable_list)

        self.apply_templates_to_others = apply_templates

        self.mat_vars = ["h"]

        # templates contains the learned "templates" of the form:
        # ((i, mat_vars), <i-th learned vector for these mat_vars variables, PolyRow form>)
        self.templates_poly_ = None  # for plotting only: all templats stacked in one

        # new representation, making sure we don't compute the same thing twice.
        self.constraints = []
        self.templates = []
        self.constraint_index = 0
        self.index_tested = set()
        self.templates_known = []
        self.templates_known_sub = []

        # list of dual costs
        self.df_tight = None
        self.ranks = []
        self.dual_costs = []
        self.variable_list = []
        self.solver_vars = None

        self.n_inits = n_inits
        self.reset_tightness_dict()

    def reset_tightness_dict(self):
        self.tightness_dict = {"rank": None, "cost": None}

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

    def check_violation(self, dual_cost):
        primal_cost = self.solver_vars["qcqp_cost"]
        if primal_cost is None:
            print("warning can't check violation, no primal cost.")
            return False
        return (dual_cost - primal_cost) / abs(dual_cost) > TOL_REL_GAP

    def duality_gap_is_zero(self, dual_cost, verbose=False, data_dict={}):
        primal_cost = self.solver_vars["qcqp_cost"]
        RDG = (primal_cost - dual_cost) / abs(dual_cost)
        res = RDG < TOL_REL_GAP
        data_dict["RDG"] = RDG
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

    def is_rank_one(self, eigs, verbose=False, data_dict={}):
        SVR = eigs[0] / eigs[1]
        data_dict["SVR"] = SVR
        res = SVR > TOL_RANK_ONE
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

    def is_tight(self, verbose=False, data_dict={}, tightness=None):
        if tightness is None:
            tightness = self.lifter.TIGHTNESS

        if self.tightness_dict[tightness] is not None:
            return self.tightness_dict[tightness]

        A_b_list_all = self.get_A_b_list()
        A_list = [A for A, __ in A_b_list_all[1:]]  # for debugging only

        B_list = self.lifter.get_B_known()
        X, info = self._test_tightness(A_b_list_all, B_list, verbose=verbose)

        self.solver_vars["X"] = X

        self.dual_costs.append(info["cost"])
        self.variable_list.append(self.mat_vars)

        data_dict["q"] = self.solver_vars["qcqp_cost"]
        data_dict.update(self.solver_vars)

        if info["cost"] is None:
            self.ranks.append(np.zeros(self.lifter.get_dim_x()))
            print(f"Warning: solver failed with message: {info['msg']}")
            max_error, bad_list = self.lifter.test_constraints(A_list, errors="print")
            print("Maximum error:", max_error)
            return False
        elif self.check_violation(info["cost"]):
            self.ranks.append(np.zeros(A_list[0].shape[0]))
            print(
                f"Dual cost higher than QCQP: d={info['cost']:.2e}, q={self.solver_vars['qcqp_cost']:.2e}"
            )
            print(
                "Usually this means that MOSEK tolerances are too loose, or that there is a mistake in the constraints."
            )
            max_error, bad_list = self.lifter.test_constraints(A_list, errors="print")
            print("Maximum feasibility error at random x:", max_error)

            print(
                "It can also mean that we are not sampling enough of the space close to the true solution."
            )
            tol = 1e-10
            xhat = self.solver_vars["xhat"]
            max_error = -np.inf
            for Ai in A_list:
                error = xhat.T @ Ai @ xhat

                errorX = np.trace(X @ Ai)
                max_error = max(errorX, max_error)
                if abs(error) > tol:
                    print(
                        f"Feasibility error too high! xAx:{error:.2e}, <X,A>:{errorX:.2e}"
                    )
            print(f"Maximum feasibility error at solution x: {max_error}")

        data_dict["d"] = info["cost"]

        # sanity check
        final_cost = np.trace(self.solver_vars["Q"] @ X)
        if abs(final_cost - info["cost"]) / info["cost"] >= 1e-1:
            print(
                f"Warning: cost is inconsistent: {final_cost:.3e}, {info['cost']:.3e}"
            )

        eigs = np.linalg.eigvalsh(X)[::-1]
        self.ranks.append(eigs)

        if self.lifter.robust:
            x_dim = self.lifter.d + self.lifter.d**2 + 1
            wi = X[0, x_dim::x_dim]
            print("should be plus or minus ones:", wi.round(4))

        cost_tight = self.duality_gap_is_zero(
            info["cost"], verbose=tightness == "cost", data_dict=data_dict
        )
        rank_tight = self.is_rank_one(
            eigs, verbose=tightness == "rank", data_dict=data_dict
        )
        self.tightness_dict["rank"] = rank_tight
        self.tightness_dict["cost"] = cost_tight

        if tightness == "rank":
            return rank_tight
        elif tightness == "cost":
            return cost_tight

    def get_A_b_list(self):
        A_known = []
        if self.use_known:
            # A_known = self.lifter.get_A_known()
            A_known += [constraint.A_sparse_ for constraint in self.templates_known]
        A_list = A_known + [constraint.A_sparse_ for constraint in self.constraints]
        A_b_list_all = self.lifter.get_A_b_list(A_list)
        return A_b_list_all

    def generate_minimal_subset(
        self,
        reorder=False,
        tightness="rank",
        use_last=None,
        use_bisection=False,
    ):
        def function(A_b_list_here, df_data):
            """Function for bisection or brute_force"""
            if len(A_b_list_here) in df_data.keys():
                new_data = df_data[len(A_b_list_here)]
            else:
                new_data = {"lifter": str(self.lifter), "reorder": reorder}
                X, info = self._test_tightness(
                    A_b_list_here, B_list=B_list, verbose=False
                )
                dual_cost = info["cost"]
                new_data["dual cost"] = dual_cost
                if dual_cost is None:
                    print(f"{len(A_b_list_here)}: solver error? msg: {info['msg']}")
                    new_data["eigs"] = np.full(self.lifter.get_dim_X(), np.nan)
                    new_data["cost_tight"] = False
                    new_data["rank_tight"] = False
                    df_data[len(A_b_list_here)] = deepcopy(new_data)
                    return False

                elif self.duality_gap_is_zero(dual_cost):
                    print(f"{len(A_b_list_here)}: cost-tight")
                    new_data["cost_tight"] = True
                else:
                    print(f"{len(A_b_list_here)}: not cost-tight yet")
                    new_data["cost_tight"] = False

                eigs = np.linalg.eigvalsh(X)[::-1]
                new_data["eigs"] = eigs
                if self.is_rank_one(eigs):
                    print(f"{len(A_b_list_here)}: rank-tight")
                    new_data["rank_tight"] = True
                else:
                    new_data["rank_tight"] = False
                    print(f"{len(A_b_list_here)}: not rank-tight yet")
                df_data[len(A_b_list_here)] = deepcopy(new_data)

            if tightness == "rank":
                return new_data["rank_tight"]
            else:
                return new_data["cost_tight"]

        A_b_list_all = self.get_A_b_list()
        B_list = self.lifter.get_B_known()

        force_first = 1
        if self.use_known:
            force_first += len(self.templates_known)

        if reorder:
            # find the importance of each constraint
            __, lamdas = solve_lambda(
                self.solver_vars["Q"],
                A_b_list_all,
                self.solver_vars["xhat"],
                B_list=B_list,
                force_first=force_first,
                tol=TOL,
                adjust=True,
                verbose=False,
            )
            if lamdas is None:
                print("Warning: problem doesn't have feasible solution!")
                print("Sanity checks:")
                B_list = self.lifter.get_B_known()
                X, info = self._test_tightness(A_b_list_all, B_list, verbose=False)
                assert info["msg"] == "converged"
                xhat_from_X, __ = rank_project(X, p=1)
                xhat = self.solver_vars["xhat"]
                print("xhat error:", xhat - xhat_from_X)
                print("Hx=", info["H"] @ xhat)
                print("Hx_from_X=", info["H"] @ xhat_from_X)
                eigs = np.linalg.eigvalsh(info["H"].toarray())
                print("eigs of H", eigs)
                return None

            print("found valid lamdas")
            # order the redundant constraints by importance
            redundant_idx = np.argsort(np.abs(lamdas[force_first:]))[::-1]
            # example: force_first is 4
            # [0, 1, 2, 3], 4 + [1, 0, 2] = [0, 1, 2, 3, 5, 4, 6]
            sorted_idx = np.r_[np.arange(force_first), force_first + redundant_idx]
        else:
            # if force_first is 7, then
            # sorted idx is simply 7, 8, 9, ..., 20
            sorted_idx = range(len(A_b_list_all))

        inputs = [A_b_list_all[idx] for idx in sorted_idx]

        B_list = self.lifter.get_B_known()
        df_data = []

        if use_last is None:
            start_idx = force_first
        else:
            start_idx = max(len(inputs) - use_last, force_first)

        df_data = {}
        if use_bisection:
            bisection(
                function, (inputs, df_data), left_num=start_idx, right_num=len(inputs)
            )
        else:
            brute_force(
                function, (inputs, df_data), left_num=start_idx, right_num=len(inputs)
            )

        df_tight = pd.DataFrame(df_data.values(), index=df_data.keys())
        if self.df_tight is None:
            self.df_tight = df_tight
        else:
            self.df_tight = pd.concat([self.df_tight, df_tight], axis=0)

        minimal_indices = []
        if tightness == "cost":
            min_num = df_tight[df_tight.cost_tight == True].index.min()
        elif tightness == "rank":
            min_num = df_tight[df_tight.rank_tight == True].index.min()
        if not np.isnan(min_num):
            minimal_indices = list(sorted_idx[:min_num])
        return minimal_indices

    def find_local_solution(self, verbose=False, plot=False):
        np.random.seed(NOISE_SEED)
        Q, y = self.lifter.get_Q(noise=self.noise)
        qcqp_that, qcqp_cost, info = find_local_minimum(
            self.lifter, y=y, verbose=verbose, n_inits=self.n_inits, plot=plot
        )
        if qcqp_cost is not None:
            xhat = self.lifter.get_x(qcqp_that)
            self.solver_vars = dict(Q=Q, y=y, qcqp_cost=qcqp_cost, xhat=xhat)
            self.solver_vars.update(info)
            self.solver_vars.update(self.lifter.get_error(qcqp_that))
            return True
        self.solver_vars = dict(Q=Q, y=y, qcqp_cost=qcqp_cost, xhat=None)

    def find_global_solution(self, data_dict={}):
        if self.tightness_dict["rank"] is True:
            X = self.solver_vars["X"]
            x, info = rank_project(X, p=1)
            print("rank projection info:", info)
        else:
            """Try to solve dual problem."""
            xhat = self.solver_vars["xhat"]

            A_b_list_all = self.get_A_b_list()
            H, info = solve_certificate(
                self.solver_vars["Q"], A_b_list_all, xhat, adjust=True
            )
            if info["eps"] is not None:
                print(f"global solution eps: {info['eps']:.2e}")
                cert = abs(info["eps"]) <= GLOBAL_THRESH
                data_dict["global solution cert"] = cert

            if info["eps"] and cert:
                x = xhat
            else:
                x = None

            # sanity check: solve wtih local minimum
            keys = [key for key in data_dict.keys() if key.startswith("local solution")]
            for key in keys:
                x_local = data_dict[key]
                x_local = self.lifter.get_x(theta=x_local)
                H, info = solve_certificate(
                    self.solver_vars["Q"], A_b_list_all, x_local, adjust=True
                )
                if info["eps"] is not None:
                    print(f"local solution eps: {info['eps']:.2e}")
                    cert = abs(info["eps"]) <= GLOBAL_THRESH
                    data_dict[key + " cert"] = cert

        if x is not None:
            theta = self.lifter.get_theta(x)
            cost = self.lifter.get_cost(theta, self.solver_vars["y"])
            data_dict["global theta"] = theta
            data_dict["global cost"] = cost
            return True
        return False

    def _test_tightness(self, A_b_list_all, B_list=[], verbose=False):
        if self.solver_vars is None:
            self.find_local_solution(verbose=verbose)

        # compute lambas by solving dual problem
        X, info = solve_sdp_cvxpy(
            self.solver_vars["Q"],
            A_b_list_all,
            B_list,
            adjust=ADJUST_Q,
            verbose=verbose,
            primal=PRIMAL,
            tol=TOL,
        )  # , rho_hat=qcqp_cost)
        return X, info

    def update_variables(self):
        # add new variable to the list of variables to study
        try:
            self.mat_vars = next(self.variable_iter)
            return True
        except StopIteration:
            return False

    def extract_known_templates(self):
        templates_known_sub = []
        for c in self.templates_known:
            var_subset = set(c.A_poly_.get_variables())
            if var_subset.issubset(self.mat_var_dict):
                templates_known_sub.append(c)

        new_index_set = set([t.index for t in templates_known_sub])
        old_index_set = set([t.index for t in self.templates_known_sub])
        diff_index_set = new_index_set.difference(old_index_set)

        self.templates_known_sub = templates_known_sub
        # all_idx = new_index_set.union(old_index_set)
        # for idx in all_idx:
        #    matches = [t for t in templates_known_sub + self.templates_known_sub if t.index == idx]
        #    self.templates_known_sub.append(matches[0])
        return len(diff_index_set)

    # @profile
    def learn_templates(self, plot=False, data_dict=None):
        templates = []

        t1 = time.time()
        Y = self.lifter.generate_Y(var_subset=self.mat_vars, factor=FACTOR)
        if self.use_known:
            a_vectors = []
            for c in self.templates:
                ai = self.lifter.get_vec(c.A_poly_.get_matrix(self.mat_var_dict))
                bi = self.lifter.augment_using_zero_padding(ai, self.mat_var_dict)
                a_vectors.append(bi)
            for c in self.templates_known_sub:
                ai = self.lifter.get_vec(c.A_poly_.get_matrix(self.mat_var_dict))
                bi = self.lifter.augment_using_zero_padding(ai, self.mat_var_dict)
                a_vectors.append(bi)
            Y = np.vstack([Y] + a_vectors)

        if plot:
            fig, ax = plt.subplots()

        print(f"data matrix Y has shape {Y.shape} ")
        for i in range(self.lifter.N_CLEANING_STEPS + 1):
            print(f"cleaning step {i+1}/{self.lifter.N_CLEANING_STEPS+1}...", end="")
            basis_new, S = self.lifter.get_basis(Y, method=METHOD_NULL)
            print(f"...done")
            corank = basis_new.shape[0]
            if corank > 0:
                self.lifter.test_S_cutoff(S, corank)

            bad_idx = self.lifter.clean_Y(basis_new, Y, S, plot=False)

            if plot:
                if len(bad_idx):
                    plot_singular_values(
                        S, eps=self.lifter.EPS_SVD, label=f"run {i}", ax=ax
                    )
                else:
                    plot_singular_values(S, eps=self.lifter.EPS_SVD, ax=ax, label=None)

            if len(bad_idx) > 0:
                Y = np.delete(Y, bad_idx, axis=0)
            else:
                break

        if basis_new.shape[0]:
            templates += [
                Constraint.init_from_b(
                    index=self.constraint_index + i,
                    mat_var_dict=self.mat_var_dict,
                    b=b,
                    lifter=self.lifter,
                    convert_to_polyrow=self.apply_templates_to_others,
                    known=False,
                )
                for i, b in enumerate(basis_new)
            ]
            self.constraint_index += basis_new.shape[0]

            # we assume that all known constraints are linearly independent, and also
            # that all known+previously found constraints are linearly independent.
            indep_templates = self.clean_constraints(
                new_constraints=templates,
                before_constraints=self.templates,
                remove_dependent=True,
                remove_imprecise=False,
            )

        if data_dict is not None:
            ttot = time.time() - t1
            data_dict["t learn templates"] = ttot
            data_dict["n rank"] = Y.shape[1] - corank
            data_dict["n nullspace"] = corank

        if len(templates) > 0:
            n_all = len(indep_templates)
            n_new = n_all - len(self.templates)
            self.templates = indep_templates
            return n_new, n_all
        return 0, len(self.constraints)

    def apply_templates(self, reapply_all=False):
        # the new templates are all the ones corresponding to the new matrix variables.
        new_constraints = []
        for template in self.templates:
            constraints = self.lifter.apply_template(
                template.polyrow_b_,
                n_landmarks=self.lifter.n_landmarks,
            )
            template.applied_list = [
                Constraint.init_from_polyrow_b(
                    index=self.constraint_index + i,
                    polyrow_b=new_constraint,
                    lifter=self.lifter,
                    template_idx=template.index,
                    known=template.known,
                )
                for i, new_constraint in enumerate(constraints)
            ]
            new_constraints += template.applied_list
            self.constraint_index += len(constraints)

        if not len(new_constraints):
            return 0, 0

        # determine which of these constraints are actually independent, after reducing them to ai.
        indep_constraints = self.clean_constraints(
            new_constraints=new_constraints,
            before_constraints=[],
            remove_dependent=True,
            remove_imprecise=False,
        )
        n_all = len(indep_constraints)
        n_new = n_all - len(self.constraints)
        self.constraints = indep_constraints
        return n_new, n_all

    def clean_constraints(
        self,
        new_constraints,
        before_constraints,
        remove_dependent=True,
        remove_imprecise=True,
    ):
        """
        This function is used in two different ways.

        First use case: Given the new templates, in b-PolyRow form, we determine which of the templates are actually
        independent to a_current. We only want to augment the independent ones, otherwise we waste computing effort.

        Second use case: After applying the templates to as many variable pairs as we wish, we call this function again,
        to make sure all the matrices going into the SDP are in fact linearly independent.
        """
        constraints = before_constraints + new_constraints
        if remove_dependent:
            # find which constraints are lin. dep.
            A_vec = sp.vstack(
                [constraint.a_full_ for constraint in constraints], format="coo"
            ).T

            # make sure that matrix is tall (we have less constraints than number of dimensions of x)
            if A_vec.shape[0] < A_vec.shape[1]:
                print("Warning: fat matrix.")

            bad_idx = find_dependent_columns(A_vec)
            if len(bad_idx):
                for idx in sorted(bad_idx)[::-1]:
                    del constraints[idx]

        if remove_imprecise:
            error, bad_idx = self.lifter.test_constraints(
                [c.A_sparse_ for c in constraints if c.index not in self.index_tested],
                errors="ignore",
                n_seeds=2,
            )
            self.index_tested = self.index_tested.union([c.index for c in constraints])
            if len(bad_idx):
                print(f"removing {bad_idx} because high error, up to {error:.2e}")
                for idx in list(sorted(bad_idx))[
                    ::-1
                ]:  # reverse order to not mess up indexing
                    del constraints[idx]
        return constraints

    def create_known_templates(self, unroll=False):
        # TODO(FD) we should not always recompute from scratch, but it's not very expensive so it's okay
        n_known = 0
        self.templates_known = []
        if unroll:
            target_dict = self.lifter.get_var_dict_unroll()
        else:
            target_dict = self.lifter.var_dict

        for i, Ai in enumerate(self.lifter.get_A_known(target_dict, output_poly=True)):
            template = Constraint.init_from_A_poly(
                lifter=self.lifter,
                A_poly=Ai,
                known=True,
                index=self.constraint_index,
                template_idx=self.constraint_index,
                mat_var_dict=self.lifter.var_dict,
            )
            self.constraint_index += 1
            self.templates_known.append(template)
            n_known += 1
        print(f"using {n_known} new known constraints")
        return n_known

    def scale_templates(self, learner, new_order, data_dict={}):
        """Use the templates in learner to populate the own templates etc."""
        all_constraints = learner.templates_known + learner.constraints
        template_indices = sorted(
            [t.index for t in learner.templates + learner.templates_known]
        )
        t1 = time.time()
        if new_order is not None:
            template_unique_idx = set()

            for i in new_order:
                # the first constraint ALWAYS corresponds to A0, whichs not part of our templates.
                if i > 0:
                    new_index = all_constraints[i - 1].template_idx
                    assert new_index in template_indices
                    template_unique_idx.add(new_index)

            self.templates = []
            for t in template_unique_idx:
                template = [
                    template
                    for template in learner.templates + learner.templates_known
                    if template.index == t
                ][0]

                if not template.known:
                    # (make sure the dimensions of the constraints are correct)
                    scaled_template = template.scale_to_new_lifter(self.lifter)
                    self.templates.append(scaled_template)
        else:
            self.templates = learner.templates
        # apply the templates
        data_dict[f"n templates"] = len(self.templates)
        self.create_known_templates()
        n_new, n_total = self.apply_templates(reapply_all=True)
        data_dict[f"n constraints"] = n_total
        data_dict[f"t create constraints"] = time.time() - t1

        # TODO(FD) below should not be necessary
        # self.constraints = self.clean_constraints(
        #    self.constraints, [], remove_imprecise=False
        # )

    # @profile
    def run(self, verbose=False, plot=False):
        data = []
        success = False

        if self.use_known:
            self.create_known_templates()

        while 1:
            # add one more variable to the list of variables to vary
            if not self.update_variables():
                print("no more variables to add")
                break
            print(f"======== {self.mat_vars} ========")

            n_new = 0
            if self.use_known:
                n_new += self.extract_known_templates()

            data_dict = {"variables": self.mat_vars}
            data_dict["n dims"] = self.lifter.get_dim_Y(self.mat_vars)

            print(f"-------- templates learning --------")
            # learn new templates, orthogonal to the ones found so far.
            n_new_learned, n_all = self.learn_templates(plot=plot, data_dict=data_dict)
            n_new += n_new_learned
            print(
                f"found {n_new_learned} learned templates, new total learned: {n_all} "
            )
            data_dict["n templates"] = (
                len(self.templates) + len(self.templates_known) + 1
            )
            if n_new == 0:
                print("new variables didn't have any effect")
                data.append(data_dict)
                continue

            # apply the pattern to all landmarks
            if self.apply_templates_to_others:
                print(f"------- applying templates ---------")
                t1 = time.time()
                n_new, n_all = self.apply_templates(reapply_all=True)
                print(f"found {n_new} independent constraints, new total: {n_all} ")
                ttot = time.time() - t1

                data_dict["n constraints"] = n_all + len(self.templates_known) + 1
                data_dict["t apply templates"] = ttot
            else:
                self.constraints = []
                for temp in self.templates:
                    con = deepcopy(temp)
                    con.template_idx = temp.index
                    self.constraints.append(con)

            t1 = time.time()
            print(f"-------- checking tightness ----------")
            self.reset_tightness_dict()
            is_tight = self.is_tight(verbose=False, data_dict=data_dict)
            ttot = time.time() - t1
            data_dict["t check tightness"] = ttot
            data.append(data_dict)
            if is_tight:
                success = True
                break
        return data, success

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

        variable_dict_j = list(templates_poly.variable_dict_j.keys())
        for i, key_i in enumerate(templates_poly.variable_dict_i):
            data = {j: float(val) for j, val in templates_poly.matrix[key_i].items()}
            for key, idx_list in add_columns.items():
                # if the list is not empty, then indicate which constraints are required.
                if idx_list is not None and len(idx_list):
                    idx_list = list(idx_list)
                    try:
                        data[key] = idx_list.index(i)
                    except Exception:
                        data[key] = -1
                # if the list is empty, all of them are required (and more)
                else:
                    data[key] = 1.0
            series.append(
                pd.Series(
                    data,
                    index=variable_dict_j + list(add_columns.keys()),
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
            constraints = self.templates_known + self.constraints

        plot_rows = []
        plot_row_labels = []
        j = -1
        old_mat_vars = ""
        for constraint in constraints:
            mat_vars = constraint.mat_var_dict
            i = constraint.index
            if factor_out_parameters:  # use a and not b.
                if constraint.polyrow_a_ is not None:
                    plot_rows.append(constraint.polyrow_a_)
                else:
                    if constraint.a_ is not None:
                        polyrow_a = self.lifter.convert_a_to_polyrow(
                            constraint.a_, constraint.mat_var_dict
                        )
                    elif constraint.a_full_ is not None:
                        polyrow_a = self.lifter.convert_a_to_polyrow(
                            constraint.a_full_, constraint.mat_var_dict
                        )
                    plot_rows.append(polyrow_a)
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
            mat_vars, force_parameters_off=factor_out_parameters
        )
        return templates_poly

    def save_sorted_templates(
        self, df, fname_root="", title="", drop_zero=False, simplify=True
    ):
        from utils.plotting_tools import plot_basis

        # convert to poly matrix for plotting purposes only.
        poly_matrix = PolyMatrix(symmetric=False)
        keys_j = []
        keys_i = []
        for i, row in df.iterrows():
            for k, val in row[~row.isna()].items():
                if "order" in k or "required" in k:
                    continue
                poly_matrix[i, k] = val
                keys_j.append(k)
                keys_i.append(i)

        variables_j = self.lifter.var_dict_row(
            var_subset=self.lifter.var_dict, force_parameters_off=False
        )
        assert set(keys_j).issubset(variables_j)
        if drop_zero:
            variables_j = {k: v for k, v in variables_j.items() if k in keys_j}
        variables_i = {i: 1 for i in keys_i}
        fig, ax = plot_basis(
            poly_matrix,
            variables_j=variables_j,
            variables_i=variables_i,
            discrete=True,
        )
        ax.set_yticklabels([])
        ax.set_yticks([])
        if simplify:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            new_xticks = []
            for lbl in ax.get_xticklabels():
                lbl = lbl.get_text()
                if "_" in lbl:  # avoid double subscript
                    new_lbl = f"${lbl.replace('l-', '').replace(':', '^')}$"
                else:
                    new_lbl = f"${lbl.replace('l-', '').replace(':', '_')}$"
                new_xticks.append(new_lbl)
            ax.set_xticklabels(new_xticks, fontsize=7)

        # plot a red vertical line at each new block of parameters.
        params = [v.split("-")[0] for v in variables_j]
        old_param = params[0]
        for i, p in enumerate(params):
            if p != old_param:
                ax.axvline(i, color="red", linewidth=1.0)
                ax.annotate(
                    text=f"${p.replace(':0', '^x').replace(':1', '^y').replace('l.','').replace('.','')}$",
                    xy=[i, 0],
                    fontsize=8,
                    color="red",
                )
                old_param = p
        ax.set_title(title)
        if "required (sorted)" in df.columns:
            from matplotlib.patches import Rectangle

            for i, (__, row) in enumerate(df.iterrows()):
                if row["required (sorted)"] < 0:
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
        fig.set_size_inches(6, 3)
        im0 = axs[0].matshow(
            1 - A_agg.toarray(), vmin=0, vmax=1, cmap="gray"
        )  # 1 (white) is empty, 0 (black) is nonempty

        import matplotlib

        vmin = min(-np.max(Q), np.min(Q))
        vmax = max(np.max(Q), -np.min(Q))
        norm = matplotlib.colors.SymLogNorm(10**-5, vmin=vmin, vmax=vmax)
        im1 = axs[1].matshow(Q, norm=norm)

        for ax in axs:
            add_rectangles(ax, self.lifter.var_dict)

        from utils.plotting_tools import add_colorbar

        add_colorbar(fig, axs[1], im1, nticks=3)
        # only for dimensions
        add_colorbar(fig, axs[0], im0, visible=False)
        if fname_root != "":
            savefig(fig, fname_root + "_matrices-sparisty.png")
        return fig, axs

    def save_matrices_poly(
        self,
        A_matrices=None,
        n_matrices=5,
        fname_root="",
        reduced_mode=False,
        save_individual=False,
        max_matrices=PLOT_MAX_MATRICES,
    ):
        if A_matrices is None:
            A_matrices = self.A_matrices

        n_rows = n_matrices // 10 + 1
        n_cols = min(n_matrices, 10)
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
        fig.set_size_inches(5 * n_cols / n_rows, 5)
        axs = axs.flatten()
        i = 0
        for i, A_poly in enumerate(A_matrices):
            if reduced_mode:
                sorted_i = sorted(A_poly.variable_dict_i.keys())
            else:
                sorted_i = self.lifter.var_dict_unroll
            from utils.plotting_tools import initialize_discrete_cbar

            plot_axs = []
            if i < n_matrices:
                plot_axs.append(axs[i])

            if save_individual and (i < max_matrices):
                figi, axi = plt.subplots()
                figi.set_size_inches(3, 3)
                plot_axs.append(axi)

            A_sparse = A_poly.get_matrix(sorted_i)
            cmap, norm, colorbar_yticks = initialize_discrete_cbar(A_sparse.data)

            for ax in plot_axs:
                im = ax.matshow(A_sparse.toarray(), cmap=cmap, norm=norm)
                add_rectangles(ax, self.lifter.var_dict)
                cax = add_colorbar(fig, ax, im, size=0.1)
                cax.set_yticklabels(colorbar_yticks)

            if save_individual and (i < max_matrices):
                savefig(figi, fname_root + f"_matrix{i}.pdf")
        for ax in axs[i + 1 :]:
            ax.axis("off")

        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        # if fname_root != "":
        #    savefig(fig, fname_root + "_matrices-poly.png")
        return fig, axs
