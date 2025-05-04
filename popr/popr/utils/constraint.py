import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from poly_matrix.poly_matrix import PolyMatrix
from popr.utils.common import get_vec
from popr.utils.plotting_tools import plot_basis


def remove_dependent_constraints(constraints, verbose=False):
    from cert_tools.linalg_tools import find_dependent_columns

    # find which constraints are lin. dep.
    A_vec = sp.vstack(
        [constraint.a_full_ for constraint in constraints], format="coo"
    ).T

    bad_idx = find_dependent_columns(A_vec, verbose=verbose)
    if len(bad_idx):
        np.testing.assert_allclose(bad_idx, sorted(bad_idx))
        # important: by changing the order we
        for idx in sorted(bad_idx)[::-1]:
            del constraints[idx]


def generate_poly_matrix(constraints, factor_out_parameters=False, lifter=None):
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
                    assert (
                        lifter is not None
                    ), "Need to provide lifter because a_ is not defined"
                    polyrow_a = lifter.convert_a_to_polyrow(
                        constraint.a_, constraint.mat_var_dict
                    )
                elif constraint.a_full_ is not None:
                    assert (
                        lifter is not None
                    ), "Need to provide lifter because a_full_ is not defined"
                    polyrow_a = lifter.convert_a_to_polyrow(
                        constraint.a_full_, constraint.mat_var_dict
                    )
                plot_rows.append(polyrow_a)
        else:
            if constraint.polyrow_b_ is not None:
                plot_rows.append(constraint.polyrow_b_)
            else:
                assert (
                    lifter is not None
                ), "Need to provide lifter because polyrow_b_ is not defined."
                plot_rows.append(
                    lifter.convert_b_to_polyrow(
                        constraint.b_, mat_vars, constraint.mat_param_dict
                    )
                )

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
    return templates_poly


def plot_poly_matrix(
    poly_matrix, variables_j=None, variables_i=None, simplify=True, ax=None, hom="h"
):
    if variables_i is None:
        variables_i = poly_matrix.variable_dict_i
    if variables_j is None:
        variables_j = poly_matrix.variable_dict_j

    if ax is None:
        fig, ax = plt.subplots()
    # plot the templates stored in poly_matrix.

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
                new_lbl = f"${lbl.replace(f'{hom}.', '').replace(':', '^')}$"
            else:
                new_lbl = f"${lbl.replace(f'{hom}.', '').replace(':', '_')}$"
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
    return fig, ax


class Constraint(object):
    """
    This class serves the main purpose of not recomputing representations of constraints more than once.
    """

    def __init__(
        self,
        index=0,
        polyrow_a=None,
        polyrow_b=None,
        A_poly=None,
        A_sparse=None,
        b=None,
        a=None,
        a_full=None,
        b_full=None,
        mat_var_dict=None,
        mat_param_dict=None,
        known=False,
        template_idx=0,
    ):
        self.index = index
        self.mat_var_dict = mat_var_dict
        self.mat_param_dict = mat_param_dict

        self.b_ = b
        self.polyrow_b_ = polyrow_b
        self.polyrow_a_ = polyrow_a
        self.A_poly_ = A_poly
        self.A_sparse_ = A_sparse
        self.a_ = a
        self.b_full_ = b_full
        self.a_full_ = a_full

        self.known = known
        self.template_idx = template_idx

        # list of applied constraints derived from this constraint.
        self.applied_list = []

    @staticmethod
    # @profile
    def init_from_b(
        index: int,
        b: np.ndarray,
        mat_var_dict: dict,
        lifter=None,
        mat_param_dict: dict = None,
        convert_to_polyrow: bool = True,
        known: bool = True,
        template_idx: int = None,
    ):
        a = None
        A_sparse = None
        a_full = None
        if lifter is not None:
            a = lifter.get_reduced_a(
                b, var_subset=mat_var_dict, param_subset=mat_param_dict, sparse=True
            )
            A_sparse = lifter.get_mat(a, var_dict=mat_var_dict, sparse=True)
            a_full = get_vec(A_sparse, sparse=True)
            if a_full is None:
                return None
        if convert_to_polyrow:
            A_poly, __ = PolyMatrix.init_from_sparse(
                A_sparse, var_dict=lifter.var_dict, unfold=True
            )
            polyrow_b = lifter.convert_b_to_polyrow(
                b, mat_var_dict, param_subset=mat_param_dict
            )
        else:
            A_poly = None
            polyrow_b = None
        return Constraint(
            index=index,
            a=a,
            b=b,
            A_sparse=A_sparse,
            A_poly=A_poly,
            polyrow_b=polyrow_b,
            a_full=a_full,
            mat_var_dict=mat_var_dict,
            mat_param_dict=mat_param_dict,
            known=known,
            template_idx=template_idx,
        )

    @staticmethod
    def init_from_A_poly(
        lifter,
        A_poly: PolyMatrix,
        mat_var_dict: dict,
        known: bool = False,
        index: int = 0,
        template_idx: int = None,
        compute_polyrow_b=False,
    ):
        Ai_sparse_small = A_poly.get_matrix(variables=mat_var_dict)
        ai = get_vec(Ai_sparse_small, correct=True)
        bi = lifter.augment_using_zero_padding(ai)
        if compute_polyrow_b:
            polyrow_b = lifter.convert_b_to_polyrow(bi, mat_var_dict)
        else:
            polyrow_b = None
        polyrow_a = lifter.convert_a_to_polyrow(ai, mat_var_dict)
        Ai_sparse = A_poly.get_matrix(variables=lifter.var_dict)
        return Constraint(
            a=ai,
            polyrow_a=polyrow_a,
            b=bi,
            polyrow_b=polyrow_b,
            A_poly=A_poly,
            A_sparse=Ai_sparse,
            known=known,
            index=index,
            mat_var_dict=mat_var_dict,
            template_idx=template_idx,
        )

    @staticmethod
    def init_from_polyrow_b(
        polyrow_b: PolyMatrix,
        lifter,
        index: int = 0,
        known: bool = False,
        template_idx: int = None,
        mat_var_dict: dict = None,
    ):
        if mat_var_dict is None:
            mat_var_dict = lifter.var_dict
        A_poly = lifter.convert_polyrow_to_Apoly(polyrow_b)
        dict_unroll = lifter.get_var_dict(mat_var_dict, unroll_keys=True)
        A_sparse = A_poly.get_matrix(dict_unroll)
        a_full = get_vec(A_sparse, sparse=True)
        return Constraint(
            index=index,
            A_poly=A_poly,
            polyrow_b=polyrow_b,
            A_sparse=A_sparse,
            a_full=a_full,
            known=known,
            template_idx=template_idx,
            mat_var_dict=mat_var_dict,
        )

    def scale_to_new_lifter(self, lifter):
        if self.known:
            # known matrices are stored in origin variables, not unrolled form
            self.A_sparse_ = self.A_poly_.get_matrix(lifter.var_dict)
            self.a_full_ = get_vec(self.A_sparse_, sparse=True)

        else:
            # known matrices are stored in origin variables, not unrolled form
            target_dict_unroll = lifter.get_var_dict(unroll_keys=True)
            self.A_sparse_ = self.A_poly_.get_matrix(target_dict_unroll)
            self.a_full_ = get_vec(self.A_sparse_, sparse=True)
        return self
