import numpy as np
import scipy.sparse as sp

from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix


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
        known=False,
        template_idx=0,
    ):
        self.index = index
        self.mat_var_dict = mat_var_dict

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
        lifter: StateLifter,
        convert_to_polyrow: bool = True,
        known: bool = True,
        template_idx: int = None,
    ):
        a = lifter.get_reduced_a(b, mat_var_dict, sparse=True)
        A_sparse = lifter.get_mat(a, var_dict=mat_var_dict, sparse=True)
        a_full = lifter.get_vec(A_sparse, sparse=True)
        # a_full = lifter.augment_using_zero_padding(a, mat_var_dict)
        if convert_to_polyrow:
            # A_poly = lifter.convert_b_to_Apoly(b, mat_var_dict)
            A_poly, __ = PolyMatrix.init_from_sparse(
                A_sparse, var_dict=lifter.var_dict, unfold=True
            )
            polyrow_b = lifter.convert_b_to_polyrow(b, mat_var_dict)
            # polyrow_a = lifter.convert_a_to_polyrow(a, mat_var_dict)
            return Constraint(
                index=index,
                a=a,
                b=b,
                A_sparse=A_sparse,
                A_poly=A_poly,
                polyrow_b=polyrow_b,
                # polyrow_a=polyrow_a,
                a_full=a_full,
                mat_var_dict=mat_var_dict,
                known=known,
                template_idx=template_idx,
            )
        return Constraint(
            index=index,
            a=a,
            b=b,
            A_sparse=A_sparse,
            a_full=a_full,
            mat_var_dict=mat_var_dict,
            known=known,
            template_idx=template_idx,
        )

    @staticmethod
    def init_from_A_poly(
        lifter: StateLifter,
        A_poly: PolyMatrix,
        mat_var_dict: dict,
        known: bool = False,
        index: int = 0,
        template_idx: int = None,
    ):
        Ai_sparse_small = A_poly.get_matrix(variables=mat_var_dict)
        ai = lifter.get_vec(Ai_sparse_small, correct=True)
        bi = lifter.augment_using_zero_padding(ai)
        # Below takes unnecessarily long, but is currently only required for plotting.
        # We set it to None and calculate it on demand.
        # polyrow_b = lifter.convert_b_to_polyrow(bi, mat_var_dict)
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
        lifter: StateLifter,
        index: int = 0,
        known: bool = False,
        template_idx: int = None,
        mat_var_dict: dict = None,
    ):
        if mat_var_dict is None:
            mat_var_dict = lifter.var_dict
        A_poly = lifter.convert_polyrow_to_Apoly(polyrow_b)
        dict_unroll = lifter.get_var_dict_unroll(mat_var_dict)
        A_sparse = A_poly.get_matrix(dict_unroll)
        a_full = lifter.get_vec(A_sparse, sparse=True)
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

    def scale_to_new_lifter(self, lifter: StateLifter):
        if self.known:
            # known matrices are stored in origin variables, not unrolled form
            self.A_sparse_ = self.A_poly_.get_matrix(lifter.var_dict)
            self.a_full_ = lifter.get_vec(self.A_sparse_, sparse=True)

        else:
            # known matrices are stored in origin variables, not unrolled form
            target_dict_unroll = lifter.var_dict_unroll
            self.A_sparse_ = self.A_poly_.get_matrix(target_dict_unroll)
            self.a_full_ = lifter.get_vec(self.A_sparse_, sparse=True)
        return self
