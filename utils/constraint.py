from lifters.state_lifter import StateLifter
from poly_matrix.poly_matrix import PolyMatrix

import numpy as np


class Constraint(object):
    """
    This class serves the main purpose of not recomputing representations of constraints more than once.
    """
    def __init__(
        self,
        index=0,
        value=0,
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
        self.value = value
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
            A_poly = lifter.convert_b_to_Apoly(b, mat_var_dict)
            polyrow_b = lifter.convert_b_to_polyrow(b, mat_var_dict)
            polyrow_a = lifter.convert_a_to_polyrow(a, mat_var_dict)
            return Constraint(
                index=index,
                a=a,
                b=b,
                A_sparse=A_sparse,
                A_poly=A_poly,
                polyrow_b=polyrow_b,
                polyrow_a=polyrow_a,
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
    def init_from_polyrow_b(
        index: int,
        polyrow_b: PolyMatrix,
        lifter: StateLifter,
        known: bool = False,
        template_idx: int = None,
    ):
        A_poly = lifter.convert_polyrow_to_Apoly(polyrow_b)
        A_sparse = A_poly.get_matrix(lifter.var_dict_unroll)
        a_full = lifter.get_vec(A_sparse, sparse=True)
        return Constraint(
            index=index,
            A_poly=A_poly,
            polyrow_b=polyrow_b,
            A_sparse=A_sparse,
            a_full=a_full,
            known=known,
            template_idx=template_idx,
        )

    def scale_to_new_lifter(self, lifter: StateLifter):
        target_dict_unroll = lifter.var_dict_unroll
        self.A_sparse_ = self.A_poly_.get_matrix(target_dict_unroll)
        self.a_full_ = lifter.get_vec(self.A_sparse_, sparse=True)
        return self