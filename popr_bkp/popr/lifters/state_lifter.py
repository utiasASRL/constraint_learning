import itertools
from abc import abstractmethod

import numpy as np
from poly_matrix import PolyMatrix
from popr.utils.common import get_vec
from popr.utils.constraint import Constraint, remove_dependent_constraints

from ._base_class import BaseClass


class StateLifter(BaseClass):
    HOM = "h"

    # tolerance for feasibility error of learned constraints
    EPS_ERROR = 1e-8

    LEVELS = ["no"]
    PARAM_LEVELS = ["no", "p", "ppT"]
    VARIABLE_LIST = ["h"]

    # properties of template scaling
    ALL_PAIRS = True
    # Below only have effect if ALL_PAIRS is False.
    # Then, they determine the clique size hierarchy.
    CLIQUE_SIZE = 5
    STEP_SIZE = 1

    TIGHTNESS = "cost"

    def __init__(
        self,
        level="no",
        param_level="no",
        d=2,
        variable_list=None,
        robust=False,
        n_outliers=0,
    ):
        self.robust = robust
        self.n_outliers = n_outliers

        assert level in self.LEVELS
        self.level = level

        assert param_level in self.PARAM_LEVELS
        self.param_level = param_level

        if variable_list is not None:
            self.variable_list = variable_list
        else:
            self.variable_list = self.VARIABLE_LIST

        # variables that get overwritten upon initialization
        self.parameters_ = None
        self.theta_ = None
        self.var_dict_ = None
        self.y_ = None

        self.d = d
        self.generate_random_setup()
        super().__init__()

    def get_involved_param_dict(self, var_subset):
        keys = [self.HOM]
        for v in var_subset:
            index = v.split("_")
            if len(index) > 1:
                index = int(index[-1])
                key = f"p_{index}"
                if key not in keys:
                    keys.append(key)
        return [k for k in keys if k in self.param_dict]

    def compute_Ai(self, templates, var_dict, param_dict):
        """
        Take all elements from the list of templates and apply them
        to the given pair of var_list and param_list.
        """

        A_list = []
        for template in templates:
            assert isinstance(template, Constraint)
            # First, we find the current parameters, so that we can factor
            # them into b and compute a from it.
            p_here = self.get_p(param_subset=param_dict)

            # We need to partition the vector b into its subblocks
            # so that we can compute a from it.
            X_dim = self.get_dim_X(template.mat_var_dict)
            assert self.get_dim_X(var_dict) == X_dim
            p_dim = self.get_dim_p(template.mat_param_dict)
            assert self.get_dim_p(param_dict) == p_dim
            n_blocks = len(template.b_) / X_dim
            assert n_blocks == p_dim

            a = p_here[0] * template.b_[:X_dim]
            for i in range(int(n_blocks) - 1):
                a += p_here[i + 1] * template.b_[(i + 1) * X_dim : (i + 2) * X_dim]

            a_test = self.get_reduced_a(
                template.b_,
                var_subset=template.mat_var_dict,
                parameters=p_here,
                sparse=False,
            )
            np.testing.assert_allclose(a, a_test)

            # Get a symmetric matrix where the upper and lower parts have been filled with a,
            # and applying the correction to the diagonal.
            # Note that we do not set var_dict because otherwise A would already
            # be the zero-padded large matrix.
            A = self.get_mat(a, sparse=True, correct=True)

            # Get the corresponding PolyMatrix.
            A_poly, __ = PolyMatrix.init_from_sparse(
                A,
                var_dict=self.get_var_dict(var_dict),
                symmetric=True,
                unfold=False,
            )
            A_list.append(A_poly)

        return A_list

    def apply_template(self, bi_poly, n_parameters=None, verbose=False):
        if n_parameters is None:
            n_parameters = len(self.parameters)

        new_poly_rows = []
        # find the number of variables that this constraint touches.
        unique_idx = set()
        for key in bi_poly.variable_dict_j:
            param, var_keys = key.split("-")
            vars = var_keys.split(".")
            vars += param.split(".")
            for var in vars:
                var_base = var.split(":")[0]
                if "_" in var_base:
                    i = int(var_base.split("_")[-1])
                    unique_idx.add(i)

        if len(unique_idx) == 0:
            return [bi_poly]
        elif len(unique_idx) > 3:
            raise ValueError("unexpected triple dependencies!")

        variable_indices = self.get_variable_indices(self.var_dict)
        # if z_0 is in this constraint, repeat the constraint for each landmark.
        for idx in itertools.combinations(variable_indices, len(unique_idx)):
            new_poly_row = PolyMatrix(symmetric=False)
            for key in bi_poly.variable_dict_j:
                # need intermediate variables cause otherwise z_0 -> z_1 -> z_2 etc. can happen.
                key_ij = key
                for from_, to_ in zip(unique_idx, idx):
                    key_ij = key_ij.replace(f"x_{from_}", f"xi_{to_}")
                    key_ij = key_ij.replace(f"w_{from_}", f"wi_{to_}")
                    key_ij = key_ij.replace(f"z_{from_}", f"zi_{to_}")
                    key_ij = key_ij.replace(f"p_{from_}", f"pi_{to_}")
                key_ij = (
                    key_ij.replace("zi", "z")
                    .replace("pi", "p")
                    .replace("xi", "x")
                    .replace("wi", "w")
                )
                if verbose and (key != key_ij):
                    print("changed", key, "to", key_ij)

                try:
                    params = key_ij.split("-")[0]
                    pi, pj = params.split(".")
                    pi, di = pi.split(":")
                    pj, dj = pj.split(":")
                    if pi == pj:
                        if not (int(dj) >= int(di)):
                            raise IndexError(
                                "something went wrong in augment_basis_list"
                            )
                except ValueError as e:
                    pass
                new_poly_row[self.HOM, key_ij] = bi_poly[self.HOM, key]
            new_poly_rows.append(new_poly_row)
        return new_poly_rows

    def apply_templates(
        self, templates, starting_index=0, var_dict=None, all_pairs=None
    ):

        if all_pairs is None:
            all_pairs = self.ALL_PAIRS
        if var_dict is None:
            var_dict = self.var_dict

        new_constraints = []
        index = starting_index
        for template in templates:
            constraints = self.apply_template(template.polyrow_b_)
            template.applied_list = []
            for new_constraint in constraints:
                template.applied_list.append(
                    Constraint.init_from_polyrow_b(
                        index=index,
                        polyrow_b=new_constraint,
                        lifter=self,
                        template_idx=template.index,
                        known=template.known,
                        mat_var_dict=var_dict,
                    )
                )
                new_constraints += template.applied_list
                index += 1

        if len(new_constraints):
            remove_dependent_constraints(new_constraints)
        return new_constraints

    def get_vec_around_gt(self, delta: float = 0):
        """Sample around ground truth.
        :param delta: sample from gt + std(delta) (set to 0 to start from gt.)
        """
        return self.theta + np.random.normal(size=self.theta.shape, scale=delta)

    def test_constraints(self, A_list, errors: str = "raise", n_seeds: int = 3):
        """
        :param A_list: can be either list of sparse matrices, or poly matrices
        :param errors: "raise" or "print" detected violations.
        """
        max_violation = -np.inf
        j_bad = set()

        for j, A in enumerate(A_list):
            if isinstance(A, PolyMatrix):
                A = A.get_matrix(self.get_var_dict(unroll_keys=True))

            for i in range(n_seeds):
                if i == 0:
                    x = self.get_x()
                else:
                    np.random.seed(i)
                    t = self.sample_theta()
                    p = self.sample_parameters()
                    x = self.get_x(theta=t, parameters=p)

                constraint_violation = abs(x.T @ A @ x)
                max_violation = max(max_violation, constraint_violation)
                if constraint_violation > self.EPS_ERROR:
                    msg = f"big violation at {j}: {constraint_violation:.1e}"
                    j_bad.add(j)
                    if errors == "raise":
                        raise ValueError(msg)
                    elif errors == "print":
                        print(msg)
                    elif errors == "ignore":
                        pass
                    else:
                        raise ValueError(errors)
        return max_violation, j_bad

    def get_A0(self, var_subset=None):
        if var_subset is not None:
            var_dict = {k: self.var_dict[k] for k in var_subset}
        else:
            var_dict = self.var_dict
        A0 = PolyMatrix()
        A0[self.HOM, self.HOM] = 1.0
        return A0.get_matrix(var_dict)

    def get_A_b_list(self, A_list, var_subset=None):
        return [(self.get_A0(var_subset), 1.0)] + [(A, 0.0) for A in A_list]

    def get_A_known(self, var_dict=None, output_poly: bool = False) -> list:
        return []

    def get_B_known(self) -> list:
        return []

    def sample_parameters(self, theta=None) -> np.ndarray:
        assert (
            self.param_level == "no"
        ), "Need to overwrite sample_parameters to use level different than 'no'"
        return {self.HOM: 1.0}

    def sample_parameters_landmarks(self, landmarks):
        """Used by RobustPoseLifter, RangeOnlyLocLifter: the default way of adding landmarks to parameters."""
        if self.landmarks is None:
            self.landmarks = landmarks
        parameters = {self.HOM: 1.0}

        if self.param_level == "no":
            return parameters

        for i in range(self.n_landmarks):
            if self.param_level == "p":
                parameters[f"p_{i}"] = landmarks[i]
            elif self.param_level == "ppT":
                parameters[f"p_{i}"] = np.hstack(
                    [
                        landmarks[i],
                        get_vec(np.outer(landmarks[i], landmarks[i]), correct=False),
                    ]
                )
        return parameters

    @abstractmethod
    def sample_theta(self) -> np.ndarray:
        raise NotImplementedError("need to implement sample_theta")

    def get_grad(self, t, y) -> np.ndarray:
        raise NotImplementedError("get_grad not implement yet")

    def get_cost(self, theta, y) -> float:
        x = self.get_x(theta=theta)
        Q, y = self.get_Q()
        return x.T @ Q @ x

    def get_error(self, t) -> dict:
        err = np.linalg.norm(t - self.theta) ** 2 / self.theta.size
        return {"MSE": err, "error": err}

    def get_level_dims(self, n=1):
        assert (
            self.level == "no"
        ), "Need to overwrite get_level_dims to use level different than 'no'"
        return {"no": 0}

    def get_Q(self, output_poly=False):
        raise NotImplementedError(
            "Need to impelement get_Q in inheriting class if you want to use it."
        )
        return None, None

    def local_solver(self, t0, y=None, verbose=False):
        raise NotImplementedError(
            "Need to implement local_solver in inheriting class if you want to use it."
        )

    def set_noise(self, noise):
        self.noise = noise

    def generate_random_setup(self):
        if self.parameters is None:
            self.parameters = self.sample_parameters()
        if self.theta is None:
            self.theta = self.sample_theta()

    @property
    def var_dict(self):
        raise ValueError("Inheriting class must implement this!")

    @property
    def param_dict(self):
        assert (
            self.param_level == "no"
        ), "Need to overwrite param_dict to use level different than 'no'"
        return {self.HOM: 1}

    @property
    def param_dict_landmarks(self):
        param_dict = {self.HOM: 1}
        if self.param_level == "no":
            return param_dict
        if self.param_level == "p":
            param_dict.update({f"p_{i}": self.d for i in range(self.n_landmarks)})
        if self.param_level == "ppT":
            # Note that ppT is actually
            # [p; vech(ppT)] (linear and quadratic terms)
            param_dict.update(
                {
                    f"p_{i}": self.d + int(self.d * (self.d + 1) / 2)
                    for i in range(self.n_landmarks)
                }
            )
        return param_dict

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        if theta is None:
            theta = self.theta
        if parameters is None:
            parameters = self.parameters
        if var_subset is None:
            var_subset = self.var_dict

        x_data = []
        for key in var_subset:
            if key == self.HOM:
                x_data.append(1.0)
            else:
                print(
                    "Warning: just using theta in x because there is no specific implementation."
                )
                x_data += list(theta)
        return np.array(x_data)

    def get_p(self, parameters: dict = None, param_subset: dict | list = None):
        if parameters is None:
            parameters = self.parameters
        if param_subset is None:
            param_subset = self.param_dict

        p_data = []
        for key in param_subset:
            if key == self.HOM:
                p_data.append(1.0)
            else:
                param = parameters[key]
                if np.ndim(param) == 0:
                    p_data.append(param)
                else:
                    p_data += list(param)
        return np.array(p_data)
