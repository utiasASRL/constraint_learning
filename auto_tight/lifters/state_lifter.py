import itertools

import numpy as np
from poly_matrix import PolyMatrix

from ._base_class import BaseClass


class StateLifter(BaseClass):
    HOM = "h"

    # set elements below this threshold to zero.
    EPS_SPARSE = 1e-9

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
        elif len(unique_idx) > 2:
            raise ValueError("unexpected triple dependencies!")

        raise ValueError(
            "this is where the mistake happens! should return all but is only returning [0]"
        )
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
                new_poly_row[self.HOM, key_ij] = bi_poly["h", key]
            new_poly_rows.append(new_poly_row)
        return new_poly_rows

    def apply_templates(
        self, templates, starting_index=0, var_dict=None, all_pairs=None
    ):
        from utils.constraint import Constraint, remove_dependent_constraints

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
                A = A.get_matrix(self.var_dict_unroll)

            for i in range(n_seeds):
                np.random.seed(i)
                t = self.sample_theta()
                p = self.get_parameters()
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
        if self.param_level == "no":
            return np.ndarray([1.0])

    def sample_theta(self) -> np.ndarray:
        raise NotImplementedError("need to implement sample_theta")

    def generate_random_setup(self):
        self.theta = self.sample_theta()
        self.parameters = self.sample_parameters()

    def get_parameters(self, var_subset=None) -> list:
        if var_subset is not None:
            raise ValueError("var_subset not supported for default get_parameters.")
        if self.param_level == "no":
            return [1.0]

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

    @property
    def base_var_dict(self):
        var_dict = {"x": self.d**2 + self.d}
        return var_dict

    @property
    def sub_var_dict(self):
        level_dim = self.get_level_dims()[self.level]
        var_dict = {f"z_{k}": self.d + level_dim for k in range(self.n_parameters)}
        return var_dict

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = {self.HOM: 1}
            self.var_dict_.update(self.base_var_dict)
            self.var_dict_.update(self.sub_var_dict)
        return self.var_dict_
