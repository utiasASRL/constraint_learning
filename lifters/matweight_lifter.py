import numpy as np
import spatialmath.base as sm
from mwcerts.stereo_problems import LocalizationProblem, SLAMProblem

from lifters.state_lifter import StateLifter

# pose to pose noise parameters
P2P_STD_TRANS = 0.1
P2P_STD_ROT = 10 * np.pi / 180


class MatWeightLifter(StateLifter):
    HOM = "h"
    NOISE = 0.1

    @staticmethod
    def get_variable_indices(var_subset, variable=["xC", "xt", "xt0", "m"]):
        return np.unique(
            [
                int(key.split("_")[-1])
                for key in var_subset
                if any(key.startswith(f"{v}_") for v in variable)
            ]
        )

    ADMM_OPTIONS = dict(
        early_stop=False,
        maxiter=10,
        use_fusion=True,
        rho_start=1e2,
    )
    ADMM_INIT_XHAT = False

    def __init__(self, prob: SLAMProblem = None, **kwargs):
        self.Q_poly = None
        self.prob = prob
        self.trans_frame_ = self.prob.trans_frame
        self.dim = np.sum(list(self.prob.var_list.values())) - 1

        self.n_landmarks = prob.Nm
        self.n_poses = prob.Np
        self.n_cliques = self.n_poses - 1
        super().__init__(d=3, **kwargs)

    @property
    def trans_frame(self):
        return self.trans_frame_

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = self.prob.var_list
        return self.var_dict_

    def get_A_known(self, var_dict=None, output_poly=False, add_redundant=False):
        if var_dict is None:
            var_dict = self.var_dict
        use_i = np.unique([v.split("_")[1] for v in var_dict if "x" in v])
        use_nodes = [f"x_{i}" for i in use_i]
        constraints = self.prob.generate_constraints(use_nodes=use_nodes)
        if add_redundant:
            constraints += self.prob.generate_redun_constraints(use_nodes=use_nodes)

        if output_poly:
            # exclude A0 because it is treated differently by us.
            return [c.A for c in constraints if c.label != "Homog"]
        else:
            return [c.A.get_matrix(var_dict) for c in constraints if c.label != "Homog"]

    def test_constraints(self, *args, **kwargs):
        bad_j = []
        max_violation = 0
        for j, c in enumerate(self.prob.constraints + self.prob.constraints_r):
            if c.label == "Homog":
                continue
            x = self.get_x()
            try:
                A = c.A.get_matrix(self.var_dict)
                error = abs(x.T @ A @ x)
                max_violation = max(max_violation, error)
                bad_j.append(j)
                assert error <= 1e-10
            except AssertionError:
                print("Constraint didn't pass:", c.label)
        return max_violation, bad_j

    def get_gt_theta(
        self,
    ):
        theta = {}
        for v in self.prob.G.Vp.values():
            if v.label == "world":
                continue
            if self.prob.trans_frame == "world":
                theta[f"xt0_{v.index}"] = v.r_in0.flatten()
            else:
                theta[f"xt_{v.index}"] = v.C_p0 @ v.r_in0.flatten()
            theta[f"xC_{v.index}"] = v.C_p0.flatten("F")
        for v in self.prob.G.Vm.values():
            theta[v.label] = v.r_in0.flatten()
        return theta

    def sample_theta(self):
        theta = {}
        # poses
        for v in self.prob.G.Vp.values():
            if v.label == "world":
                continue
            t = 10 * np.random.rand(3)
            if self.prob.trans_frame == "world":
                theta[f"xt0_{v.index}"] = t
            else:
                theta[f"xt_{v.index}"] = t

            vec = np.random.rand(3, 1)
            vec /= np.linalg.norm(vec)
            rot = sm.angvec2r(np.random.rand(), vec)
            theta[f"xC_{v.index}"] = rot.flatten(order="F")

        # landmarks
        for v in self.prob.G.Vm.values():
            theta[v.label] = np.random.rand(3)
        return theta

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.get_gt_theta()
        return self.theta_

    def get_x(self, theta=None, parameters=None, var_subset=None):
        if (parameters is not None) and (not parameters == [1.0]):
            raise ValueError("we don't support parameters yet.")
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.prob.var_list

        # Construct x-vector using specified variables only.
        vec = []
        for var in var_subset:
            if var == "h":  # Homogenous
                vec += [1]
            elif "z" in var:  # subsitutions
                _, map_i, pose_i = var.split("_")
                C_i0 = theta[f"xC_{pose_i}"].reshape((3, 3), order="F")
                t_k0_0 = theta[f"m_{map_i}"]
                if self.prob.trans_frame == "world":
                    t_i0_i = C_i0 @ theta[f"xt0_{pose_i}"]
                elif self.prob.trans_frame == "local":
                    t_i0_i = theta[f"xt_{pose_i}"]
                t_ki_i = C_i0 @ t_k0_0 - t_i0_i
                vec += [t_ki_i.flatten()]
            else:  # Other variables
                try:
                    vec += [theta[var].flatten("F")]
                except KeyError:
                    T = theta[var.replace("xC", "x").replace("xt", "x")]
                    if "xC" in var:
                        vec += [T.C_ba().flatten("F")]
                    elif "xt" in var:
                        vec += [T.r_ab_inb().flatten("F")]

        return np.hstack(vec)

    def get_dim_x(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        return sum(size for v, size in self.var_dict.items() if v in var_subset)

    def get_vec_around_gt(self, delta=0):
        return self.prob.init_gauss_newton(sigma=delta)

    def local_solver(self, t0=None, y=None, verbose=False, **kwargs):
        if y is not None:
            # TODO(FD) not currently supported to use other than self.y_
            assert y == self.y_
        xhat, info = self.prob.gauss_newton(x_init=t0, verbose=verbose)
        if info["term_crit"] == "ITER":
            print("Warning: GN reached maximum number of iterations!")
        return xhat, info, info["cost"]

    def get_cost(self, theta, y):
        Q = self.prob.generate_cost(edges=y)
        x = self.get_x(theta=theta)
        return x.T @ Q.get_matrix(self.var_dict) @ x

    def get_Q(
        self, noise: float = None, output_poly=False, use_cliques=None, sparsity=1.0
    ):
        if noise is None:
            noise = self.NOISE
        if use_cliques is None:
            use_cliques = [f"x_{i}" for i in np.arange(self.n_poses)]

        if self.y_ is None:
            self.simulate_y(noise=noise, sparsity=sparsity)

        Q = self.prob.generate_cost(edges=self.y_, use_nodes=use_cliques)

        # make sure that all elements in cost matrix are actually in var_dict!
        assert set(Q.get_variables()).issubset(self.var_dict.keys())

        if output_poly:
            return Q, self.prob.G, self.y_
        else:
            # using a lsit makes sure an error is thrown when a key is not available.
            return Q.get_matrix(self.var_dict), self.y_

    def get_error(self, theta_hat):
        error_dict = {"error_trans": 0, "error_rot": 0, "error": 0}
        for key, val in self.theta.items():
            if "xt" in key:  # translation errors
                err = np.linalg.norm(val - theta_hat[key])
                error_dict["error_trans"] += err
                error_dict["error"] += err
            elif "xC" in key:  # translation errors
                err = np.linalg.norm(val - theta_hat[key])
                error_dict["error_rot"] += err
                error_dict["error"] += err
        return error_dict

    # clique stuff
    def base_size(self):
        return self.var_dict["h"]

    def node_size(self):
        return self.var_dict["xt_0"] + self.var_dict[f"xC_0"]

    def get_clique_vars_ij(self, *args):
        var_dict = {
            "h": self.var_dict["h"],
        }
        for i in args:
            var_dict.update(
                {
                    f"xC_{i}": self.var_dict[f"xC_{i}"],
                    f"xt_{i}": self.var_dict[f"xt_{i}"],
                }
            )
        return var_dict

    def get_clique_vars(self, i, n_overlap=0):
        used_landmarks = list(range(i, min(i + n_overlap + 1, self.n_poses)))
        vars = {
            "h": self.var_dict["h"],
        }
        for j in used_landmarks:
            vars.update(
                {
                    f"xC_{j}": self.var_dict[f"xC_{j}"],
                    f"xt_{j}": self.var_dict[f"xt_{j}"],
                }
            )
        return vars

    def get_clique_cost(self, i):
        return self.prob.generate_cost(
            use_nodes=[f"x_{i}", f"x_{i+1}"],
            overlaps={f"x_{i}": 0.5 for i in range(1, self.n_cliques)},
        )


class MatWeightSLAMLifter(MatWeightLifter):
    VARIABLE_LIST = [
        ["h", "xt_0", "xC_0"],
        ["h", "xt_0", "xC_0", "xt_1", "xC_1"],
        ["h", "xt_0", "xC_0", "m_0"],
        ["h", "xt_0", "xC_0", "m_0", "xt_1", "xC_1", "m_1"],
    ]

    def __init__(self, n_landmarks=10, n_poses=5, trans_frame="local", **kwargs):
        prob = SLAMProblem.create_structured_problem(
            Nm=n_landmarks, Np=n_poses, trans_frame=trans_frame
        )
        super().__init__(prob=prob, **kwargs)

    def simulate_y(self, noise, sparsity=1.0):
        edges_p2m = self.prob.G.gen_map_edges(alpha=sparsity)
        # from mwcerts.stereo_problems import Camera
        # c = Camera.get_realistic_model()
        # self.prob.stereo_meas_model(edges_p2m, c=c)
        self.prob.gauss_isotrp_meas_model(edges_p2m, sigma=noise)

        edges_p2p = self.prob.G.gen_pg_edges(pg_type="chain")
        if noise > 0:
            self.prob.add_p2p_meas(
                edges_p2p, p2p_std_rot=P2P_STD_ROT, p2p_std_trans=P2P_STD_TRANS
            )
        else:
            # for testing only
            self.prob.add_p2p_meas(edges_p2p, p2p_std_rot=0, p2p_std_trans=0)

        # fix the first pose to origin.
        if noise > 0:
            self.prob.add_init_pose_prior()
        self.y_ = self.prob.G.E

    def __repr__(self):
        return "mw_slam_lifter"


class MatWeightLocLifter(MatWeightLifter):
    VARIABLE_LIST = [
        ["h", "xt_0", "xC_0"],
        ["h", "xt_0", "xC_0", "xt_1", "xC_1"],
    ]
    ALL_PAIRS = False
    CLIQUE_SIZE = 2

    def __init__(self, n_landmarks=10, n_poses=5, trans_frame="local", **kwargs):
        prob = LocalizationProblem.create_structured_problem(
            Nm=n_landmarks, Np=n_poses, trans_frame=trans_frame
        )
        # prob = LocalizationProblem.create_test_problem(
        #    Nm=5, which="lookup-fixed", trans_frame=trans_frame
        # )
        super().__init__(prob=prob, **kwargs)

    def get_all_variables(self):
        # label = "t" if self.trans_frame == "local" else "t0"
        # variables = ["h"]
        # for i in range(self.n_poses):
        #    variables += [f"xC_{i}", f"x{label}_{i}"]
        return [list(self.prob.var_list.keys())]

    def simulate_y(self, noise, sparsity=1.0):
        from mwcerts.stereo_problems import Camera

        edges_p2m = self.prob.G.gen_map_edges(alpha=sparsity)
        c = Camera.get_realistic_model(sigma_u=noise, sigma_v=noise)
        self.prob.stereo_meas_model(edges_p2m, c=c)
        # self.prob.gauss_isotrp_meas_model(edges_p2m, sigma=noise)

        edges_p2p = self.prob.G.gen_pg_edges(pg_type="chain")
        if noise > 0:
            self.prob.add_p2p_meas(
                edges_p2p, p2p_std_trans=P2P_STD_TRANS, p2p_std_rot=P2P_STD_ROT
            )
        else:
            # for testing only
            self.prob.add_p2p_meas(edges_p2p, p2p_std_trans=0.0, p2p_std_rot=0.0)
        self.y_ = self.prob.G.E

    def __repr__(self):
        return "mw_loc_lifter"
