import numpy as np
import spatialmath.base as sm

from constraint_learning.lifters.state_lifter import StateLifter
from mwcerts.stereo_problems import LocalizationProblem, SLAMProblem


class MatWeightLifter(StateLifter):
    HOM = "w_0"
    NOISE = 0.01

    def __init__(self, prob: SLAMProblem = None, **kwargs):
        self.Q_poly = None
        self.prob = prob
        self.trans_frame_ = self.prob.trans_frame
        self.dim = np.sum(list(self.prob.var_list.values())) - 1

        self.n_landmarks = prob.Nm
        self.n_poses = prob.Np
        super().__init__(d=3, **kwargs)

    @property
    def trans_frame(self):
        return self.trans_frame_

    @property
    def var_dict(self):
        if self.var_dict_ is None:
            self.var_dict_ = self.prob.var_list
        return self.var_dict_

    def generate_random_setup(self):
        pass

    def get_A_known(self):
        # self.prob.generate_constraints()
        # self.prob.generate_redun_constraints()
        # exclude A0 because it is treated differently by us.
        return [
            c.A.get_matrix(self.var_dict)
            for c in self.prob.constraints + self.prob.constraints_r
            if not c.label == "Homog"
        ]

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
                theta[v.label + "_t0"] = v.r_in0.flatten()
            else:
                theta[v.label + "_t"] = v.C_p0 @ v.r_in0.flatten()
            theta[v.label + "_C"] = v.C_p0.flatten("F")
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
                theta[v.label + "_t0"] = t
            else:
                theta[v.label + "_t"] = t

            vec = np.random.rand(3, 1)
            vec /= np.linalg.norm(vec)
            rot = sm.angvec2r(np.random.rand(), vec)
            theta[v.label + "_C"] = rot.flatten(order="F")

        # landmarks
        for v in self.prob.G.Vm.values():
            theta[v.label] = np.random.rand(3)
        return theta

    @property
    def theta(self):
        if self.theta_ is None:
            self.theta_ = self.get_gt_theta()
        return self.theta_

    def get_x(self, parameters=None, theta=None, var_subset=None):
        if theta is None:
            theta = self.theta
        if var_subset is None:
            var_subset = self.prob.var_list

        # Construct x-vector using specified variables only.
        vec = []
        for var in var_subset:
            if "w" in var:  # Homogenous
                vec += [1]
            elif "m" in var and "x" in var:  # subsitutions
                map_lbl = var.split("_")[0]
                pose_lbl = var.split("_")[1]
                C_i0 = theta[pose_lbl + "_C"].reshape((3, 3), order="F")
                t_k0_0 = theta[map_lbl]
                if self.prob.trans_frame == "world":
                    t_i0_i = C_i0 @ theta[pose_lbl + "_t0"]
                elif self.prob.trans_frame == "local":
                    t_i0_i = theta[pose_lbl + "_t"]
                t_ki_i = C_i0 @ t_k0_0 - t_i0_i
                vec += [t_ki_i]
            else:  # Other variables
                vec += [theta[var]]
        return np.hstack(vec)

    def get_dim_x(self, var_subset=None):
        if var_subset is None:
            var_subset = self.var_dict
        return sum(size for v, size in self.var_dict.items() if v in var_subset)

    def get_vec_around_gt(self, delta=0):
        return self.prob.init_gauss_newton(sigma=delta)

    def local_solver(self, t0=None, verbose=False, **kwargs):
        xhat, info = self.prob.gauss_newton(x_init=t0, verbose=verbose)
        return xhat, info, info["cost"]

    def get_Q(self, noise: float = None, output_poly=False):
        if noise is None:
            noise = self.NOISE

        if self.Q_poly is None:
            self.fill_graph(noise=noise)
            self.prob.generate_cost()
            self.Q_poly = self.prob.Q

        # make sure that all elements in cost matrix are actually in var_dict!
        assert set(self.prob.Q.get_variables()).issubset(self.var_dict.keys())

        if output_poly:
            return self.Q_poly, None
        else:
            # using a lsit makes sure an error is thrown when a key is not available.
            return self.Q_poly.get_matrix(self.var_dict), None

    def get_error(self, theta_hat):
        error_dict = {"error_trans": 0, "error_rot": 0}
        for key, val in self.theta.items():
            if "_t" in key:  # translation errors
                error_dict["error_trans"] = np.linalg.norm(val - theta_hat[key])
            elif "_C" in key:  # translation errors
                error_dict["error_rot"] = np.linalg.norm(val - theta_hat[key])
        return error_dict


class MatWeightSLAMLifter(MatWeightLifter):
    def __init__(self, n_landmarks=10, n_poses=5, trans_frame="local", **kwargs):
        prob = SLAMProblem.create_structured_problem(
            Nm=n_landmarks, Np=n_poses, trans_frame=trans_frame
        )
        super().__init__(prob=prob, **kwargs)

    def fill_graph(self, noise):
        edges_p2m = self.prob.G.gen_map_edges_full()
        # from mwcerts.stereo_problems import Camera
        # c = Camera.get_realistic_model()
        # self.prob.stereo_meas_model(edges_p2m, c=c)
        self.prob.gauss_isotrp_meas_model(edges_p2m, sigma=noise)

        edges_p2p = self.prob.G.gen_pg_edges(pg_type="chain")
        self.prob.add_p2p_meas(edges_p2p, p2p_std_rot=noise, p2p_std_trans=noise)

        # fix the first pose to origin.
        if noise > 0:
            self.prob.add_init_pose_prior(v2=self.prob.G.Vp["x0"])

    def __repr__(self):
        return "mw_slam_lifter"


class MatWeightLocLifter(MatWeightLifter):
    VARIABLE_LIST = [
        ["w_0", "x0_t"],
        ["w_0", "x0_C"],
        ["w_0", "x0_t", "x1_t"],
        ["w_0", "x0_t", "x1_C"],
        ["w_0", "x0_t", "x0_C", "x1_t"],
        ["w_0", "x0_t", "x0_C", "x1_C"],
        ["w_0", "x0_t", "x0_C", "x1_t", "x1_C"],
    ]

    def __init__(self, n_landmarks=10, n_poses=5, trans_frame="local", **kwargs):
        prob = LocalizationProblem.create_structured_problem(
            Nm=n_landmarks, Np=n_poses, trans_frame=trans_frame
        )
        # prob = LocalizationProblem.create_test_problem(
        #    Nm=5, which="lookup-fixed", trans_frame=trans_frame
        # )
        super().__init__(prob=prob, **kwargs)

    def fill_graph(self, noise):
        # from mwcerts.stereo_problems import Camera

        edges_p2m = self.prob.G.gen_map_edges_full()
        # c = Camera.get_realistic_model()
        # self.prob.stereo_meas_model(edges_p2m, c=c)
        self.prob.gauss_isotrp_meas_model(edges_p2m, sigma=noise)

        # edges_p2p = self.prob.G.gen_pg_edges(pg_type="chain")
        # self.prob.add_p2p_meas(edges_p2p, p2p_std_trans=noise, p2p_std_rot=noise)

    def __repr__(self):
        return "mw_loc_lifter"
