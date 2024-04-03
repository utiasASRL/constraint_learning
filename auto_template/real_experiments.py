import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pylgmath.so3.operations import hat

from auto_template.learner import Learner
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from starloc.reader import read_calib, read_data, read_landmarks
from utils.geometry import get_theta_from_C_r
from utils.plotting_tools import plot_frame, savefig

REJECT_OUTLIERS = False
OUTLIER_THRESHOLD = 0.5

# how many single pose estimates to plot
PLOT_NUMBER = 0  # 10

RESULTS_DIR = "_results_new"

# below are parameters for RO only
RANGE_TYPE = "range_calib"
DEGENERATE_DICT = {
    0: [4, 9, 11, 12],  # top (11, 12 high)
    1: [4, 9, 6, 7],  # inclined (6, 7 low)
    2: [4, 5, 6, 12],  # diagonal (6, 12 vertical)
    3: [9, 5, 7, 11],  # diagonal (7, 11 vertical)
    4: [4, 6, 7, 12],  # diagonal
}
# how to loop through subsets of anchors (random or round-robin or all), RO only
ANCHOR_CHOICE = "all"
EXCLUDE_ANCHORS = [6, 7]

DATASET_ROOT = str(Path(__file__).parent.parent)


def load_experiment(dataset):
    if dataset == "starrynight":
        exp = Experiment(
            dataset_root=DATASET_ROOT, dataset="starrynight", data_type="stereo"
        )
    else:
        data_type = "apriltag_cal_individual"
        exp = Experiment(
            dataset_root=DATASET_ROOT, dataset=dataset, data_type=data_type
        )
    return exp


class Experiment(object):
    def __init__(self, dataset_root, dataset, data_type, from_id=None):
        assert data_type in ["apriltag_cal_individual", "stereo", "uwb"]
        self.dataset = dataset
        self.data_type = data_type

        # an additional transform from measurements to ground truth frame.
        # only used for starrynight dataset currently
        self.T_cv = None
        self.T_vc = None

        if dataset == "starrynight":
            from scipy.io import loadmat

            matfile = os.path.join(dataset_root, "starrynight", "dataset3.mat")
            self.data = loadmat(matfile)

            self.all_landmarks = self.data["rho_i_pj_i"].T  # 20 x 3, all landmarks

            C_cv = self.data["C_c_v"]
            rho_v_cv = self.data["rho_v_c_v"]
            self.T_cv = np.vstack(
                [
                    np.hstack([C_cv, -C_cv @ rho_v_cv]),  # C_cv, rho_c_vc
                    [0, 0, 0, 1],
                ]
            )
            self.T_vc = np.vstack(
                [
                    np.hstack([C_cv.T, rho_v_cv]),  # C_vc, rho_v_cv
                    [0, 0, 0, 1],
                ]
            )

            # fmt: off
            self.M_matrix = np.r_[
                np.c_[self.data["fu"], 0, self.data["cu"], self.data["fu"] * self.data["b"] / 2, ],
                np.c_[0, self.data["fv"], self.data["cv"], 0],
                np.c_[self.data["fu"], 0, self.data["cu"], -self.data["fu"] * self.data["b"] / 2, ],
                np.c_[0, self.data["fv"], self.data["cv"], 0],
            ]
            # fmt: on
        else:
            dataset_root = os.path.join(dataset_root, "starloc")
            self.all_landmarks = read_landmarks(
                dataset_root=dataset_root, dataset=dataset, data=[data_type]
            )
            self.data = read_data(
                dataset_root=dataset_root, dataset=dataset, data=[data_type]
            )
            if from_id is not None:
                self.data = self.data.loc[self.data.from_id == from_id]
            self.landmark_ids = list(self.all_landmarks.id.values)

            if len(EXCLUDE_ANCHORS) and data_type == "uwb":
                self.all_landmarks = self.all_landmarks[
                    ~self.all_landmarks.id.isin(EXCLUDE_ANCHORS)
                ]
                for a in EXCLUDE_ANCHORS:
                    self.landmark_ids.remove(a)
                self.data = self.data[~self.data.to_id.isin(EXCLUDE_ANCHORS)]

            if "apriltag" in data_type:
                calib_dict = read_calib(dataset_root=dataset_root, dataset=dataset)
                # fmt: off
                self.M_matrix = np.r_[
                    np.c_[calib_dict["fu"], 0, calib_dict["cu"], 0],
                    np.c_[0, calib_dict["fv"], calib_dict["cv"], 0],
                    np.c_[calib_dict["fu"], 0, calib_dict["cu"], -calib_dict["fu"] * calib_dict["b"]],
                    np.c_[0, calib_dict["fv"], calib_dict["cv"], 0],
                ]
                # fmt: on

    def set_params(self, min_n_landmarks, max_n_landmarks, sim_noise, use_gt, level):
        self.params = dict(
            min_n_landmarks=min_n_landmarks,
            max_n_landmarks=max_n_landmarks,
            sim_noise=sim_noise,
            use_gt=use_gt,
            level=level,
        )

    def get_range_measurements(
        self,
        time_idx=0,
        n_positions=None,
        combine_measurements=True,
        chosen_idx=-1,
    ):
        if "position_idx" not in self.data.columns:
            if combine_measurements:
                ref_id = self.landmark_ids[0]
                times = self.data.loc[self.data.to_id == ref_id]
                self.data["position_idx"] = np.cumsum(self.data.to_id.values == ref_id)
            else:
                times = self.data.time_s.unique()
                self.data["position_idx"] = range(len(times))

            # find the positions at which we measure at least min_n_landmarks.
            valid_position_idx = self.data.position_idx.value_counts().gt(
                self.params["min_n_landmarks"]
            )
            valid_position_idx = valid_position_idx[valid_position_idx].index
            self.data = self.data[self.data.position_idx.isin(valid_position_idx)]

            # rename position idx from 0 to num_positions
            self.data.position_idx = self.data.position_idx.replace(
                dict(
                    zip(
                        self.data.position_idx.unique(),
                        range(len(self.data.position_idx.unique())),
                    )
                )
            )

        if n_positions is None:
            n_positions = len(self.data.position_idx.unique())

        data_here = self.data[
            (self.data.position_idx >= time_idx)
            & (self.data.position_idx < time_idx + n_positions)
        ]
        if REJECT_OUTLIERS:
            mask = np.abs(data_here.bias_calib) < OUTLIER_THRESHOLD
            n_outliers = np.sum(~mask)
            if n_outliers > 0:
                print(f"removing {n_outliers} outliers")
            data_here = data_here[mask]

        assert len(data_here.position_idx.unique()) == n_positions

        if chosen_idx >= 0:
            unique_ids = DEGENERATE_DICT[chosen_idx]
        else:
            unique_ids = data_here.to_id.unique()
            if len(unique_ids) > self.params["max_n_landmarks"]:
                unique_ids = unique_ids[
                    np.random.choice(
                        range(len(unique_ids)),
                        self.params["max_n_landmarks"],
                        replace=False,
                    )
                ]

        landmarks = self.all_landmarks.loc[self.all_landmarks.id.isin(unique_ids)]

        self.landmarks = landmarks[["x", "y", "z"]].values
        self.landmark_ids = list(landmarks.id.values)
        print(f"using landmark ids: {self.landmark_ids}")
        n_landmarks = self.landmarks.shape[0]
        if n_landmarks != self.params["max_n_landmarks"]:
            print("Warning: less landmarks than requested.")
            return False

        # y_ is of shape n_positions * n_landmarks
        self.y_ = np.zeros((n_positions, n_landmarks))
        self.W_ = np.zeros((n_positions, n_landmarks))
        self.positions = np.empty((n_positions, 3))

        position_ids = list(data_here.position_idx.unique())
        for position_id, df_sub in data_here.groupby("position_idx"):
            position = df_sub[["x", "y", "z"]].mean()
            position_idx = position_ids.index(position_id)
            self.positions[position_idx, :] = position
            for __, row in df_sub.iterrows():
                if row.to_id not in self.landmark_ids:
                    continue
                landmark_idx = self.landmark_ids.index(row.to_id)

                self.W_[position_idx, landmark_idx] = 1.0
                self.y_[position_idx, landmark_idx] = row[RANGE_TYPE] ** 2
        return True

    def get_stereo_measurements(self, time_idx=0):
        """
        We use the following conventions:
        - r_c_0c is the vector from c to world expressed in c frame (we use "0" to denote world frame)
        - C_c0 is the rotation to change a vector expressed in world to the same vector expressed in c.

        With these conventions, we have:
        [r_ic_c]  =  [C_c0  r_0c_c] [r_i0_0]
        [  1   ]     [ 0     1    ] [  1   ]
                     -----T_cw-----
        """

        if self.dataset == "starrynight":
            from scipy.spatial.transform import Rotation

            n_meas = self.data["t"].shape[1]
            if time_idx >= n_meas:
                print(f"Skipping {time_idx}>={n_meas}")
                return None

            print(f"Choosing time idx {time_idx} from {n_meas}")

            # axis-angle representation of gt rotation
            psi_v0 = self.data["theta_vk_i"][:, [time_idx]]
            psi_mag = np.linalg.norm(psi_v0)

            # conversion to rotation matrix
            C_v0 = (
                np.cos(psi_mag) * np.eye(3)
                + (1 - np.cos(psi_mag)) * (psi_v0 / psi_mag) @ (psi_v0 / psi_mag).T
                - np.sin(psi_mag) * hat(psi_v0 / psi_mag)
            )
            r_v0_0 = self.data["r_i_vk_i"][:, time_idx]  # gt translation

            C_cv = self.data["C_c_v"]
            r_cv_v = self.data["rho_v_c_v"].flatten()

            C_c0 = C_cv @ C_v0
            r_0c_c = -C_c0 @ r_v0_0 - C_cv @ r_cv_v  # r_0v_c + r_vc_c

            y = self.data["y_k_j"][:, time_idx, :].T  # 20 x 4

            valid_idx = np.all(y >= 0, axis=1)
            self.landmarks = self.all_landmarks[valid_idx, :]
            self.landmark_ids = []  # unused
            self.y_ = y[valid_idx, :]
            self.theta = get_theta_from_C_r(C_c0, r_0c_c)  # corresponds to T_c0

        else:
            # for reproducibility
            np.random.seed(1)

            from scipy.spatial.transform import Rotation

            times = self.data.time_s.unique()
            # time = times[times >= min_time][0]
            print(f"choosing {time_idx} from {len(times)}")
            time = times[time_idx]

            # y_ is of shape n_positions * n_landmarks

            df_sub = self.data[self.data.time_s == time]
            r_c0_0 = df_sub.iloc[0][["cam_x", "cam_y", "cam_z"]].astype(float)
            q_c0 = df_sub.iloc[0][
                ["cam_rot_x", "cam_rot_y", "cam_rot_z", "cam_w"]
            ].astype(float)
            C_c0 = Rotation.from_quat(q_c0).as_matrix()

            r_0c_c = -C_c0 @ r_c0_0  #

            self.theta = get_theta_from_C_r(C_c0, r_0c_c)
            self.landmark_ids = list(df_sub.apriltag_id.unique())
            n_landmarks = len(self.landmark_ids)

            self.landmarks = np.zeros((n_landmarks, 3))
            self.y_ = np.zeros((n_landmarks, 4))
            for apriltag_id, df_subsub in df_sub.groupby("apriltag_id"):
                landmark_idx = self.landmark_ids.index(apriltag_id)
                pixels = df_subsub[["left_u", "left_v", "right_u", "right_v"]].median()
                self.landmarks[landmark_idx] = self.all_landmarks.loc[
                    self.all_landmarks.april_id == apriltag_id, ["x", "y", "z"]
                ]

                # sanity check
                T_cw = np.r_[np.c_[C_c0, r_0c_c], np.c_[np.zeros((1, 3)), 1.0]]
                r_iw_w = np.r_[self.landmarks[landmark_idx], 1.0]
                r_ic_c = T_cw @ r_iw_w
                p_i = self.M_matrix @ r_ic_c / r_ic_c[2]
                try:
                    assert all(np.abs(p_i - pixels.values) <= 100)
                except AssertionError:
                    print("WARNING: not good match in get_stereo_measurements...")
                    print("simulated pixels:", p_i)
                    print("measured pixels:", pixels.values)
                self.y_[landmark_idx, :] = pixels

    def get_lifter(
        self,
        time_idx,
        chosen_idx=-1,
    ):
        if self.data_type in ["apriltag_cal_individual", "stereo"]:
            self.get_stereo_measurements(time_idx=time_idx)

            n_landmarks = self.landmarks.shape[0]
            if n_landmarks <= self.params["min_n_landmarks"]:
                return None
            elif n_landmarks > self.params["max_n_landmarks"]:
                self.landmarks = self.landmarks[: self.params["max_n_landmarks"]]
                self.y_ = self.y_[: self.params["max_n_landmarks"], :]
            self.landmark_ids = []

            new_lifter = Stereo3DLifter(
                n_landmarks=self.landmarks.shape[0],
                level=self.params["level"],
                param_level="ppT",
            )
            if isinstance(self.all_landmarks, pd.DataFrame):
                new_lifter.all_landmarks = self.all_landmarks[["x", "y", "z"]].values
            else:
                new_lifter.all_landmarks = self.all_landmarks
            new_lifter.theta = self.theta
            new_lifter.landmarks = self.landmarks
            new_lifter.parameters = np.r_[1, self.landmarks.flatten()]
            new_lifter.M_matrix = self.M_matrix
            if self.params["use_gt"]:
                new_lifter.y_ = new_lifter.simulate_y(noise=self.params["sim_noise"])
            else:
                new_lifter.y_ = self.y_

        elif self.data_type == "uwb":
            combine_measurements = True
            n_positions = 1

            success = self.get_range_measurements(
                time_idx=time_idx,
                n_positions=n_positions,
                combine_measurements=combine_measurements,
                chosen_idx=chosen_idx,
            )
            if not success:
                return None

            new_lifter = RangeOnlyLocLifter(
                d=3,
                n_landmarks=self.landmarks.shape[0],
                n_positions=self.positions.shape[0],
                level=self.params["level"],
            )
            new_lifter.chosen_idx = chosen_idx

            new_lifter.theta = self.positions.flatten()
            if isinstance(self.all_landmarks, pd.DataFrame):
                new_lifter.all_landmarks = self.all_landmarks[["x", "y", "z"]].values
            else:
                new_lifter.all_landmarks = self.all_landmarks
            new_lifter.landmarks = self.landmarks
            new_lifter.parameters = np.r_[1.0, new_lifter.landmarks.flatten()]
            new_lifter.W = self.W_
            if self.params["use_gt"]:
                new_lifter.y_ = new_lifter.simulate_y(noise=self.params["sim_noise"])
            else:
                new_lifter.y_ = self.y_
        else:
            raise ValueError(f"Unknown data_type {self.data_type}")
        return new_lifter


def run_real_experiment(
    new_lifter,
    use_orders=["sorted"],
    add_oneshot=False,
    from_scratch=False,
    fname_root="",
    results_dir=RESULTS_DIR,
):
    fname_autotemplate = f"{results_dir}/autotemplate_{new_lifter}.pkl"
    if not from_scratch:
        try:
            with open(fname_autotemplate, "rb") as f:
                learner = pickle.load(f)
                order_dict = pickle.load(f)
        except FileNotFoundError:
            print(f"cannot read {fname_autotemplate}, need to run run_all_study first.")
            return
        order_dict = {k: v for k, v in order_dict.items() if k in use_orders}
    else:
        learner = Learner(lifter=new_lifter, variable_list=new_lifter.variable_list)
        learner.run()
        order_dict = {}
        if "basic" in use_orders:
            order_dict["basic"] = np.arange(len(learner.templates))
        if "sorted" in use_orders:
            order_dict["sorted"] = learner.generate_minimal_subset(
                reorder=True, tightness=new_lifter.TIGHTNESS
            )
        if "original" in use_orders:
            order_dict["original"] = learner.generate_minimal_subset(
                reorder=False, tightness=new_lifter.TIGHTNESS
            )

    plot = fname_root != ""

    df_data = []
    for name, new_order in order_dict.items():
        print(f"=========== running {name} ===============")
        data_dict = {}
        data_dict["type"] = name

        # apply the templates to all new landmarks
        new_learner = Learner(lifter=new_lifter, variable_list=new_lifter.variable_list)
        if from_scratch:
            new_learner.run()

        new_learner.templates = learner.get_sufficient_templates(new_order, new_lifter)
        new_learner.templates_known = new_learner.get_known_templates()
        new_learner.apply_templates()

        new_learner.find_local_solution(plot=plot)

        t1 = time.time()
        new_learner.is_tight(verbose=False, data_dict=data_dict)
        if abs(data_dict["RDG"]) > 1e-1 and abs(data_dict["max res"]) < 1e-5:
            print("Skipping invalid datapoint; are landmarks colinear?")
            print(new_learner.lifter.landmarks)
            continue

        data_dict[f"t solve SDP"] = time.time() - t1
        data_dict["gt theta"] = new_learner.lifter.theta
        success = new_learner.find_global_solution(data_dict=data_dict)

        if plot:
            fig = plt.gcf()
            ax = plt.gca()

            if success:
                that = data_dict["global theta"]
                cost = data_dict["global cost"]

                plot_frame(
                    ax,
                    theta=that,
                    color="k",
                    marker="o",
                    label=f"global min",
                    facecolors="none",
                    s=50,
                )
            ax.legend(framealpha=1.0, loc="lower left")
            fname = f"{fname_root}_local.pdf"
            savefig(fig, fname)

        df_data.append(deepcopy(data_dict))

    # try oneshot
    if add_oneshot:
        data_dict = {}
        data_dict["type"] = "from scratch"
        all_var_list = new_lifter.get_all_variables()
        new_lifter.param_level = "no"
        new_lifter.EPS_SVD = 1e-5
        new_learner = Learner(
            lifter=new_lifter, variable_list=all_var_list, apply_templates=False
        )
        data_new, success = new_learner.run(verbose=True)
        data_dict[f"t solve SDP"] = data_new[0]["t check tightness"]
        data_dict[f"t create constraints"] = data_new[0]["t learn templates"]
        data_dict[f"n constraints"] = data_new[0]["n templates"]
        data_dict.update(data_new[0])
        df_data.append(deepcopy(data_dict))

    df = pd.DataFrame(df_data)
    return df


def create_rmse_table(df, fname_root, add_n_landmarks=False):
    # use only once local solution per row (the best one)
    df.reset_index(inplace=True)
    for i, row in df.iterrows():
        if "local cost 2" in df.columns:
            local_min_idx = np.argmin(
                row[["local cost 0", "local cost 1", "local cost 2"]]
            )
        elif "local cost 1" in df.columns:
            local_min_idx = np.argmin(row[["local cost 0", "local cost 1"]])
        elif "local cost 0" in df.columns:
            local_min_idx = 0
        else:
            continue
        try:  # for stereo
            df.loc[i, "local C error"] = row[f"local {local_min_idx} C error"]
            df.loc[i, "local r error"] = row[f"local {local_min_idx} r error"]
        except KeyError:  # for ro
            df.loc[i, "local error"] = row[f"local {local_min_idx} error"]
        df.loc[i, "local cost"] = row[f"local cost {local_min_idx}"]
        df.loc[i, "local solution cert"] = row[f"local solution {local_min_idx} cert"]

    # drop the rows where no local solution was found.
    print("total number of datapoints:", len(df), end=", ")
    df = df[df["success rate"] != 1.0]
    print("after pruning:", len(df))
    if not len(df):
        return
    df = df.apply(pd.to_numeric, errors="ignore")

    # df.loc[:, "local C error"] = df[
    #    ["local 0 C error", "local 1 C error", "local 2 C error"]
    # ].min(axis=1)
    if "local r error" in df.columns:
        values = [
            "r error",
            "C error",
            "local r error",
            "local C error",
            # "local solution cert",
            # "global solution cert",
        ]
    else:
        values = [
            "error",
            "local error",
            # "local solution cert",
            # "global solution cert",
        ]
    index = ["dataset", "time index"]
    columns = ["n landmarks"] if add_n_landmarks else []
    pt = pd.pivot_table(
        data=df,
        values=values,
        index=index,
        columns=columns,
        sort=False,
    )
    pt.rename(
        columns={
            "local r error": "local $e_t$",
            "local C error": "local $e_C$",
            "r error": "global $e_t$",
            "C error": "global $e_C$",
            "local error": "local $e_t$",
            "error": "global $e_t$",
        },
        inplace=True,
    )
    error_table = pt.agg(["min", "max", "median", "mean", "std"])
    print(error_table)
    # s = error_table.style.highlight_min(props="itshape:; bfseries:;", axis=1)
    s = error_table.style
    if add_n_landmarks:
        out_name = fname_root + f"_error_table_landmarks.tex"
    else:
        out_name = fname_root + f"_error_table.tex"
    with open(out_name, "w"):
        s.to_latex(out_name)
    print(f"saved as {out_name}")


def run_experiments(
    exp: Experiment,
    n_successful=100,
    out_name="",
    stride=1,
    results_dir=RESULTS_DIR,
    start_idx=0,
):
    df_list = []
    counter = 0

    chosen_idx = 0
    for time_idx in np.arange(start_idx, 1900, step=stride):
        try:
            # choose which anchors are going to be used (for RO only)
            np.random.seed(time_idx)
            if ANCHOR_CHOICE == "random":
                chosen_idx = np.random.choice(len(DEGENERATE_DICT))
            elif ANCHOR_CHOICE == "round-robin":
                chosen_idx = (chosen_idx + 1) % len(DEGENERATE_DICT)
            elif ANCHOR_CHOICE == "all":
                chosen_idx = -1
            else:
                raise ValueError(ANCHOR_CHOICE)

            new_lifter = exp.get_lifter(
                time_idx=time_idx,
                chosen_idx=chosen_idx,
            )
        except IndexError:
            print(f"Warning: finished early: {counter}<{n_successful}")
            break
        if new_lifter is None:
            print(f"skipping {time_idx} because not enough valid landmarks")
            continue

        counter += 1
        if counter >= n_successful:
            break

        if counter < PLOT_NUMBER:
            fname_root = out_name.split(".")[0] + f"_{counter}"
        else:
            fname_root = ""
        df_here = run_real_experiment(
            new_lifter,
            use_orders=["sorted"],
            add_oneshot=False,
            fname_root=fname_root,
            results_dir=results_dir,
            from_scratch=False,
        )
        df_here["time index"] = time_idx
        df_here["chosen idx"] = chosen_idx
        df_here["n landmarks"] = new_lifter.n_landmarks
        if exp.data_type == "uwb":
            df_here["landmarks"] = str(sorted(exp.landmark_ids))
        df_list.append(df_here)

        if counter % 10 == 0:
            df = pd.concat(df_list)
            if out_name != "":
                df.to_pickle(out_name)
                print(f"===== saved intermediate as {out_name} ==========")
            print(df)

    df = pd.concat(df_list)
    if out_name != "":
        df.to_pickle(out_name)
        print(f"===== saved final as {out_name} ==========")
    return df
