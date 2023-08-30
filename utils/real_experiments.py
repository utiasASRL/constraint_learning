from copy import deepcopy
import os
import pickle
import time

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylgmath.so3.operations import hat

from lifters.learner import Learner

from starloc.reader import read_landmarks, read_data, read_calib
from utils.plotting_tools import savefig

from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo3d_lifter import Stereo3DLifter

RANGE_TYPE = "range_calib"


def plot_results(df, ylabel="RDG", fname_root=""):
    kwargs = {"edgecolor": "none"}
    for x in ["total error", "cond Hess", "max res", "q"]:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        sns.scatterplot(
            data=df, x=x, y=ylabel, style="n landmarks", ax=ax, hue="dataset", **kwargs
        )
        ax.legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(x)
        if fname_root != "":
            savefig(fig, f"{fname_root}_{x.replace(' ', '_')}.pdf")

    fig, ax = plt.subplots()
    df["success rate"] = df["n global"] / (
        df["n global"] + df["n fail"] + df["n local"]
    )
    sns.scatterplot(
        data=df,
        x="success rate",
        y=ylabel,
        style="n landmarks",
        ax=ax,
        hue="dataset",
    )
    ax.set_yscale("log")


class Experiment(object):
    def __init__(self, dataset_root, dataset, data_type, from_id=None):
        self.dataset = dataset
        self.data_type = data_type

        if dataset == "starrynight":
            from scipy.io import loadmat

            matfile = os.path.join(dataset_root, "starrynight", "dataset3.mat")
            self.data = loadmat(matfile)

            self.all_landmarks = self.data["rho_i_pj_i"].T  # 20 x 3, all landmarks

            # fmt: off
            self.M_matrix = np.r_[
                np.c_[ self.data["fu"], 0, self.data["cu"], self.data["fu"] * self.data["b"] / 2, ],
                np.c_[0, self.data["fv"], self.data["cv"], 0],
                np.c_[ self.data["fu"], 0, self.data["cu"], -self.data["fu"] * self.data["b"] / 2, ],
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

            if "apriltag" in data_type:
                calib_dict = read_calib(dataset_root=dataset_root, dataset=dataset)
                # fmt: off
                self.M_matrix = np.r_[ 
                    np.c_[ calib_dict["fu"], 0, calib_dict["cu"], 0],
                    np.c_[ 0, calib_dict["fv"], calib_dict["cv"], 0],
                    np.c_[ calib_dict["fu"], 0, calib_dict["cu"], -calib_dict["fu"] * calib_dict["b"]],
                    np.c_[ 0, calib_dict["fv"], calib_dict["cv"], 0],
                ]
                # fmt: on

    def get_range_measurements(
        self, time_idx=0, n_positions=None, combine_measurements=True, min_n_landmarks=4
    ):
        if not "position_idx" in self.data.columns:
            if combine_measurements:
                ref_id = self.landmark_ids[0]
                times = self.data.loc[self.data.to_id == ref_id]
                self.data["position_idx"] = np.cumsum(self.data.to_id.values == ref_id)
            else:
                times = self.data.time_s.unique()
                self.data["position_idx"] = range(len(times))

            # find the positions at which we measure at least min_n_landmarks.
            valid_position_idx = self.data.position_idx.value_counts().gt(
                min_n_landmarks
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

        assert len(data_here.position_idx.unique()) == n_positions

        landmarks = self.all_landmarks.loc[
            self.all_landmarks.id.isin(data_here.to_id.unique())
        ]
        self.landmarks = landmarks[["x", "y", "z"]].values
        self.landmark_ids = list(landmarks.id.values)
        n_landmarks = self.landmarks.shape[0]

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
                landmark_idx = self.landmark_ids.index(row.to_id)
                self.W_[position_idx, landmark_idx] = 1.0
                self.y_[position_idx, landmark_idx] = row[RANGE_TYPE]

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
            r_0c_c = -C_c0 @ r_v0_0 - C_cv @ r_cv_v
            a_c0 = Rotation.from_matrix(C_c0).as_euler("xyz")

            y = self.data["y_k_j"][:, time_idx, :].T  # 20 x 4

            valid_idx = np.all(y >= 0, axis=1)
            self.landmarks = self.all_landmarks[valid_idx, :]
            self.y_ = y[valid_idx, :]
            self.theta = np.r_[r_0c_c, a_c0]

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
            a_c0 = Rotation.from_matrix(C_c0).as_euler("xyz")

            self.theta = np.r_[r_0c_c, a_c0]
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
        level,
        min_n_landmarks,
        max_n_landmarks,
        use_gt=False,
        sim_noise=None,
    ):
        if self.data_type in ["apriltag_cal_individual", "stereo"]:
            self.get_stereo_measurements(time_idx=time_idx)

            n_landmarks = self.landmarks.shape[0]
            if n_landmarks <= min_n_landmarks:
                return None
            elif n_landmarks > max_n_landmarks:
                self.landmarks = self.landmarks[:max_n_landmarks]
                self.y_ = self.y_[:max_n_landmarks, :]

            new_lifter = Stereo3DLifter(
                n_landmarks=self.landmarks.shape[0], level=level, param_level="ppT"
            )
            new_lifter.theta = self.theta
            new_lifter.landmarks = self.landmarks
            new_lifter.parameters = np.r_[1, self.landmarks.flatten()]
            new_lifter.M_matrix = self.M_matrix
            if use_gt:
                new_lifter.y_ = new_lifter.simulate_y(noise=sim_noise)
            else:
                new_lifter.y_ = self.y_

        elif self.data_type == "uwb":
            if min_n_landmarks or max_n_landmarks:
                raise NotImplementedError(
                    "min / max landmarks not implemented for range measurements yet"
                )
            combine_measurements = True
            n_positions = 3
            self.get_range_measurements(
                time_idx=time_idx,
                n_positions=n_positions,
                combine_measurements=combine_measurements,
            )

            new_lifter = RangeOnlyLocLifter(
                d=3,
                n_landmarks=self.landmarks.shape[0],
                n_positions=self.positions.shape[0],
                level=level,
            )
            new_lifter.theta = self.positions.flatten()
            new_lifter.landmarks = self.landmarks
            new_lifter.parameters = np.r_[1.0, new_lifter.landmarks.flatten()]
            new_lifter.W = self.W_
            if use_gt:
                new_lifter.y_ = new_lifter.simulate_y()
            else:
                new_lifter.y_ = self.y_
        else:
            raise ValueError(f"Unknown data_type {data_type}")
        return new_lifter


def run_real_experiment(
    new_lifter,
    add_oneshot=True,
    add_original=False,
    add_sorted=False,
    add_basic=True,
    from_scratch=False,
):
    # set distance measurements
    fname_root = f"_results/scalability_{new_lifter}"
    fname = fname_root + "_order_dict.pkl"
    with open(fname, "rb") as f:
        order_dict = pickle.load(f)
        learner = pickle.load(f)

    df_data = []
    for name, new_order in order_dict.items():
        if name == "original" and not add_original:
            print("skipping original")
            continue
        if name == "sorted" and not add_sorted:
            print("skipping sorted")
            continue
        if name == "basic" and not add_basic:
            print("skipping basic")
            continue
        print(f"=========== running {name} ===============")
        data_dict = {}
        data_dict["type"] = name

        # apply the templates to all new landmarks
        new_learner = Learner(lifter=new_lifter, variable_list=new_lifter.variable_list)

        if from_scratch:
            new_learner.run()
        else:
            new_learner.scale_templates(learner, new_order, data_dict)

        t1 = time.time()
        new_learner.is_tight(verbose=False, data_dict=data_dict)
        data_dict[f"t solve SDP"] = time.time() - t1
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


def run_all(
    exp: Experiment,
    level,
    min_n_landmarks,
    max_n_landmarks,
    use_gt=False,
    sim_noise=None,
    n_successful=100,
    out_name="",
):
    df_list = []
    counter = 0
    for time_idx in np.arange(1900):
        try:
            new_lifter = exp.get_lifter(
                time_idx=time_idx,
                level=level,
                min_n_landmarks=min_n_landmarks,
                max_n_landmarks=max_n_landmarks,
                use_gt=use_gt,
                sim_noise=sim_noise,
            )
        except IndexError:
            print(f"Warning: finished early: {counter}<{n_successful}")
            break
        if new_lifter is None:
            print(f"skipping {time_idx} because not enough valid landmarks")
            continue

        counter += 1
        df_here = run_real_experiment(
            new_lifter,
            add_oneshot=False,
            add_sorted=True,
            add_original=False,
            add_basic=False,
        )
        df_here["time index"] = time_idx
        df_here["n landmarks"] = new_lifter.n_landmarks
        df_list.append(df_here)

        if counter >= n_successful:
            break
        elif counter % 10 == 0:
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
