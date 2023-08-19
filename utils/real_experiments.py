from copy import deepcopy
import pickle
import time

import numpy as np
import pandas as pd

from lifters.learner import Learner

from starloc.reader import read_landmarks, read_data, read_calib


class Experiment(object):
    def __init__(self, dataset_root, dataset, data_type, from_id=None):
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
        self,
        min_time=0,
        n_positions=None,
        range_type="range",
        combine_measurements=True,
    ):
        if combine_measurements:
            ref_id = self.landmark_ids[0]
            times = self.data.loc[self.data.to_id == ref_id]
            self.data["position_idx"] = np.cumsum(self.data.to_id.values == ref_id)
        else:
            times = self.data.time_s.unique()
            times = list(times[times >= min_time])
            self.data["position_idx"] = range(len(times))

        if n_positions is None:
            n_positions = len(self.data.position_idx.unique())

        self.data = self.data[self.data["position_idx"] < n_positions]
        n_landmarks = len(self.landmarks)

        # y_ is of shape n_positions * n_landmarks
        self.y_ = np.zeros((n_positions, n_landmarks))
        self.W_ = np.zeros((n_positions, n_landmarks))
        self.positions = np.empty((n_positions, 3))
        for position_idx, df_sub in self.data.groupby("position_idx"):
            position = df_sub[["x", "y", "z"]].mean()
            self.positions[position_idx, :] = position
            for __, row in df_sub.iterrows():
                landmark_idx = self.landmark_ids.index(row.to_id)
                self.W_[position_idx, landmark_idx] = 1.0
                self.y_[position_idx, landmark_idx] = row[range_type]

        self.landmarks = self.all_landmarks[["x", "y", "z"]].values

    def get_stereo_measurements(self, time_idx=0, stereo_type="", extra_noise=0.0):
        """
        :param stereo_type: can be either "" (normal data) or "gt_".


        We use the following conventions:
        - r_c_0c is the vector from c to world expressed in c frame (we use "0" to denote world frame)
        - C_c0 is the rotation to change a vector expressed in world to the same vector expressed in c.

        With these conventions, we have:
        [r_ic_c]  =  [C_c0  r_0c_c] [r_i0_0]
        [  1   ]     [ 0     1    ] [  1   ]
                     -----T_cw-----
        """

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
        q_c0 = df_sub.iloc[0][["cam_rot_x", "cam_rot_y", "cam_rot_z", "cam_w"]].astype(
            float
        )
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
            pixels = df_subsub[
                [
                    f"{stereo_type}left_u",
                    f"{stereo_type}left_v",
                    f"{stereo_type}right_u",
                    f"{stereo_type}right_v",
                ]
            ].median()
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
                print("simulated pixels:", p_i)
                print("measured pixels:", pixels.values)
                raise

            if extra_noise > 0:
                pixels += np.random.normal(scale=extra_noise, loc=0, size=len(pixels))
            self.y_[landmark_idx, :] = pixels


def run_real_experiment(
    new_lifter, add_oneshot=True, add_original=False, add_sorted=False
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
        print(f"=========== running {name} ===============")
        data_dict = {}
        data_dict["type"] = name

        # apply the templates to all new landmarks
        new_learner = Learner(lifter=new_lifter, variable_list=new_lifter.variable_list)
        new_learner.scale_templates(learner, new_order, data_dict)

        t1 = time.time()
        new_learner.is_tight(verbose=True, data_dict=data_dict)
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
        data_dict[f"n templates"] = data_new[0]["n templates"]
        data_dict[f"n constraints"] = data_new[0]["n templates"]
        data_dict["RDG"] = data_new[0]["RDG"]
        data_dict["SVR"] = data_new[0]["SVR"]
        data_dict.update(data_new[0]["error dict"])
        df_data.append(deepcopy(data_dict))

    df = pd.DataFrame(df_data)
    return df
