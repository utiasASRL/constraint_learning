from copy import deepcopy
import pickle
import time

import numpy as np
import pandas as pd

from lifters.learner import Learner

from starloc.reader import read_landmarks, read_data


class Experiment(object):
    def __init__(self, dataset_root, dataset, data_type, from_id=1):
        self.landmarks = read_landmarks(
            dataset_root=dataset_root, dataset=dataset, data=[data_type]
        )
        data = read_data(dataset_root=dataset_root, dataset=dataset, data=[data_type])
        self.data = data.loc[data.from_id == from_id]
        self.landmark_ids = list(self.landmarks.id.values)

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
            times = list(times[times > min_time])
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

    def get_stereo_measurements(
        self,
        min_time=0,
        n_positions=None,
    ):
        times = self.data.time_s.unique()
        times = list(times[times > min_time])
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


def run_real_experiment(new_lifter):
    # set distance measurements
    fname_root = f"_results/scalability_{new_lifter}"
    fname = fname_root + "_order_dict.pkl"
    with open(fname, "rb") as f:
        order_dict = pickle.load(f)
        learner = pickle.load(f)

    df_data = []
    for name, new_order in order_dict.items():
        data_dict = {}
        data_dict["type"] = name

        new_learner = Learner(lifter=new_lifter, variable_list=new_lifter.variable_list)
        success = new_learner.find_local_solution()

        # apply the templates to all new landmarks
        new_learner.scale_templates(learner, new_order, data_dict)

        print(f"=========== tightness test: {name} ===============")
        t1 = time.time()
        new_learner.is_tight(verbose=True, data_dict=data_dict)
        data_dict[f"t solve SDP"] = time.time() - t1
        df_data.append(deepcopy(data_dict))
    df = pd.DataFrame(df_data)
    fname = fname_root + "_real.pkl"
    print(df)
    df.to_pickle(fname)
    print("saved as", fname)
