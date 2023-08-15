from copy import deepcopy
from pathlib import Path
import pickle
import time

import pandas as pd
import numpy as np

from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.learner import Learner

from starloc.reader import read_landmarks, read_data

if __name__ == "__main__":
    dataset_root = str(Path(__file__).parent.parent / "starloc")
    dataset = "loop-3d_s3"
    data_type = "uwb"
    from_id = 1
    level = "quad"
    range_type = "gt_range"
    min_time = 10  # in seconds
    n_positions = 12

    landmarks = read_landmarks(
        dataset_root=dataset_root, dataset=dataset, data=[data_type]
    )
    data = read_data(dataset_root=dataset_root, dataset=dataset, data=[data_type])
    data = data.loc[data.from_id == from_id]

    times = data.time_s.unique()
    times = list(times[times > min_time][:n_positions])

    n_landmarks = len(landmarks)
    new_lifter = RangeOnlyLocLifter(
        d=3, n_landmarks=n_landmarks, n_positions=n_positions, level=level
    )

    # y_ is of shape n_positions * n_landmarks
    landmark_ids = list(landmarks.id.values)
    y_ = np.zeros((n_positions, n_landmarks))
    W_ = np.zeros((n_positions, n_landmarks))
    positions = np.empty((n_positions, 3))
    for t in times:
        df_sub = data[data.time_s == t]
        for i, row in df_sub.iterrows():
            landmark_idx = landmark_ids.index(row.to_id)
            position_idx = times.index(row.time_s)
            W_[position_idx, landmark_idx] = 1.0
            y_[position_idx, landmark_idx] = row[range_type]
            positions[position_idx, :] = [row.tag_pos_x, row.tag_pos_y, row.tag_pos_z]

    new_lifter.theta = positions.flatten()
    new_lifter.landmarks = landmarks[["x", "y", "z"]].values
    new_lifter.parameters = np.r_[1.0, new_lifter.landmarks.flatten()]
    new_lifter.W = W_
    new_lifter.y_ = y_

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
        new_learner.is_tight(verbose=True)
        data_dict[f"t solve SDP"] = time.time() - t1
        df_data.append(deepcopy(data_dict))
    df = pd.DataFrame(df_data)
    df.to_pickle(fname_root + "_real")
