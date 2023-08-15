from pathlib import Path

import numpy as np

from lifters.range_only_lifters import RangeOnlyLocLifter

from utils.real_experiments import Experiment, run_real_experiment

DATASET_ROOT = str(Path(__file__).parent.parent / "starloc")


def create_lifter_from_data(
    dataset, from_id=1, level="quad", combine_measurements=True
):
    data_type = "uwb"
    exp = Experiment(DATASET_ROOT, dataset, data_type, from_id=from_id)

    # range_type = "gt_range"
    range_type = "range"
    min_time = 10  # in seconds
    n_positions = 12
    exp.get_range_measurements(
        min_time=min_time,
        n_positions=n_positions,
        range_type=range_type,
        combine_measurements=combine_measurements,
    )

    new_lifter = RangeOnlyLocLifter(
        d=3,
        n_landmarks=exp.landmarks.shape[0],
        n_positions=exp.positions.shape[0],
        level=level,
    )
    new_lifter.theta = exp.positions.flatten()
    new_lifter.landmarks = exp.landmarks
    new_lifter.parameters = np.r_[1.0, new_lifter.landmarks.flatten()]
    new_lifter.W = exp.W_
    new_lifter.y_ = exp.y_
    return new_lifter


if __name__ == "__main__":
    dataset = "loop-3d_s3"
    # level = "quad"
    level = "no"
    new_lifter = create_lifter_from_data(
        dataset=dataset, from_id=1, level=level, combine_measurements=True
    )
    run_real_experiment(new_lifter)
