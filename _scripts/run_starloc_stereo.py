from pathlib import Path

import numpy as np

from lifters.stereo3d_lifter import Stereo3DLifter

from utils.real_experiments import Experiment, run_real_experiment

DATASET_ROOT = str(Path(__file__).parent.parent / "starloc")


def create_lifter_from_data(dataset):
    data_type = "apriltag"
    exp = Experiment(DATASET_ROOT, dataset, data_type)

    min_time = 10  # in seconds
    exp.get_stereo_measurements(
        min_time=min_time,
    )

    new_lifter = Stereo3DLifter(
        n_landmarks=exp.landmarks.shape[0], level="urT", param_level="ppT"
    )
    new_lifter.landmarks = exp.landmarks
    new_lifter.parameters = np.r_[1, exp.landmarks.flatten()]
    new_lifter.y_ = exp.y_
    new_lifter.M_matrix = exp.M_matrix
    return new_lifter


if __name__ == "__main__":
    dataset = "loop-3d_s3"
    # level = "quad"
    level = "no"
    new_lifter = create_lifter_from_data(dataset=dataset)
    run_real_experiment(new_lifter)
