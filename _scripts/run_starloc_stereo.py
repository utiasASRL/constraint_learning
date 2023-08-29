from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lifters.stereo3d_lifter import Stereo3DLifter
from pylgmath.so3.operations import hat

from utils.plotting_tools import savefig
from utils.real_experiments import Experiment, run_real_experiment

DATASET_ROOT = str(Path(__file__).parent.parent / "starloc")
MAX_N_LANDMARKS = 10
MIN_N_LANDMARKS = 8

USE_GT = False
SIM_NOISE = 0.1

RECOMPUTE = False


def create_lifter_from_data(
    dataset,
    time_idx=0,
    max_n_landmarks=MAX_N_LANDMARKS,
    min_n_landmarks=MIN_N_LANDMARKS,
):
    if dataset == "starrynight":
        from scipy.io import loadmat
        from scipy.spatial.transform import Rotation

        matfile = str(Path(__file__).parent.parent / "starrynight" / "dataset3.mat")
        data = loadmat(matfile)

        n_meas = data["t"].shape[1]
        if time_idx >= n_meas:
            print(f"Skipping {time_idx}>={n_meas}")
            return None

        print(f"Choosing time idx {time_idx} from {n_meas}")

        # axis-angle representation of gt rotation
        psi_v0 = data["theta_vk_i"][:, [time_idx]]
        psi_mag = np.linalg.norm(psi_v0)
        # conversion to rotation matrix
        C_v0 = (
            np.cos(psi_mag) * np.eye(3)
            + (1 - np.cos(psi_mag)) * (psi_v0 / psi_mag) @ (psi_v0 / psi_mag).T
            - np.sin(psi_mag) * hat(psi_v0 / psi_mag)
        )
        r_v0_0 = data["r_i_vk_i"][:, time_idx]  # gt translation

        C_cv = data["C_c_v"]
        r_cv_v = data["rho_v_c_v"].flatten()

        C_c0 = C_cv @ C_v0
        r_0c_c = -C_c0 @ r_v0_0 - C_cv @ r_cv_v
        a_c0 = Rotation.from_matrix(C_c0).as_euler("xyz")

        y = data["y_k_j"][:, time_idx, :].T  # 20 x 4

        valid_idx = np.all(y >= 0, axis=1)
        if np.sum(valid_idx) <= min_n_landmarks:
            return None
        elif np.sum(valid_idx) > max_n_landmarks:
            np.random.seed(0)
            valid = np.where(valid_idx)[0]
            valid_idx = np.zeros(len(valid_idx), dtype=bool)
            valid_idx[valid[:max_n_landmarks]] = 1
            assert sum(valid_idx) == max_n_landmarks

        y = y[valid_idx, :]
        all_landmarks = data["rho_i_pj_i"].T  # 20 x 3, all landmarks
        landmarks = all_landmarks[valid_idx, :]

        M_matrix = np.r_[
            np.c_[data["fu"], 0, data["cu"], data["fu"] * data["b"] / 2],
            np.c_[0, data["fv"], data["cv"], 0],
            np.c_[data["fu"], 0, data["cu"], -data["fu"] * data["b"] / 2],
            np.c_[0, data["fv"], data["cv"], 0],
        ]
        new_lifter = Stereo3DLifter(
            n_landmarks=landmarks.shape[0], level="urT", param_level="ppT"
        )
        new_lifter.theta = np.r_[r_0c_c, a_c0]
        new_lifter.landmarks = landmarks
        new_lifter.parameters = np.r_[1, landmarks.flatten()]
        new_lifter.M_matrix = M_matrix
        if USE_GT:
            new_lifter.y_ = new_lifter.simulate_y(noise=SIM_NOISE)
        else:
            new_lifter.y_ = y
        return new_lifter

    else:
        data_type = "apriltag_cal_individual"
        exp = Experiment(DATASET_ROOT, dataset, data_type)
        # exp.get_stereo_measurements(
        #    time_idx=time_idx, stereo_type="gt_", extra_noise=0.1
        # )
        exp.get_stereo_measurements(time_idx=time_idx, stereo_type="")

        n_landmarks = exp.landmarks.shape[0]
        if n_landmarks <= min_n_landmarks:
            return None
        elif n_landmarks > max_n_landmarks:
            exp.landmarks = exp.landmarks[:max_n_landmarks]
            exp.y_ = exp.y_[:max_n_landmarks, :]

        new_lifter = Stereo3DLifter(
            n_landmarks=exp.landmarks.shape[0], level="urT", param_level="ppT"
        )
        new_lifter.theta = exp.theta
        new_lifter.landmarks = exp.landmarks
        new_lifter.parameters = np.r_[1, exp.landmarks.flatten()]
        new_lifter.M_matrix = exp.M_matrix
        if USE_GT:
            new_lifter.y_ = new_lifter.simulate_y(noise=SIM_NOISE)
        else:
            new_lifter.y_ = exp.y_
        return new_lifter


def run_all(dataset, out_name="", n_successful=100):
    df_list = []
    counter = 0
    for time_idx in np.arange(1900):
        try:
            new_lifter = create_lifter_from_data(dataset=dataset, time_idx=time_idx)
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
        x = new_lifter.get_x()
        t = x[1:4]
        C = x[4:13].reshape(3, 3)
        if len(df_here) == 1:
            df_here["landmarks"] = [new_lifter.landmarks]
            df_here["t"] = [t]
            df_here["C"] = [C]
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


def plot_other_stats(df):
    trans = np.array([*df.t.values])
    spread = np.empty(len(df))
    for i, row in df.iterrows():
        d_matrix = np.linalg.norm(
            row.landmarks[None, :, :] - row.landmarks[:, None, :], axis=2
        )
        spread[i] = np.max(d_matrix)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=trans[:, 2], y="RDG", style="n landmarks", ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("camera z [m]")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=spread, y="RDG", style="n landmarks", ax=ax, hue="q")
    ax.set_yscale("log")
    ax.set_xlabel("landmark spread [m]")

    fig, ax = plt.subplots()
    sns.scatterplot(x=df.q, y=df["max res"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("q value")
    ax.set_ylabel("max residual")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="max res",
        y="RDG",
        style="n landmarks",
        ax=ax,
        hue=np.log10(df.q),
    )
    ax.legend(loc="upper left", bbox_to_anchor=[1.0, 1.0], title="log-q")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("max residual")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="q", y="RDG", style="n landmarks", hue="dataset", ax=ax)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("q value")

    for dataset, df_sub in df.groupby("dataset"):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        ax.scatter(range(len(df_sub.RDG)), df_sub["n local"], label="local")
        ax.scatter(range(len(df_sub.RDG)), df_sub["n global"], label="global")
        ax.scatter(range(len(df_sub.RDG)), df_sub["n fail"], label="failed")
        ax.legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
        ax.set_ylabel("number")
        ax.set_xlabel("index")
        ax.set_title(dataset)
        plt.tight_layout()


if __name__ == "__main__":
    df_list = []
    n_successful = 100
    # datasets = ["eight_s3", "starrynight"]
    datasets = ["zigzag_s3", "loop-2d_s4", "eight_s3", "starrynight"]
    for dataset in datasets:
        if USE_GT:
            fname = f"_results/{dataset}_output_{n_successful}.pkl"
        else:
            fname = f"_results/{dataset}_output_{n_successful}_gt.pkl"
        try:
            assert RECOMPUTE is False
            df_all = pd.read_pickle(fname)
        except (AssertionError, FileNotFoundError):
            df_all = run_all(dataset, out_name=fname, n_successful=n_successful)
        df_all["dataset"] = dataset
        df_list.append(df_all)

    constraint_type = "sorted"
    df = pd.concat(df_list)
    df = df[df.type == constraint_type]
    df["RDG"] = df["RDG"].abs()

    for x in ["total error", "cond Hess", "max res", "q"]:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        sns.scatterplot(
            data=df, x=x, y="RDG", style="n landmarks", ax=ax, hue="dataset"
        )
        ax.legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel("absolute RDG")
        ax.set_xlabel(x)
        plt.tight_layout()
        savefig(fig, f"_results/stereo_{x.replace(' ', '_')}.pdf")

    fig, ax = plt.subplots()
    df["success rate"] = df["n global"] / (
        df["n global"] + df["n fail"] + df["n local"]
    )
    sns.scatterplot(
        data=df,
        x="success rate",
        y="RDG",
        style="n landmarks",
        ax=ax,
        hue="dataset",
    )
    ax.set_yscale("log")
    plt.show()
    print("done")
