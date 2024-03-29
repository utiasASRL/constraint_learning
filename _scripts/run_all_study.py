import pickle
import time

import numpy as np
import pandas as pd

from auto_template.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.wahba_lifter import WahbaLifter

RECOMPUTE = True

RESULTS_DIR = "_results_new"


def generate_results(lifters, seed=0, results_dir=RESULTS_DIR):
    all_list = []
    for Lifter, dict in lifters:
        np.random.seed(seed)
        lifter = Lifter(**dict)
        fname = f"{results_dir}/autotemplate_{lifter}.pkl"

        print(f"\n\n ======================== {lifter} ==========================")
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
        t1 = time.time()
        dict_list, success = learner.run(verbose=False, plot=False)
        if not success:
            raise RuntimeError(
                f"{lifter}: did not achieve {learner.lifter.TIGHTNESS} tightness."
            )

        t1 = time.time()
        idx_subset_reorder = learner.generate_minimal_subset(
            reorder=True,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
            tightness=learner.lifter.TIGHTNESS,
        )
        t_suff = time.time() - t1
        if idx_subset_reorder is None:
            print(f"{lifter}: did not find valid lamdas tightness.")

        idx_subset_original = learner.generate_minimal_subset(
            reorder=False,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
            tightness=learner.lifter.TIGHTNESS,
        )
        for d in dict_list:
            d["lifter"] = str(lifter)
            d["t find sufficient"] = t_suff
            d["n required"] = (
                len(idx_subset_reorder) if idx_subset_reorder is not None else None
            )
        all_list += dict_list

        order_dict = {"sorted": idx_subset_reorder, "original": idx_subset_original}
        with open(fname, "wb") as f:
            pickle.dump(learner, f)
            pickle.dump(order_dict, f)
        print(f"Wrote templates to {fname}")
    df = pd.DataFrame(all_list)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def run_all(recompute=RECOMPUTE, results_dir=RESULTS_DIR):
    np.random.seed(0)
    lifters = [
        # (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="no")),
        # (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="quad")),
        (Stereo2DLifter, dict(n_landmarks=3, param_level="ppT", level="urT")),
        (Stereo3DLifter, dict(n_landmarks=4, param_level="ppT", level="urT")),
        # (WahbaLifter, dict(n_landmarks=5, d=3, robust=True, level="xwT", n_outliers=1)),
        # (MonoLifter, dict(n_landmarks=6, d=3, robust=True, level="xwT", n_outliers=1)),
        # (WahbaLifter, dict(n_landmarks=4, d=3, robust=False, level="no", n_outliers=0)),
        # (MonoLifter, dict(n_landmarks=5, d=3, robust=False, level="no", n_outliers=0)),
    ]

    fname = f"{results_dir}/all_df_new.pkl"
    try:
        assert recompute is False
        df = pd.read_pickle(fname)
        print(f"read {fname}")
        lifters_str = set([str(L(**d)) for L, d in lifters])
        assert lifters_str.issubset(
            df.lifter.unique().astype(str)
        ), f"{lifters_str.difference(df.lifter.unique())} not in df"
        df = df[df.lifter.isin(lifters_str)]
    except (FileNotFoundError, AssertionError) as e:
        print(e)
        df = generate_results(lifters)
        df.to_pickle(fname)

    times = {
        "t learn templates": "$t_n$",
        "t apply templates": "$t_a$",
        "t check tightness": "$t_s$",
        "t find sufficient": "$t_r$",
    }
    lifter_names = {
        "wahba_3d_no_no": "PPR",
        "mono_3d_no_no": "PPL",
        "wahba_3d_xwT_no_robust": "rPPR",
        "mono_3d_xwT_no_robust": "rPLR",
        "rangeonlyloc3d_no": "RO ($z_n$)",
        "rangeonlyloc3d_quad": f"RO ($\\vc{{y}}_n$)",
        "stereo2d_urT_ppT": "stereo (2d)",
        "stereo3d_urT_ppT": "stereo (3d)",
    }
    fname = f"{results_dir}/all_df.tex"
    with open(fname, "w") as f:
        for out in (lambda x: print(x, end=""), f.write):
            out(
                f"problem & $n$ per variable group  & $N_l$ per variable group & \\# constraints & \\# required & {' & '.join(times.values())} & total [s] & RDG & SVR \\\\ \n"
            )
            out("\\midrule \n")
            for lifter, df_sub in df.groupby("lifter", sort=False):
                out(lifter_names[lifter] + " & ")
                out(str(df_sub["n dims"].values) + " & ")
                out(str(df_sub["n nullspace"].values) + " & ")
                out(str(df_sub["n constraints"].values[-1]) + " & ")
                out(str(df_sub["n required"].values[-1]) + " & ")
                for t in times.keys():
                    out(f"{df_sub[t].sum():.2f} &")
                out(f"{df_sub[times.keys()].sum().sum():.2f} & ")
                out(f"{abs(df_sub['RDG'].values[-1]):.2e} & ")
                out(f"{df_sub['SVR'].values[-1]:.2e} \\\\ \n")
    print("\nwrote above in", fname)


if __name__ == "__main__":
    run_all()
