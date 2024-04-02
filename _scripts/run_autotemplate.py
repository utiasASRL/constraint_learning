import pickle
import time

import numpy as np
import pandas as pd

from auto_template.learner import Learner
from auto_template.sim_experiments import save_autotight_order
from lifters.mono_lifter import MonoLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.wahba_lifter import WahbaLifter

RECOMPUTE = True

RESULTS_DIR = "_results_server_v3"

LIFTERS_NO = [
    (Stereo2DLifter, dict(n_landmarks=3, param_level="no", level="no")),
    (Stereo3DLifter, dict(n_landmarks=4, param_level="no", level="no")),
]

LIFTERS = [
    (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="no")),
    (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="quad")),
    (Stereo2DLifter, dict(n_landmarks=3, param_level="ppT", level="urT")),
    (Stereo3DLifter, dict(n_landmarks=4, param_level="ppT", level="urT")),
    (WahbaLifter, dict(n_landmarks=4, d=3, robust=False, level="no", n_outliers=0)),
    (MonoLifter, dict(n_landmarks=5, d=3, robust=False, level="no", n_outliers=0)),
    (WahbaLifter, dict(n_landmarks=5, d=3, robust=True, level="xwT", n_outliers=1)),
    (MonoLifter, dict(n_landmarks=6, d=3, robust=True, level="xwT", n_outliers=1)),
]


def generate_results(lifters, seed=0, results_dir=RESULTS_DIR):
    all_list = []
    for Lifter, dict in lifters:
        np.random.seed(seed)
        lifter = Lifter(**dict)
        fname_root = f"{results_dir}/autotemplate_{lifter}"
        fname = f"{fname_root}.pkl"

        print(f"\n\n ======================== {lifter} ==========================")
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list, n_inits=1)
        t1 = time.time()
        dict_list, success = learner.run(verbose=False, plot=False)

        t1 = time.time()
        idx_subset_reorder = learner.generate_minimal_subset(
            reorder=True,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
            tightness=learner.lifter.TIGHTNESS,
        )
        t_suff = time.time() - t1

        idx_subset_original = learner.generate_minimal_subset(
            reorder=False,
            use_bisection=learner.lifter.TIGHTNESS == "cost",
            tightness=learner.lifter.TIGHTNESS,
        )

        save_autotight_order(
            learner, fname_root, use_bisection=learner.lifter.TIGHTNESS == "cost"
        )

        if not success:
            print(f"{lifter}: did not achieve {learner.lifter.TIGHTNESS} tightness.")
            continue

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
    # Run lifters that are not tight
    if recompute:
        np.random.seed(0)
        generate_results(LIFTERS_NO, results_dir=results_dir)

    # Run lifter that are tight
    fname = f"{results_dir}/all_df_new.pkl"
    try:
        assert recompute is False
        df = pd.read_pickle(fname)
        print(f"read {fname}")
        lifters_str = set([str(L(**d)) for L, d in LIFTERS])
        assert lifters_str.issubset(
            df.lifter.unique().astype(str)
        ), f"{lifters_str.difference(df.lifter.unique())} not in df"
        df = df[df.lifter.isin(lifters_str)]
    except (FileNotFoundError, AssertionError) as e:
        print(e)
        np.random.seed(0)
        df = generate_results(LIFTERS, results_dir=results_dir)
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
            header = (
                [
                    f"Problem & Dimension $n$ per iteration",
                    "\\# Constraints",
                    "\\# Reduced",
                ]
                + [f"{t} [s]" for t in times.values()]
                + [
                    "total [s]",
                    "RDG",
                    "SVR",
                ]
            )
            out(" & ".join(header) + "\\\\ \n")
            out("\\midrule \n")
            for lifter, df_sub in df.groupby("lifter", sort=False):
                out(lifter_names[lifter] + " & ")
                out(str(df_sub["n dims"].values) + " & ")
                # out(str(df_sub["n nullspace"].values) + " & ")
                out(str(df_sub["n constraints"].values[-1]) + " & ")
                out(str(df_sub["n required"].values[-1]) + " & ")
                for t in times.keys():
                    out(f"{df_sub[t].sum():.2f} &")
                out(f"{df_sub[times.keys()].sum().sum():.2f} & ")
                rdg = abs(df_sub["RDG"].values[-1])
                svr = df_sub["SVR"].values[-1]
                if svr > 1e7:
                    out(f"{rdg:.2e} & ")
                    out(f"\\textbf{{{svr:.2e}}} \\\\ \n")
                else:
                    out(f"\\textbf{{{rdg:.2e}}} & ")
                    out(f"{svr:.2e} \\\\ \n")
    print("\nwrote above in", fname)


if __name__ == "__main__":
    run_all(recompute=False)
