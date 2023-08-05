import pandas as pd
import numpy as np

from lifters.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.wahba_lifter import WahbaLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter


def run_all(lifters, seed=0):
    all_list = []
    for Lifter, dict in lifters:
        np.random.seed(seed)
        lifter = Lifter(**dict)
        print(f"\n\n ======================== {lifter} ==========================")
        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list
        )
        dict_list = learner.run(verbose=True, use_known=False, plot=False, tightness="cost")
        for d in dict_list:
            d["lifter"] = str(lifter)
        all_list += dict_list
    df = pd.DataFrame(all_list)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


if __name__ == "__main__":

    # fix seed for reproducibility

    lifters = [
        #(WahbaLifter, dict(n_landmarks=3, d=2, robust=False, level="no")), # ok with and without B_list
        #(WahbaLifter, dict(n_landmarks=4, d=3, robust=False, level="no")), # ok "
        #(WahbaLifter, dict(n_landmarks=3, d=2, robust=True, level="xwT")), # ok "
        #(WahbaLifter, dict(n_landmarks=4, d=3, robust=True, level="xwT")), # ok "
        #(MonoLifter, dict(n_landmarks=4, d=2, robust=False, level="no")),  # ok
        #(MonoLifter, dict(n_landmarks=8, d=3, robust=False, level="no")),  # ok
        #(MonoLifter, dict(n_landmarks=4, d=2, robust=True, level="xwT")),  # ok (super small violation:  2.92e-06 dual vs. 2.91e-06 qcqp)
        #(MonoLifter, dict(n_landmarks=8, d=3, robust=True, level="xwT")),  # ok
        (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="no")), # ok
        (RangeOnlyLocLifter, dict(n_positions=3, n_landmarks=10, d=3, level="quad")), # ok
        #(Stereo2DLifter, dict(n_landmarks=3, param_level="ppT", level="urT")), # ok
        #(Stereo3DLifter, dict(n_landmarks=4, param_level="ppT", level="urT")), # ok
    ]
    recompute = True

    try:
        assert recompute is False
        df = pd.read_pickle("_results/all_df.pkl")
        lifters_str = set([str(l) for l in lifters])
        assert lifters_str.issubset(df.lifter.unique().astype(str)), f"{lifters_str.difference(df.lifter.unique())} not in df"
        df = df[df.lifter.isin(lifters_str)]
    except (FileNotFoundError, AssertionError) as e:
        print(e)
        df = run_all(lifters)
        df.to_pickle("_results/all_df.pkl")

    times = ['t check tightness', 't learn templates', 't apply templates']
    with open("_results/all_df.tex", "w") as f:
        for out in (lambda x: print(x, end=""), f.write):
            out("problem& variables& dimension Y& time learn& time apply& time SDP& time total \\\\ \n")
            for lifter, df_sub in df.groupby("lifter", sort=False):
                out(lifter.replace("_", "-") + " & ")
                out("[")
                for vars in df_sub.variables:
                    out("[")
                    for v in vars[:-1]:
                        out(f"${v}$, ")
                    out(f"${vars[-1]}$] ")
                out("] & ")
                out(str(df_sub["n dims"].values) + " & ")
                for t in times:
                    out(f"{df_sub[t].sum():.4f} &")
                out(f"{df_sub[times].sum().sum():.4f} \\\\ \n")