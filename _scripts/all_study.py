import pandas as pd

from lifters.learner import Learner
from lifters.mono_lifter import MonoLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from lifters.range_only_lifters import RangeOnlyLocLifter


def run_all(lifters):
    all_list = []
    for lifter in lifters:
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
    
    lifters = [
        RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="no"),
        RangeOnlyLocLifter(n_positions=3, n_landmarks=10, d=3, level="quad"),
        Stereo2DLifter(n_landmarks=3, param_level="ppT", level="urT"),
        Stereo3DLifter(n_landmarks=4, param_level="ppT", level="urT"),
        MonoLifter(n_landmarks=4, d=2, robust=False, level="no"),
        MonoLifter(n_landmarks=8, d=3, robust=False, level="no"),
        MonoLifter(n_landmarks=4, d=2, robust=True, level="xwT"),
        MonoLifter(n_landmarks=8, d=3, robust=True, level="xwT"),
    ]
    try:
        df = pd.read_pickle("_results/all_df.pkl")
        lifters_str = set([str(l) for l in lifters])
        assert lifters_str.issubset(df.lifter.unique().astype(str))
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