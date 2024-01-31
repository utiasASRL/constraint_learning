import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

from decomposition.sim_experiments import generate_results
from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg
from utils.plotting_tools import USE_METHODS, savefig

# USE_METHODS = ["SDP", "dSDP", "ADMM"]
# USE_METHODS = ["local", "dSDP", "ADMM", "pADMM"]
# USE_METHODS = ["ADMM", "pADMM"]
ADD_REDUNDANT = True

RESULTS_READ = "_results_server"
RESULTS_WRITE = "_results"

if __name__ == "__main__":
    overwrite = False
    n_threads_list = [10]

    # n_params_list = np.logspace(1, 2, 10).astype(int)
    # appendix = "time"

    # goes up to 3000 (after that we have memory error).
    n_params_list = np.logspace(1, 6, 21).astype(int)[:11]
    appendix = "alltime"

    # n_params_list = [10, 20, 30]
    # n_threads_list = [2]
    # appendix = "test"

    np.random.seed(0)
    lifter_ro = RangeOnlyLocLifter(
        n_landmarks=8, n_positions=10, reg=Reg.CONSTANT_VELOCITY, d=2
    )
    lifter_mat = MatWeightLocLifter(n_landmarks=8, n_poses=10)
    for lifter in [lifter_ro, lifter_mat]:
        try:
            assert overwrite is False
            fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
            df = pd.read_pickle(fname)
            print(f"read {fname}")
        except (FileNotFoundError, AssertionError):
            fname = f"{RESULTS_WRITE}/{lifter}_{appendix}.pkl"
            add_redundant_constr = (
                True if isinstance(lifter, MatWeightLocLifter) else False
            )
            df = generate_results(
                lifter,
                n_params_list=n_params_list,
                fname=fname,
                use_methods=list(USE_METHODS.keys()),
                noise_list=[lifter.NOISE],
                add_redundant_constr=add_redundant_constr,
                n_threads_list=n_threads_list,
            )
            df.to_pickle(fname)
            print("saved final as", fname)

    def custom_plot(ax, x, y, data, **unused):
        data_methods = data["solver type"].unique()
        for method in USE_METHODS.keys():
            if method in data_methods:
                df_sub = data[data["solver type"] == method]
                ax.plot(df_sub[x], df_sub[y], **USE_METHODS[method])
            else:
                print(f"skipping {method}, not in {data_methods}")

    for lifter in [lifter_mat, lifter_ro]:
        fname = f"{RESULTS_READ}/{lifter}_{appendix}.pkl"
        df = pd.read_pickle(fname)
        for label, plot in zip(
            ["t", "cost", "RDG"], [custom_plot, sns.barplot, sns.barplot]
        ):
            value_vars = [f"{label} {m}" for m in USE_METHODS]
            value_vars = set(value_vars).intersection(df.columns.unique())
            df_long = df.melt(
                id_vars=["n params"],
                value_vars=value_vars,
                value_name=label,
                var_name="solver type",
            )
            df_long.loc[:, "solver type"] = [
                l.strip(f"{label} ") for l in df_long["solver type"]
            ]
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 3)
            plot(
                data=df_long,
                x="n params",
                y=label,
                ax=ax,
                hue="solver type",
                hue_order=USE_METHODS.keys(),
                palette={m: kwargs["color"] for m, kwargs in USE_METHODS.items()},
            )
            for group, kwargs in zip(ax.containers, USE_METHODS.values()):
                for bar in group:
                    bar.set_alpha(kwargs["alpha"])
            ax.set_yscale("log")
            if label not in ["cost", "RDG"]:
                ax.set_xscale("log")
                ax.legend(loc="upper left")
                ax.set_ylabel("cost")
            else:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [USE_METHODS[l]["label"] for l in labels]
                new_handles = [h for l, h in zip(new_labels, handles) if l is not None]
                new_labels = [l for l in new_labels if l is not None]
                ax.legend(new_handles, new_labels)
                ax.set_ylabel("time [s]")
            ax.grid("on")
            ax.set_xlabel("number of poses")
            savefig(fig, fname.replace(".pkl", f"_{label}.png"))
    print("done")
