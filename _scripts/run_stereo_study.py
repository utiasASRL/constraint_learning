import matplotlib.pylab as plt
import numpy as np

from auto_template.learner import Learner
from auto_template.sim_experiments import (
    apply_autotemplate_base,
    apply_autotemplate_plot,
    apply_autotight_base,
    plot_autotemplate_time,
)
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter
from utils.plotting_tools import add_lines, plot_matrix, savefig

RESULTS_DIR = "_results_v4"
debug = False


def apply_autotight(d=2, n_landmarks=None, results_dir=RESULTS_DIR):
    """
    Find the set of minimal constraints required for tightness for stereo problem.
    """
    if n_landmarks is None:
        n_landmarks = d + 1
    seed = 0

    # parameter_levels = ["ppT"] #["no", "p", "ppT"]
    levels = ["no", "urT"]
    param_level = "no"
    for level in levels:
        print(f"============= seed {seed} level {level} ================")
        np.random.seed(seed)

        variable_list = [["h", "x"] + [f"z_{i}" for i in range(n_landmarks)]]
        plots = ["matrix", "templates", "svd"]
        lifter = Stereo2DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )

        learner = Learner(
            lifter=lifter, variable_list=lifter.variable_list, apply_templates=False
        )
        fname_root = f"{results_dir}/{lifter}_seed{seed}"

        apply_autotight_base(learner, fname_root, plots)


def apply_autotemplate(n_seeds, recompute, d=2, results_dir=RESULTS_DIR, debug=debug):
    n_landmarks_list = [15, 20, 25, 30] if not debug else [15, 16]
    use_orders = ["sorted", "basic"] if not debug else ["sorted"]
    compute_oneshot = False if debug else True

    level = "urT"
    param_level = "ppT"

    n_landmarks = d + 1

    # variable_list = [["h", "x"] + [f"z_{i}" for i in range(n_landmarks)]] runs out of memory for d=3
    variable_list = None  # use the default one for the first step.
    np.random.seed(0)
    if d == 2:
        lifter = Stereo2DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
    elif d == 3:
        lifter = Stereo3DLifter(
            n_landmarks=n_landmarks,
            level=level,
            param_level=param_level,
            variable_list=variable_list,
        )
        # can be used for reproducibility
        # fname = "_results/stereo3d_lifter.pkl"
        # try:
        #     lifter = Stereo3DLifter.from_file(fname)
        # except FileNotFoundError:
        #     lifter = Stereo3DLifter(
        #         n_landmarks=n_landmarks,
        #         level=level,
        #         param_level=param_level,
        #         variable_list=variable_list,
        #     )
        #     lifter.to_file(fname)
        #     lifter = Stereo3DLifter.from_file(fname)

    learner = Learner(lifter=lifter, variable_list=lifter.variable_list)

    if lifter.d == 2 and not debug:
        fname_root = f"{results_dir}/autotemplate_{learner.lifter}"
        learner = Learner(lifter=lifter, variable_list=lifter.variable_list)
        apply_autotemplate_plot(learner, recompute=recompute, fname_root=fname_root)
        return

    df = apply_autotemplate_base(
        learner,
        param_list=n_landmarks_list,
        n_seeds=n_seeds,
        recompute=recompute,
        results_folder=results_dir,
        use_orders=use_orders,
        compute_oneshot=compute_oneshot,
    )
    if df is None:
        return

    fname_root = f"{results_dir}/autotemplate_{learner.lifter}"

    # this entry is invalid (it is the copy of N=25)
    df = df.loc[~((df.N == 30) & (df["type"] == "basic"))]

    fig, axs = plot_autotemplate_time(df, log=True, start="t ", legend_idx=1)
    # [ax.set_ylim(10, 1000) for ax in axs.values()]
    if not debug:
        [ax.set_ylim(2, 8000) for ax in axs]
        axs[0].set_xticks(df.N.unique(), [f"{x:.0f}" for x in df.N.unique()])

    add_lines(axs[0], df.N.unique(), start=df["t create constraints"].min(), facs=[3])
    add_lines(axs[1], df.N.unique(), start=df["t solve SDP"].min(), facs=[3])
    savefig(fig, fname_root + f"_t.pdf")


def run_all(
    n_seeds,
    recompute,
    autotight=True,
    autotemplate=True,
    results_dir=RESULTS_DIR,
    debug=debug,
):
    if autotight:
        print("========== Stereo2D autotight ===========")
        apply_autotight(d=2, results_dir=results_dir)
    if autotemplate:
        print("========== Stereo2D autotemplate ===========")
        apply_autotemplate(
            d=2,
            n_seeds=n_seeds,
            recompute=recompute,
            results_dir=results_dir,
            debug=debug,
        )
        if not debug:
            print("========== Stereo3D autotemplate ===========")
            apply_autotemplate(
                d=3, n_seeds=n_seeds, recompute=recompute, results_dir=results_dir
            )


def run_stereo_1d():
    from cert_tools.linalg_tools import rank_project
    from cert_tools.sdp_solvers import solve_sdp_cvxpy

    np.random.seed(0)
    lifter = Stereo1DLifter(n_landmarks=2)
    # lifter.theta = np.array([3.0])
    # lifter.landmarks = np.array([2.3, 4.6])

    print("theta:", lifter.theta, "landmarks:", lifter.landmarks)
    print("x:", lifter.get_x())
    # AutoTight

    A_known_all = lifter.get_A_known(add_known_redundant=True)
    A_known = lifter.get_A_known()
    # shortcut: A_learned = lifter.get_A_learned_simple(A_known=A_known)

    Y = lifter.generate_Y(factor=1.0)

    # with A_known
    basis, S = lifter.get_basis(Y, A_known=A_known)
    print("with known:", S)
    A_red = lifter.generate_matrices_simple(basis=basis)
    fig, axs = plt.subplots(1, len(A_red) + len(A_known), sharey=True)
    for i, Ai in enumerate(A_known):
        title = f"$A_{{k,{i}}}$"
        fig, ax, im = plot_matrix(Ai, ax=axs[i], colorbar=False)
        ax.set_title(title)
        print(f"known {i}", Ai.toarray())
    for j, Ai in enumerate(A_red):
        title = f"$A_{{\ell,{j}}}$"
        fig, ax, im = plot_matrix(Ai, ax=axs[i + 1 + j], colorbar=False)
        ax.set_title(title)
        print(f"learned {j}", Ai.toarray())

    # without A_known:
    basis, S = lifter.get_basis(Y, A_known=[])
    print("without known:", S)
    A_all = lifter.generate_matrices_simple(basis=basis)
    fig_raw, axs_raw = plt.subplots(1, len(A_all), sharey=True)
    for i, Ai in enumerate(A_all):
        title = f"$A_{{\ell,{i}}}$"
        plot_matrix(Ai, ax=axs_raw[i], colorbar=False)
        axs_raw[i].set_title(title)

    # hard coded for better comparability
    fig, axs = plt.subplots(3, len(A_all), sharey=True)
    plot_matrix(A_all[0] * np.sqrt(2), ax=axs[0, 0], colorbar=False)
    plot_matrix(A_all[1] * np.sqrt(2), ax=axs[0, 1], colorbar=False)
    plot_matrix(-A_all[2], ax=axs[0, 2], colorbar=False)
    axs[0, 0].set_title(f"$A_{{\ell,{1}}}$")
    axs[0, 1].set_title(f"$A_{{\ell,{2}}}$")
    axs[0, 2].set_title(f"$A_{{\ell,{3}}}$")
    print(np.round(A_all[0].toarray() * np.sqrt(2), 4))
    print(np.round(A_all[1].toarray() * np.sqrt(2), 4))
    print(np.round(-A_all[2].toarray(), 4))

    plot_matrix(A_known[0], ax=axs[1, 0], colorbar=False)
    plot_matrix(A_known[1], ax=axs[1, 1], colorbar=False)
    plot_matrix(A_red[0] * np.sqrt(2) / 2, ax=axs[1, 2], colorbar=False)
    axs[1, 0].set_title(f"$A_{{k,{1}}}$")
    axs[1, 1].set_title(f"$A_{{k,{2}}}$")
    axs[1, 2].set_title(f"$A_{{\ell,{1}}}$")

    plot_matrix(A_known_all[0], ax=axs[2, 0], colorbar=False)
    plot_matrix(A_known_all[1], ax=axs[2, 1], colorbar=False)
    plot_matrix(A_known_all[2], ax=axs[2, 2], colorbar=False)
    axs[2, 0].set_title(f"$A_{{k,{1}}}$")
    axs[2, 1].set_title(f"$A_{{k,{2}}}$")
    axs[2, 2].set_title(f"$A_{{k,{3}}}$")

    Q, y = lifter.get_Q()
    fig, ax, im = plot_matrix(Q)
    x = lifter.get_x()

    print("theta gt:", lifter.theta)

    theta_hat, info_local, cost_local = lifter.local_solver(t_init=lifter.theta, y=y)
    print("theta global:", theta_hat, "cost:", cost_local)

    # sanity check
    x_hat = lifter.get_x(theta=theta_hat)
    assert abs(x_hat.T @ Q @ x_hat - cost_local) / cost_local < 1e-10
    for A_list, label in zip([A_known, A_all], ["known", "learned"]):
        lifter.test_constraints(A_list)
        Constraints = [(lifter.get_A0(), 1.0)] + [(Ai, 0.0) for Ai in A_list]
        X, info_sdp = solve_sdp_cvxpy(Q=Q, Constraints=Constraints, verbose=False)
        x_round, info_rank = rank_project(X, 1)
        error = abs(X[0, 1] - theta_hat) / theta_hat
        RDG = abs(cost_local - info_sdp["cost"]) / cost_local
        print(
            f"{label}, theta sdp:{X[0,1]}, theta rounded:{x_round[1]} EVR:{info_rank['EVR']}, cost:{info_sdp['cost']}"
        )
        print("eigvals:", np.linalg.eigvalsh(X))
        print("RDG:", RDG)
        print("error", error)


if __name__ == "__main__":
    print("========== Stereo3D autotemplate ===========")

    run_stereo_1d()
    run_all(n_seeds=1, autotight=False, autotemplate=True, recompute=True)
