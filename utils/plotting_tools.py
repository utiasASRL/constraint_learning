from lifters.plotting_tools import import_plt
import numpy as np

from lifters.plotting_tools import savefig, add_colorbar
from poly_matrix.poly_matrix import PolyMatrix

plt = import_plt()


def plot_frame(
    lifter,
    ax,
    theta=None,
    xtheta=None,
    color="k",
    marker="+",
    label=None,
    scale=1.0,
    **kwargs,
):
    p_gt = lifter.get_position(theta=theta, xtheta=xtheta)
    try:
        C_cw = lifter.get_C_cw(theta=theta, xtheta=xtheta)
        for i, col in enumerate(["r", "g", "b"]):
            z_gt = C_cw[i, :]
            length = scale / np.linalg.norm(z_gt)
            ax.plot(
                [p_gt[0, 0], p_gt[0, 0] + length * z_gt[0]],
                [p_gt[0, 1], p_gt[0, 1] + length * z_gt[1]],
                color=col,
                ls="--",
                alpha=0.5,
            )
    except Exception as e:
        pass
    ax.scatter(*p_gt[:, :2].T, color=color, marker=marker, label=label, **kwargs)


def add_rectangles(ax, dict_sizes, color="red"):
    from matplotlib.patches import Rectangle

    cumsize = 0
    xticks = []
    xticklabels = []
    for key, size in dict_sizes.items():
        cumsize += size
        xticks.append(cumsize - 0.5)
        xticklabels.append(f"${key}$")
        ax.add_patch(Rectangle((-0.5, -0.5), cumsize, cumsize, ec=color, fc="none"))
        # ax.annotate(text=key, xy=(cumsize, 1), color='red', weight="bold")
    ax.set_xticks(xticks, xticklabels)
    ax.tick_params(axis="x", colors="red")
    ax.xaxis.tick_top()
    ax.set_yticks([])


def initialize_discrete_cbar(values):
    import matplotlib as mpl

    values = sorted(list(np.unique(values.round(3))) + [0])
    cmap = plt.get_cmap("viridis", len(values))
    cmap.set_over((1.0, 0.0, 0.0))
    cmap.set_under((0.0, 0.0, 1.0))
    bounds = [values[0] - 0.005] + [v + 0.005 for v in values]
    colorbar_yticks = [""] + list(values)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, colorbar_yticks


def plot_basis(
    basis_poly: PolyMatrix,
    variables_j: dict,
    variables_i: list = None,
    fname_root: str = "",
    discrete: bool = True,
):
    if variables_i is None:
        variables_i = basis_poly.generate_variable_dict_i()

    if discrete:
        values = basis_poly.get_matrix((variables_i, variables_j)).data
        cmap, norm, colorbar_yticks = initialize_discrete_cbar(values)
    else:
        cmap = plt.get_cmap("viridis")
        norm = None
        colorbar_yticks = None

    # reduced_ticks below has no effect because all variables in variables_j are of size 1.
    fig, ax, im = basis_poly.matshow(
        variables_i=variables_i,
        variables_j=variables_j,
        cmap=cmap,
        norm=norm,  # reduced_ticks=True
    )
    fig.set_size_inches(15, 15 * len(variables_i) / len(variables_j))
    cax = add_colorbar(fig, ax, im)
    if colorbar_yticks is not None:
        cax.set_yticklabels(colorbar_yticks)
    if fname_root != "":
        savefig(fig, fname_root + f"_basis.png")
    return fig, ax


def plot_tightness(df_tight, qcqp_cost, fname_root):
    fig, ax = plt.subplots()
    ax.axhline(qcqp_cost, color="k")
    for (order_name, order_type), df_sub in df_tight.groupby(
        ["order_name", "order_type"]
    ):
        try:
            number = np.where(df_sub.tight.values)[0][0]
            label = f"{order_name} {order_type}: {number}"
        except IndexError:
            number = None
            label = f"{order_name} {order_type}: never tight"
        ax.semilogy(range(1, len(df_sub) + 1), np.abs(df_sub.cost.values), label=label)
    ax.legend()
    ax.grid()
    if fname_root != "":
        savefig(fig, fname_root + f"_tightness.png")


def plot_matrices(df_tight, fname_root):
    from lifters.plotting_tools import plot_matrix
    import itertools
    from math import ceil

    # for (order, order_type), df_sub in df_tight.groupby(["order", "type"]):
    matrix_types = ["A", "H"]
    for (order_name, order_type), df_sub in df_tight.groupby(
        ["order_name", "order_type"]
    ):
        A_agg = None
        H = None

        n_cols = 10
        n_rows = min(ceil(len(df_sub) / n_cols), 5)  # plot maximum 5 * 2 rows
        fig, axs = plt.subplots(n_rows * 2, n_cols)
        fig.set_size_inches(n_cols, n_rows * 2)
        for j, matrix_type in enumerate(matrix_types):
            matrices = df_sub[matrix_type].values
            names_here = df_sub.name.values
            costs = df_sub.cost.values

            # make sure it's symmetric
            vmin = np.min([np.min(A) for A in matrices if (A is not None)])
            vmax = np.max([np.max(A) for A in matrices if (A is not None)])
            vmin = min(vmin, -vmax)
            vmax = max(vmax, -vmin)

            i = 0

            for row, col in itertools.product(range(n_rows), range(n_cols)):
                ax = axs[row * 2 + j, col]
                if i < len(matrices):
                    if matrices[i] is None:
                        continue
                    title = f"{matrix_type}{i}"
                    if matrix_type == "A":
                        title += f"\n{names_here[i]}"
                    else:
                        title += f"\nc={costs[i]:.2e}"

                    plot_matrix(
                        ax=ax,
                        Ai=matrices[i],
                        vmin=vmin,
                        vmax=vmax,
                        title=title,
                        colorbar=False,
                    )
                    ax.set_title(title, fontsize=5)

                    if matrix_type == "A":
                        if A_agg is None:
                            A_agg = np.abs(matrices[i].toarray()) > 1e-10
                        else:
                            A_agg = np.logical_or(
                                A_agg, (np.abs(matrices[i].toarray()) > 1e-10)
                            )
                    elif matrix_type == "H":
                        # make sure we store the last valid estimate of H, for plotting
                        if matrices[i] is not None:
                            H = matrices[i]
                else:
                    ax.axis("off")
                i += 1

        fname = fname_root + f"_{order_name}_{order_type}.png"
        savefig(fig, fname)

        return
        matrix_agg = {
            "A": A_agg.astype(int),
            "H": H,
            "Q": Q.toarray(),
        }
        for matrix_type, matrix in matrix_agg.items():
            if matrix is None:
                continue
            fname = fname_root + f"_{order_name}_{order_type}_{matrix_type}.png"
            fig, ax = plt.subplots()
            plot_matrix(
                ax=ax,
                Ai=matrix,
                colorbar=True,
                title=matrix_type,
            )
            savefig(fig, fname)
