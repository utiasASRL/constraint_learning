import os

import numpy as np

from poly_matrix.poly_matrix import PolyMatrix

USE_METHODS = {
    "local": {"color": "C0", "marker": "o", "alpha": 1.0, "label": "local"},
    "SDP": {"color": "C1", "marker": "o", "alpha": 1.0, "label": "SDP"},
    "SDP-redun": {"color": "C1", "marker": "d", "alpha": 0.5, "label": None},
    "dSDP": {"color": "C2", "marker": "o", "alpha": 1.0, "label": "dSDP"},
    "dSDP-redun": {"color": "C2", "marker": "d", "alpha": 0.5, "label": None},
    "pADMM": {"color": "C3", "marker": "o", "alpha": 1.0, "label": "altSDP"},
    "pADMM-redun": {"color": "C3", "marker": "d", "alpha": 0.5, "label": None},
}


def import_plt():
    import shutil

    import matplotlib.pylab as plt

    usetex = True if shutil.which("latex") else False
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )
    plt.rc("text.latex", preamble=r"\usepackage{bm}")
    return plt


plt = import_plt()


def add_colorbar(fig, ax, im, title=None, nticks=None, visible=True, size=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    if size is None:
        w, h = fig.get_size_inches()
        size = f"{5*h/w}%"
    cax = divider.append_axes("right", size=size, pad=0.05)
    if title is not None:
        cax.set_ylabel(title)

    if not visible:
        cax.axis("off")
        return

    fig.colorbar(im, cax=cax, orientation="vertical")

    # add symmetric nticks ticks: min and max, and equally spaced in between
    if nticks is not None:
        from math import floor

        ticks = cax.get_yticks()
        new_ticks = [ticks[0]]
        step = floor(len(ticks) / (nticks - 1))
        new_ticks += list(ticks[step + 1 :: step])
        new_ticks += [ticks[-1]]
        # print(f"reduce {ticks} to {new_ticks}")
        assert len(new_ticks) == nticks
        cax.set_yticks(ticks[::step])
    return cax


def add_scalebar(
    ax, size=5, size_vertical=1, loc="lower left", fontsize=8, color="black", pad=0.1
):
    """Add a scale bar to the plot.

    :param ax: axis to use.
    :param size: size of scale bar.
    :param size_vertical: height (thckness) of the bar
    :param loc: location (same syntax as for matplotlib legend)
    """
    import matplotlib.font_manager as fm
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(
        ax.transData,
        size,
        "{} m".format(size),
        loc,
        pad=pad,
        color=color,
        frameon=False,
        size_vertical=size_vertical,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)


def make_dirs_safe(path):
    """Make directory of input path, if it does not exist yet."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def savefig(fig, name, verbose=True):
    make_dirs_safe(name)
    fig.savefig(name, bbox_inches="tight", pad_inches=0, dpi=200)
    if verbose:
        print(f"saved plot as {name}")


def plot_frame(
    lifter,
    ax,
    theta=None,
    xtheta=None,
    color="k",
    marker="+",
    label=None,
    scale=1.0,
    ls="--",
    alpha=0.5,
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
                ls=ls,
                alpha=alpha,
                zorder=-1,
            )
    except Exception as e:
        pass
    ax.scatter(
        *p_gt[:, :2].T,
        color=color,
        marker=marker,
        label=label,
        zorder=1,
        **kwargs,
    )
    return


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


def plot_matrix(
    Ai,
    vmin=None,
    vmax=None,
    nticks=None,
    title="",
    colorbar=True,
    fig=None,
    ax=None,
    log=True,
    discrete=False,
):
    import matplotlib

    if ax is None:
        fig, ax = plt.subplots()
    if fig is None:
        fig = plt.gcf()

    norm = None
    if log:
        norm = matplotlib.colors.SymLogNorm(10**-5, vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("viridis")
    colorbar_yticks = None
    if discrete:
        values = np.unique(Ai[Ai != 0])
        nticks = None
        cmap, norm, colorbar_yticks = initialize_discrete_cbar(values)

    if type(Ai) is np.ndarray:
        im = ax.matshow(Ai, norm=norm, cmap=cmap)
    else:
        im = ax.matshow(Ai.toarray(), norm=norm, cmap=cmap)
    ax.axis("off")
    ax.set_title(title, y=1.0)
    if colorbar:
        cax = add_colorbar(fig, ax, im, nticks=nticks)
    else:
        cax = add_colorbar(fig, ax, im, nticks=nticks, visible=False)
    if colorbar_yticks is not None:
        cax.set_yticklabels(colorbar_yticks)
    return fig, ax, im


def plot_matrices(df_tight, fname_root):
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


def plot_singular_values(S, eps=None, label="singular values", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    fig.set_size_inches(4, 2)
    ax.semilogy(S, marker="o", label=label)
    if eps is not None:
        ax.axhline(eps, color="C1")
    ax.grid()
    ax.set_xlabel("index")
    ax.set_ylabel("abs. singular values")
    if label is not None:
        ax.legend(loc="upper right")
    return fig, ax


def plot_aggregate_sparsity(mask):
    fig, ax = plt.subplots()
    ax.matshow(mask.toarray())
    plt.show(block=False)
    return fig, ax


def matshow_list(*args, log=False, ticks=False, **kwargs):
    fig, axs = plt.subplots(1, len(args), squeeze=False)
    fig.set_size_inches(3 * len(args), 3)
    for i, arg in enumerate(args):
        try:
            if log:
                axs[0, i].matshow(np.log10(np.abs(arg)), **kwargs)
            else:
                axs[0, i].matshow(arg, **kwargs)
        except Exception as e:
            if log:
                axs[0, i].matshow(np.log10(np.abs(arg.toarray())), **kwargs)
            else:
                axs[0, i].matshow(arg.toarray(), **kwargs)

        if not ticks:
            axs[0, i].set_yticks([])
            axs[0, i].set_xticks([])
    return fig, axs
