import matplotlib.pylab as plt
import numpy as np

from lifters.plotting_tools import savefig, add_colorbar
from poly_matrix.poly_matrix import PolyMatrix


def plot_basis(basis_poly: PolyMatrix, lifter, fname_root):
    all_product_dict = lifter.get_augmented_dict()
    variables_j = all_product_dict
    variables_i = basis_poly.generate_variable_dict_i()

    import matplotlib as mpl

    cmap = mpl.colors.ListedColormap(
        [
            [0.0, 0.4, 1.0],
            [0.0, 0.8, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.8, 0.0],
            [1.0, 0.4, 0.0],
        ]
    )
    cmap.set_over((1.0, 0.0, 0.0))
    cmap.set_under((0.0, 0.0, 1.0))
    bounds = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig, ax, im = basis_poly.matshow(
        variables_i=variables_i, variables_j=variables_j, cmap=cmap, norm=norm
    )
    cax = add_colorbar(fig, ax, im)
    cax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    scale = 0.2
    fig.set_size_inches(len(variables_j) * scale, len(variables_i) * scale)
    for p in range(1, lifter.get_dim_P()):
        ax.axvline(p * lifter.get_dim_X() - 0.5, color="red")
    if fname_root != "":
        savefig(fig, fname_root + f"_basis.png")


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
