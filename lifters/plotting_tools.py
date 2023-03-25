import os

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


def get_dirname():
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_plots"))
    return dirname


def partial_plot_and_save(lifter, Q, A_list, save=True):
    if save:
        dirname = get_dirname()

    # plot resulting matrices
    from lifters.plotting_tools import plot_matrices
    from math import ceil

    n = 10
    chunks = min(ceil(len(A_list) / n), 10)
    vmin = np.min([np.min(A) for A in A_list])
    vmax = np.max([np.max(A) for A in A_list])
    plot_count = 0
    for k in np.arange(chunks):
        fig, axs = plot_matrices(
            A_list,
            n_matrices=n,
            start_idx=k * n,
            colorbar=False,
            nticks=3,
            vmin=vmin,
            vmax=vmax,
            lines=[1 + lifter.N, 1 + lifter.N + lifter.M],
        )
        if k in [0, 1, chunks - 2, chunks - 1]:
            if save:
                savefig(fig, f"{dirname}/A{plot_count}_{lifter}.png")
            plot_count += 1

    if Q is not None:
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 3)
        ax.matshow(np.abs(Q) > 1e-10)
        ax.set_title("Q mask")
        if save:
            savefig(fig, f"{dirname}/Q_{lifter}.png")


def add_colorbar(fig, ax, im, title=None, nticks=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    if title is not None:
        cax.set_ylabel(title)

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


def plot_matrices(
    A_list,
    n_matrices=15,
    start_idx=0,
    colorbar=True,
    vmin=None,
    vmax=None,
    nticks=None,
    lines=[],
):
    import matplotlib

    norm = matplotlib.colors.SymLogNorm(10**-5, vmin=vmin, vmax=vmax)
    num_plots = min(len(A_list), n_matrices)

    fig, axs = plt.subplots(1, num_plots, squeeze=False)
    axs = axs.flatten()
    fig.set_size_inches(num_plots, 2)
    for i, (ax, Ai) in enumerate(zip(axs, A_list[start_idx:])):
        im = ax.matshow(Ai, norm=norm)
        ax.axis("off")
        ax.set_title(f"$A_{{{i+1}}}$", y=1.0)
        if colorbar:
            add_colorbar(fig, ax, im)

    try:
        for n in lines:
            for ax in axs[: i + 1]:
                ax.plot([-0.5, n - 0.5], [n - 0.5, n - 0.5], color="red")
                ax.plot([n - 0.5, n - 0.5], [-0.5, n - 0.5], color="red")

        add_colorbar(fig, ax, im, nticks=nticks)
        [ax.axis("off") for ax in axs[i:]]
        fig.suptitle(
            f"{start_idx+1}:{start_idx+i+1} of {len(A_list)} constraints", y=0.9
        )
    except:
        pass
    return fig, axs


def plot_singular_values(S, eps=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2)
    ax.semilogy(S, marker="o", label="singular values")
    if eps is not None:
        ax.axhline(eps, color="C1", label="threshold")
    ax.grid()
    ax.legend(loc="upper right")
    return fig, ax


def plot_tightness(df, ax=None):
    palette = "viridis"
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    try:
        sns.lineplot(
            data=df[df.shuffle == 0],
            x="n",
            y="local cost",
            style="seed",
            hue="shuffle",
            ax=ax,
            palette=palette,
            legend=False,
        )
        sns.lineplot(
            data=df,
            x="n",
            y="dual cost",
            style="seed",
            hue="shuffle",
            ax=ax,
            palette=palette,
            markers="o",
            markeredgecolor=None,
        )
    except KeyError as e:
        print("Error, no valid data for dual cost?")
    except AttributeError as e:
        print("Error, empty dataframe?")
    ax.set_ylabel("cost")
    ax.set_yscale("log")
    return fig, ax


if __name__ == "__main__":
    import numpy as np

    vmin = -1
    vmax = 1
    A_list = np.random.rand(4, 4, 3)
    fig, axs = plot_matrices(A_list, colorbar=False, vmin=vmin, vmax=vmax, nticks=4)

    dim_X = 2
    for ax in axs:
        ax.plot([-0.5, dim_X + 0.5], [dim_X + 0.5, dim_X + 0.5], color="red")
        ax.plot([dim_X + 0.5, dim_X + 0.5], [-0.5, dim_X + 0.5], color="red")
    plt.show()

    print("done")
