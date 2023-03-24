import os

import matplotlib.pylab as plt
import seaborn as sns


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
    ext = name.split(".")[-1]
    fig.savefig(name, bbox_inches="tight", pad_inches=0, dpi=200)
    if verbose:
        print(f"saved plot as {name}")


def plot_matrices(
    A_list, n_matrices=15, start_idx=0, colorbar=True, vmin=None, vmax=None, nticks=None
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
    add_colorbar(fig, ax, im, nticks=nticks)
    [ax.axis("off") for ax in axs[i:]]
    fig.suptitle(
        f"{start_idx}:{start_idx+num_plots} of {len(A_list)} constraints", y=0.9
    )
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
    plot_matrices(A_list, colorbar=False, vmin=vmin, vmax=vmax, nticks=4)
    plt.show()

    print("done")
