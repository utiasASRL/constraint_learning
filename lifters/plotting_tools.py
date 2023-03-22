import os

import matplotlib.pylab as plt


def add_colorbar(fig, ax, im, title=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    if title is not None:
        cax.set_ylabel(title)
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


def plot_matrices(A_list, n_matrices=15, start_idx=0, colorbar=True):
    num_plots = min(len(A_list), n_matrices)

    fig, axs = plt.subplots(1, num_plots, squeeze=False)
    axs = axs.flatten()
    fig.set_size_inches(num_plots, 2)
    for i, (ax, Ai) in enumerate(zip(axs, A_list[start_idx:])):
        im = ax.matshow(Ai)  # , vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"$A_{{{i+1}}}$", y=1.0)
        if colorbar:
            add_colorbar(fig, ax, im)
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
