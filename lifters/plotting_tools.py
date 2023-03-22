import matplotlib.pylab as plt


def add_colorbar(fig, ax, im, title=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    if title is not None:
        cax.set_ylabel(title)
    return cax


def savefig(fig, name, verbose=True):
    make_dirs_safe(name)
    ext = name.split(".")[-1]
    fig.savefig(name, bbox_inches="tight", pad_inches=0, dpi=200)
    if verbose:
        print(f"saved plot as {name}")


def plot_matrices(A_list, colorbar=True):
    num_plots = min(len(A_list), 15)

    fig, axs = plt.subplots(1, num_plots, squeeze=False)
    axs = axs.flatten()
    fig.set_size_inches(num_plots, 2)
    for i, (ax, Ai) in enumerate(zip(axs, A_list)):
        im = ax.matshow(Ai)  # , vmin=vmin, vmax=vmax)
        ax.axis("off")
        ax.set_title(f"$A_{{{i+1}}}$", y=1.0)
        if colorbar:
            add_colorbar(fig, ax, im)
    fig.suptitle(f"{num_plots} of {len(A_list)} constraints", y=0.9)
    return fig, axs


def plot_singular_values(S):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2)
    ax.semilogy(S, marker="o")
    ax.grid()
    return fig, ax
