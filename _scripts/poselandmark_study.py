from lifters.landmark_lifter import PoseLandmarkLifter

if __name__ == "__main__":
    import itertools

    import pandas as pd
    from progressbar import ProgressBar

    # fname = f"../_results/redundant_test.pkl"
    fname = ""

    d_list = [2]
    N_list = [1, 2, 3]
    K_list = range(1, 7)

    plot = True

    data = []
    n = len(d_list) * len(N_list) * len(K_list)
    p = ProgressBar(max_value=n)
    i = 0
    for d, N, K in itertools.product(d_list, N_list, K_list):
        p.update(i)
        i += 1

        lifter = PoseLandmarkLifter(n_landmarks=K, n_poses=N, d=d)

        Y = lifter.generate_Y()
        basis, S = lifter.get_basis(Y)
        A_list = lifter.generate_matrices(basis)

        if plot:
            from lifters.plotting_tools import *

            plot_singular_values(S)
            plot_matrices(A_list, colorbar=False)
            plt.show()
            break

        n_rot = d**2 * N
        n_sub = N * K * d
        n_red1a = N * K
        n_red1b = int(N * (N - 1) * K)
        n_red2a = int(N * K * (K - 1))
        n_red2b = int(N * (N - 1) * K * (K - 1))
        n_known = n_rot + n_sub + n_red1a + n_red1b + n_red2a + n_red2b
        n_found = len(A_list)
        data.append(
            dict(
                d=2,
                n_poses=N,
                n_landmarks=K,
                n_known=n_known,
                n_missing=n_found - n_known,
                n_found=n_found,
            )
        )

    df = pd.DataFrame(data)
    if fname != "":
        df.to_pickle(fname)
        print(f"saved as {fname}")
    print("done")
