import numpy as np
import matplotlib.pylab as plt

from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter

INCREMENTAL = True
NORMALIZE = True


def test_canonical_operations():
    n_landmarks = 1  # z_0 and z_1
    lifter = Stereo2DLifter(n_landmarks=n_landmarks, add_parameters=True, level="no")

    var_subset = "x"
    # fmt: off
    Ai_sub = np.array(
        [
            [1,  0,  0,  0,  0, 0],
            [0,  0, .5,  0,  0, 0],
            [0, .5,  0,  0, .5, 0],
            [0,  0,  0,  1,  0, 0],
            [0,  0, .5,  0,  0, 0],
            [0,  0,  0,  0,  0, 0],
        ]
    )
    from poly_matrix.poly_matrix import PolyMatrix
    Ai_poly = PolyMatrix(symmetric=True)
    Ai_poly["x", "x"] = Ai_sub 
    Ai = Ai_poly.get_matrix(lifter.var_dict)

    # fmt: on
    ai = lifter.get_vec(Ai_sub)
    # zero-pad to emulate an augmented basis vector
    bi = lifter.augment_using_zero_padding(ai)

    ai_test = lifter.get_reduced_a(bi, var_subset=var_subset)

    np.testing.assert_allclose(ai, ai_test)

    # will return a 9 x 9 matrix with zero padding.
    Ai_test = lifter.get_mat(ai_test, var_dict={var_subset: 6})
    np.testing.assert_allclose(Ai.toarray(), Ai_test.toarray())


def test_with_parameters(d=1):
    n_landmarks = 2  # z_0 and z_1
    if d == 1:
        lifter = Stereo1DLifter(n_landmarks=n_landmarks, add_parameters=True)
    elif d == 2:
        lifter = Stereo2DLifter(n_landmarks=n_landmarks, add_parameters=True)
    else:
        raise ValueError(d)

    if INCREMENTAL:
        basis_list = lifter.get_basis_list_incremental()
        from utils.plotting_tools import plot_basis
        from poly_matrix.poly_matrix import PolyMatrix

        basis_poly = PolyMatrix.init_from_row_list(basis_list)
        plot_basis(basis_poly, lifter)
        # wtf, if I remeove this then basis_small is None. If I leave it, it is defined.
        # this has to do with plt.ion(), but I don't know why.

        basis_list_all = lifter.augment_basis_list(basis_list, normalize=NORMALIZE)
        basis_poly_all = PolyMatrix.init_from_row_list(basis_list_all)
        label_dict = {l: 1 for l in lifter.get_label_list()}
        basis_learned = basis_poly_all.get_matrix(
            variables=(basis_poly_all.variable_dict_i, label_dict)
        )
        A_learned = lifter.generate_matrices(basis_learned)
    else:
        A_learned, basis_poly = lifter.get_A_learned(normalize=NORMALIZE)
        basis_learned = basis_poly.get_matrix(
            variables=(basis_poly.variable_dict_i, label_list)
        )

    # first, test that the learned constraints actually work on the original setup.
    lifter.test_constraints(A_learned, errors="raise")

    # then, with parameters, we can regenerate new learned variables for each new random setup.
    for i in range(10):
        np.random.seed(i)
        lifter.generate_random_setup()
        A_learned = lifter.generate_matrices(basis_learned)

        lifter.test_constraints(A_learned, errors="raise")


def test_learning():
    add_parameters = True
    n_landmarks = 1  # z_0 only

    lifter = Stereo2DLifter(
        n_landmarks=n_landmarks, add_parameters=add_parameters, level="no"
    )

    for var_subset in [("l", "x"), ("l", "x", "z_0")]:
        Y = lifter.generate_Y(var_subset=var_subset)
        basis_new, S = lifter.get_basis(Y)
        for i, bi_sub in enumerate(basis_new[:10, :]):
            var_dict = {k: v for k, v in lifter.var_dict.items() if k in var_subset}
            bi_sub[np.abs(bi_sub) < 1e-10] = 0.0

            # generate variable vector of this subset.
            x = lifter.get_x(var_subset=var_subset)
            x_sub = lifter.get_vec(np.outer(x, x))

            # test that bi_sub @ x_aug holds (including parameters in x_aug)
            x_sub_aug = lifter.augment_using_parameters(x_sub)
            # x_sub_aug is of the form
            # [l, l.x, l.z, l.vech(xx'), l.vech(xz'), l.vech(zz'), p0, p0.l.x, ...]
            for j, p in enumerate(lifter.get_parameters()):
                assert abs(x_sub_aug[j * lifter.get_dim_X(var_subset)] - p) < 1e-10

            assert len(x_sub_aug) == len(bi_sub)
            assert abs(bi_sub @ x_sub_aug) < 1e-10

            # test that ai @ x holds (summing out parameters)
            ai_sub = lifter.get_reduced_a(bi_sub, var_subset=var_subset)
            assert abs(ai_sub @ x_sub) < 1e-10

            # put back in all other variables.
            bi_all, bi_poly = lifter.zero_pad_subvector(bi_sub, var_subset)
            # bi_all = lifter.get_vector_dense(bi_poly)

            if var_subset == tuple(lifter.var_dict.keys()):
                try:
                    np.testing.assert_allclose(bi_all, bi_sub)
                except:
                    fig, axs = plt.subplots(2, 1)
                    axs[0].matshow(bi_sub[None, :])
                    axs[0].set_title("bi sub")
                    axs[1].matshow(bi_all[None, :])
                    axs[1].set_title("bi all")

            # generate the full matrix, placing the subvar in the correct place.
            Ai = lifter.get_mat(ai_sub, var_dict=var_dict)
            ai_all_test = lifter.get_vec(Ai)

            # generate the full matrix directly from the test vector
            ai_all = lifter.get_reduced_a(bi_all)
            Ai_test = lifter.get_mat(ai_all)
            try:
                np.testing.assert_allclose(ai_all_test, ai_all)
            except:
                fig, axs = plt.subplots(2, 1)
                axs[0].matshow(ai_all[None, :])
                axs[0].set_title("ai all")
                axs[1].matshow(ai_all_test[None, :])
                axs[1].set_title("ai all test")

            try:
                np.testing.assert_allclose(Ai.toarray(), Ai_test)
            except:
                fig, axs = plt.subplots(1, 2)
                axs[0].matshow(Ai.toarray())
                axs[0].set_title("Ai")
                axs[1].matshow(Ai_test)
                axs[1].set_title("Ai test")

            assert len(bi_all) == lifter.get_dim_X() * lifter.get_dim_P()

            x = lifter.get_x()
            x_all = lifter.get_vec(np.outer(x, x))
            x_all_aug = lifter.augment_using_parameters(x_all)

            # test that x_all_aug @ bi_all holds (x includes parameters)
            assert abs(bi_all @ x_all_aug) < 1e-10

            ai_all = lifter.get_reduced_a(bi_all)
            # test that ai_all @ x_all holds.
            assert abs(ai_all @ x_all) < 1e-10


if __name__ == "__main__":
    test_learning()
    test_with_parameters(d=1)
    test_canonical_operations()
    print("all tests passed")
