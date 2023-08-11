import numpy as np
import matplotlib.pylab as plt

from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from poly_matrix.poly_matrix import PolyMatrix

INCREMENTAL = True
NORMALIZE = True


def test_canonical_operations():
    n_landmarks = 1  # z_0 and z_1
    lifter = Stereo2DLifter(n_landmarks=n_landmarks, param_level="p", level="no")

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
    Ai_poly = PolyMatrix(symmetric=True)
    Ai_poly["x", "x"] = Ai_sub 
    Ai = Ai_poly.get_matrix(lifter.var_dict)

    # fmt: on
    ai = lifter.get_vec(Ai_sub)
    # zero-pad to emulate an augmented basis vector
    bi = lifter.augment_using_zero_padding(ai, var_subset=var_subset)

    ai_test = lifter.get_reduced_a(bi, var_subset=var_subset)

    np.testing.assert_allclose(ai, ai_test)

    # will return a 9 x 9 matrix with zero padding.
    Ai_test = lifter.get_mat(ai_test, var_dict={var_subset: 6})
    np.testing.assert_allclose(Ai.toarray(), Ai_test.toarray())


def test_learned_constraints(d=2, param_level="ppT"):
    n_landmarks = 2  # z_0 and z_1
    if d == 1:
        lifter = Stereo1DLifter(n_landmarks=n_landmarks, param_level=param_level)
    elif d == 2:
        lifter = Stereo2DLifter(
            n_landmarks=n_landmarks, param_level=param_level, level="urT"
        )
    else:
        raise ValueError(d)

    var_subset = ["l", "z_0", "z_1"]
    label_dict = lifter.var_dict_row(var_subset)

    basis_row_list = lifter.get_basis_list(var_subset, plot=False)
    basis_list = [
        basis_row.get_matrix((["l"], label_dict)) for basis_row in basis_row_list
    ]
    A_learned = lifter.generate_matrices(basis_list, var_dict=var_subset)

    # first, test that the learned constraints actually work on the original setup.
    np.random.seed(0)
    lifter.test_constraints(A_learned, errors="print")

    # then, with parameters, we can regenerate new learned variables for each new random setup.
    for i in range(0, 10):
        print(f"---- {i} -----")
        np.random.seed(i)
        lifter.generate_random_setup()
        A_learned = lifter.generate_matrices(basis_list, var_dict=var_subset)
        lifter.test_constraints(A_learned, errors="print")


def test_learned_constraints_augment(d=2, param_level="ppT"):
    n_landmarks = 2  # z_0 and z_1
    if d == 1:
        lifter = Stereo1DLifter(n_landmarks=n_landmarks, param_level="p")
    elif d == 2:
        lifter = Stereo2DLifter(
            n_landmarks=n_landmarks, param_level=param_level, level="urT"
        )
    else:
        raise ValueError(d)

    if INCREMENTAL:
        var_subset = ["l", "z_0", "z_1"]
        label_dict = lifter.var_dict_row(var_subset)

        basis_list = lifter.get_basis_list(var_subset, plot=False)
        basis_list_all = lifter.apply_templates(basis_list)
        basis_poly_all = PolyMatrix.init_from_row_list(basis_list_all)

        basis_learned = basis_poly_all.get_matrix(
            variables=(basis_poly_all.variable_dict_i, label_dict)
        )
        A_learned = lifter.generate_matrices(basis_list, var_dict=var_subset)
    else:
        var_subset = list(lifter.var_dict.keys())
        label_dict = lifter.var_dict_row(var_subset)

        A_learned, basis_poly = lifter.get_A_learned(normalize=NORMALIZE)
        basis_learned = basis_poly.get_matrix(
            variables=(basis_poly.variable_dict_i, label_dict)
        )

    # first, test that the learned constraints actually work on the original setup.
    np.random.seed(0)
    lifter.test_constraints(A_learned, errors="raise")

    # then, with parameters, we can regenerate new learned variables for each new random setup.
    for i in range(10):
        np.random.seed(i)
        lifter.generate_random_setup()
        A_learned = lifter.generate_matrices(basis_learned, var_dict=var_subset)

        lifter.test_constraints(A_learned, errors="print")


def test_b_to_a():
    n_landmarks = 1  # z_0 only

    lifter = Stereo2DLifter(n_landmarks=n_landmarks, param_level="p", level="no")

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
            x_sub_aug = lifter.augment_using_parameters(x_sub, var_subset=var_subset)

            assert len(x_sub_aug) == len(bi_sub)
            assert abs(bi_sub @ x_sub_aug) < 1e-10
            # x_sub_aug is of the form
            # [l, l.x, l.z, l.vech(xx'), l.vech(xz'), l.vech(zz'), p0, p0.l.x, ...]
            for j, p in enumerate(lifter.get_p(var_subset=var_subset)):
                assert abs(x_sub_aug[j * lifter.get_dim_X(var_subset)] - p) < 1e-10

            # test that ai @ x holds (summing out parameters)
            ai_sub = lifter.get_reduced_a(bi_sub, var_subset=var_subset)
            assert abs(ai_sub @ x_sub) < 1e-10

            # put back in all other variables.

            bi_all = lifter.zero_pad_subvector(
                bi_sub, var_subset, target_subset=lifter.var_dict
            )
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
                    raise

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
                raise

            try:
                np.testing.assert_allclose(Ai.toarray(), Ai_test)
            except:
                fig, axs = plt.subplots(1, 2)
                axs[0].matshow(Ai.toarray())
                axs[0].set_title("Ai")
                axs[1].matshow(Ai_test)
                axs[1].set_title("Ai test")
                raise

            assert len(bi_all) == lifter.get_dim_X() * lifter.get_dim_P()

            x = lifter.get_x()
            x_all = lifter.get_vec(np.outer(x, x))
            x_all_aug = lifter.augment_using_parameters(x_all)

            # test that x_all_aug @ bi_all holds (x includes parameters)
            assert abs(bi_all @ x_all_aug) < 1e-10

            ai_all = lifter.get_reduced_a(bi_all)
            # test that ai_all @ x_all holds.
            assert abs(ai_all @ x_all) < 1e-10


def test_zero_padding():
    n_landmarks = 1  # z_0 only
    lifter = Stereo2DLifter(n_landmarks=n_landmarks, param_level="p", level="no")

    for var_subset in [("l", "x"), ("l", "x", "z_0")]:
        var_dict = lifter.get_var_dict(var_subset)
        # get new patterns for this subset.
        Y = lifter.generate_Y(var_subset=var_subset)
        basis_new, S = lifter.get_basis(Y)
        for i, bi_sub in enumerate(basis_new[:10, :]):
            ai_sub = lifter.get_reduced_a(bi_sub, var_subset)
            bi_poly = lifter.convert_b_to_polyrow(bi_sub, var_subset)

            # enerate list of poly matrices from this pattern.
            new_patterns = lifter.apply_template(bi_poly)
            for new_pattern in new_patterns:
                # generate Ai from poly_row.
                ai_test = lifter.convert_polyrow_to_a(new_pattern, var_subset)
                Ai = lifter.get_mat(ai_test, var_dict=var_dict)
                # bi = new_pattern.get_matrix((["l"], row_var_dict))
                # Ai = lifter.get_mat(lifter.get_reduced_a(bi, lifter.var_dict))
                try:
                    lifter.test_constraints([Ai])
                except:
                    b_poly_test = lifter.convert_b_to_polyrow(bi_sub, var_subset)
                    print(b_poly_test)

                    Ai_sub = lifter.get_mat(ai_sub, var_dict=var_dict)
                    Ai_poly, __ = PolyMatrix.init_from_sparse(Ai_sub, var_dict)
                    Ai_poly.matshow(var_dict)

                    fig, ax = plt.subplots()
                    ax.matshow(Ai.toarray())
                    plt.show()
                    raise


if __name__ == "__main__":
    test_zero_padding()
    test_b_to_a()
    test_learned_constraints_augment(d=1, param_level="p")
    test_learned_constraints_augment(d=2, param_level="ppT")
    test_learned_constraints(d=2, param_level="ppT")
    test_canonical_operations()
    print("all tests passed")
