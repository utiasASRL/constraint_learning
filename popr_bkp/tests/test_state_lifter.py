import numpy as np
import pytest
from popr.examples import Stereo2DLifter, Stereo3DLifter
from popr.lifters import StereoLifter
from popr.utils.common import get_vec, ravel_multi_index_triu, unravel_multi_index_triu
from popr.utils.test_tools import _test_with_tol, all_lifters


def pytest_configure():
    # global variables
    pytest.A_learned = {}
    for lifter in all_lifters():
        pytest.A_learned[str(lifter)] = None


def test_ravel():
    shape = (5, 5)
    # test diagonal elements
    for i in range(shape[0]):
        idx = np.array([i])
        flat_idx = ravel_multi_index_triu([idx, idx], shape=shape)
        i_test, j_test = unravel_multi_index_triu(flat_idx, shape=shape)

        assert idx == i_test[0]
        assert idx == j_test[0]

    # test random elements
    for seed in range(100):
        np.random.seed(seed)
        i = np.random.randint(low=0, high=shape[0] - 1, size=1)
        j = np.random.randint(low=i, high=shape[0] - 1, size=1)

        flat_idx = ravel_multi_index_triu([i, j], shape=shape)
        i_test, j_test = unravel_multi_index_triu(flat_idx, shape=shape)

        assert i == i_test[0]
        assert j == j_test[0]


def test_known_constraints():
    for lifter in all_lifters():
        A_known = lifter.get_A_known()
        _test_with_tol(lifter, A_known, tol=1e-10)

        B_known = lifter.get_B_known()
        x = lifter.get_x(theta=lifter.theta)
        for Bi in B_known:
            assert x.T @ Bi @ x <= 0


def test_vec_mat():
    """Make sure that we can go back and forth from vec to mat."""
    for lifter in all_lifters():
        try:
            A_known = lifter.get_A_known()
        except AttributeError:
            print(f"could not get A_known of {lifter}")
            A_known = []

        for A in A_known:
            a_dense = get_vec(A.toarray())
            a_sparse = get_vec(A)
            np.testing.assert_allclose(a_dense, a_sparse)

            # get_vec multiplies off-diagonal elements by sqrt(2)
            a = get_vec(A)

            A_test = lifter.get_mat(a, sparse=False)
            np.testing.assert_allclose(A.toarray(), A_test)

            # get_mat divides off-diagonal elements by sqrt(2)
            A_test = lifter.get_mat(a, sparse=True)
            np.testing.assert_allclose(A.toarray(), A_test.toarray())

            a_poly = lifter.convert_a_to_polyrow(a)
            a_test = lifter.convert_polyrow_to_a(a_poly)
            np.testing.assert_allclose(a, a_test)


def test_levels():
    for level in StereoLifter.LEVELS:
        lifter_2d = Stereo2DLifter(n_landmarks=3, level=level)

        # inside below function we tests that dimensions are consistent.
        lifter_2d.get_x()

        lifter_3d = Stereo3DLifter(n_landmarks=3, level=level)
        lifter_3d.get_x()


pytest_configure()

if __name__ == "__main__":
    import warnings

    test_ravel()
    test_vec_mat()

    # import pytest
    # print("testing")
    # pytest.main([__file__, "-s"])
    # print("all tests passed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # warnings.simplefilter("error")
        test_known_constraints()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    print("all tests passed")
    print("all tests passed")
    print("all tests passed")
    print("all tests passed")
