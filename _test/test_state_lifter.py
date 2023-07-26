import matplotlib.pylab as plt
import numpy as np

from lifters.test_tools import all_lifters
from lifters.state_lifter import unravel_multi_index_triu, ravel_multi_index_triu

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
        i = np.random.randint(low=0, high=shape[0]-1, size=1)
        j = np.random.randint(low=i, high=shape[0]-1, size=1)

        flat_idx = ravel_multi_index_triu([i, j], shape=shape)
        i_test, j_test = unravel_multi_index_triu(flat_idx, shape=shape)

        assert i == i_test[0]
        assert j == j_test[0]


def test_constraints():
    def test_with_tol(A_list, tol):
        x = lifter.get_x()
        for Ai in A_list:
            err = abs(x.T @ Ai @ x)
            assert err < tol, err

            ai = lifter.get_vec(Ai.toarray())
            xvec = lifter.get_vec(np.outer(x, x))
            np.testing.assert_allclose(ai @ xvec, 0.0, atol=tol)

            ai = lifter.get_vec(Ai)
            xvec = lifter.get_vec(np.outer(x, x))
            np.testing.assert_allclose(ai @ xvec, 0.0, atol=tol)

    for lifter in all_lifters():
        A_known = lifter.get_A_known()
        test_with_tol(A_known, tol=1e-10)

        methods = ["qrp", "svd", "qr"]
        num_learned = None
        for method in methods:
            np.random.seed(0)
            A_learned = lifter.get_A_learned(method=method)
            test_with_tol(A_learned, tol=1e-4)

            # make sure each method finds the same number of matrices
            if num_learned is None:
                num_learned = len(A_learned)
            else:
                assert len(A_learned) == num_learned


def test_vec_mat():
    """Make sure that we can go back and forth from vec to mat."""
    for lifter in all_lifters():
        try:
            A_known = lifter.get_A_known()
        except:
            print(f"could not get A_known of {lifter}")
            A_known = []

        for A in A_known:
            a_dense = lifter.get_vec(A.toarray())
            a_sparse = lifter.get_vec(A)
            np.testing.assert_allclose(a_dense, a_sparse)

            # get_vec multiplies off-diagonal elements by sqrt(2)
            a = lifter.get_vec(A)

            A_test = lifter.get_mat(a, sparse=False)
            np.testing.assert_allclose(A.toarray(), A_test)

            # get_mat divides off-diagonal elements by sqrt(2)
            A_test = lifter.get_mat(a, sparse=True)
            np.testing.assert_allclose(A.toarray(), A_test.toarray())

            a_poly = lifter.convert_a_to_polyrow(a)
            a_test = lifter.convert_poly_to_a(a_poly)
            np.testing.assert_allclose(a, a_test)

        A_learned = lifter.get_A_learned(A_known=A_known, normalize=False)
        for A_l, A_k in zip(A_learned, A_known):
            np.testing.assert_allclose(A_l.toarray(), A_k.toarray())

if __name__ == "__main__":
    import sys
    import warnings

    test_ravel()
    test_vec_mat()

    # import pytest
    # print("testing")
    # pytest.main([__file__, "-s"])
    # print("all tests passed")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_constraints()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    print("all tests passed")
    # sys.exit()
