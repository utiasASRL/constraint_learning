import numpy as np

from decomposition.generate_cliques import create_clique_list_loc
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.cert_matrix import get_cost_matrices
from ro_certs.generate_cliques import generate_clique_list
from ro_certs.problem import Reg


def test_clique_creation():
    new_lifter = RangeOnlyLocLifter(
        n_landmarks=10, n_positions=5, d=3, reg=Reg.CONSTANT_VELOCITY
    )
    clique_list = create_clique_list_loc(
        new_lifter, use_known=True, use_autotemplate=False
    )
    cost_matrices = get_cost_matrices(new_lifter.prob)
    clique_list_new = generate_clique_list(new_lifter.prob, cost_matrices)
    for c1, c2 in zip(clique_list_new, clique_list):
        ii1, jj1 = c1.Q.nonzero()
        ii2, jj2 = c2.Q.nonzero()
        np.testing.assert_allclose(ii1, ii2)
        np.testing.assert_allclose(jj1, jj2)
        np.testing.assert_allclose(c1.Q.data, c2.Q.data)
        for i, (A1, A2) in enumerate(zip(c1.A_list, c2.A_list)):
            np.testing.assert_allclose(A1.toarray(), A2.toarray())


if __name__ == "__main__":
    test_clique_creation()
    print("done")
