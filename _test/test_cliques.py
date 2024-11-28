import numpy as np
from cert_tools.hom_qcqp import HomQCQP
from cert_tools.test_tools import constraints_test, cost_test

from lifters.matweight_lifter import MatWeightLocLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from ro_certs.problem import Reg


def test_clique_decomp():
    n_params = 4
    lifters = [
        MatWeightLocLifter(n_landmarks=8, n_poses=n_params),
        RangeOnlyLocLifter(
            n_landmarks=8,
            n_positions=n_params,
            reg=Reg.CONSTANT_VELOCITY,
            d=2,
            level="no",
        ),
    ]
    noises = [1.0, 1e-4]
    sparsity = 1.0
    seed = 0
    for lifter, noise in zip(lifters, noises):
        np.random.seed(seed)
        lifter.generate_random_setup()
        lifter.simulate_y(noise=noise, sparsity=sparsity)

        problem = HomQCQP.init_from_lifter(lifter)

        # doing get_asg just to suppress a warning.
        problem.get_asg()
        problem.clique_decomposition()

        constraints_test(problem)
        cost_test(problem)


if __name__ == "__main__":
    test_clique_decomp()
    print("done")
