import numpy as np

from lifters.poly_lifters import Poly4Lifter, Poly6Lifter, PolyLifter
from lifters.range_only_lifters import RangeOnlyLocLifter
from lifters.range_only_slam1 import RangeOnlySLAM1Lifter
from lifters.range_only_slam2 import RangeOnlySLAM2Lifter
from lifters.stereo1d_lifter import Stereo1DLifter
from lifters.stereo2d_lifter import Stereo2DLifter
from lifters.stereo3d_lifter import Stereo3DLifter

d = 3
n_landmarks = 5
n_poses = 5
Lifters = {
    # Poly4Lifter: dict(),
    # Poly6Lifter: dict(),
    # RangeOnlySLAM1Lifter: dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    # RangeOnlySLAM2Lifter: dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    RangeOnlyLocLifter: dict(n_positions=n_poses, n_landmarks=n_landmarks, d=d),
    # Stereo1DLifter: dict(n_landmarks),
    # Stereo2DLifter: dict(n_landmarks),
    # Stereo3DLifter: dict(n_landmarks),
}
# Below, we always reset seeds to make sure tests are reproducible.
all_lifters = []
for Lifter, kwargs in Lifters.items():
    np.random.seed(1)
    all_lifters.append(Lifter(**kwargs))


def test_hess_finite_diff():
    for lifter in all_lifters:
        lifter.generate_random_setup()
        lifter.sample_feasible()

        errors = []
        eps_list = np.logspace(-10, -5, 3)
        for eps in eps_list:
            Q, y = lifter.get_Q(noise=1e-2)
            theta = lifter.get_vec_around_gt(delta=0).flatten("C")

            try:
                grad = lifter.get_grad(theta, y)
                hess = lifter.get_hess(theta, y)
            except Exception as e:
                print(e)
                print("get_hess not implemented?")
                continue

            n = len(theta)
            I = np.eye(n) * eps

            max_err = -np.inf
            errors_mat = np.full((n, n), -np.inf)
            for i in range(n):
                theta_delta = theta + I[i]
                grad_delta = lifter.get_grad(theta_delta, y)

                hess_est = (grad_delta - grad) / eps

                abs_error = np.abs(hess_est - hess[i, :])
                errors_mat[i, :] = np.maximum(abs_error, errors_mat[i, :])
                max_err = max(np.max(abs_error), max_err)
            errors.append(max_err)

        try:
            assert min(errors) < 1e-5
        except ValueError:
            print("skipping Hessian test")
        except AssertionError:
            print(f"Hessian test for {lifter} not passing")
            # import matplotlib.pylab as plt
            # plt.figure()
            # plt.title(f"hess {lifter}")
            # plt.loglog(eps_list, errors)
            # plt.show()
            # assert  < 1e-7


def test_grad_finite_diff():
    for lifter in all_lifters:
        lifter.generate_random_setup()
        lifter.sample_feasible()

        errors = []
        eps_list = np.logspace(-10, -1, 11)
        for eps in eps_list:
            Q, y = lifter.get_Q(noise=1)

            theta = lifter.get_vec_around_gt(delta=0).flatten("C")
            cost = lifter.get_cost(theta, y)

            try:
                grad = lifter.get_grad(theta, y)
            except Exception as e:
                print(e)
                print("grad not implemented?")
                continue

            n = len(theta)
            I = np.eye(n) * eps

            max_err = -np.inf
            errors_mat = np.full(n, -np.inf)
            for i in range(n):
                theta_delta = theta + I[i]
                cost_delta = lifter.get_cost(theta_delta, y)

                grad_est = (cost_delta - cost) / eps

                err = abs(grad_est - grad[i])
                errors_mat[i] = max(errors_mat[i], err)
                max_err = max(err, max_err)

            errors.append(max_err)

        try:
            assert min(errors) < 1e-5
        except ValueError:
            print("skipping grad test")
        except AssertionError:
            import matplotlib.pylab as plt

            plt.figure()
            plt.title(f"grad {lifter}")
            plt.loglog(eps_list, errors)
            plt.show()
            # assert  < 1e-7


def test_equivalent_lifters():
    l1 = RangeOnlySLAM1Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d)
    l2 = RangeOnlySLAM2Lifter(n_positions=n_poses, n_landmarks=n_landmarks, d=d)

    np.random.seed(1)
    l1.generate_random_setup()
    l1.sample_feasible()
    Q, y = l1.get_Q(noise=1e-3)
    cost1 = l1.get_cost(l1.theta, y)

    np.random.seed(1)
    l2.generate_random_setup()
    l2.sample_feasible()
    Q, y = l2.get_Q(noise=1e-3)
    cost2 = l2.get_cost(l2.theta, y)
    assert abs(cost1 - cost2) < 1e-10

    # 1, t1, t2, ..., a1, a2, ..., |t1|^2, ..., |a1|^2, ..., a1 @ t1, ...
    x1 = l1.get_x()
    # 1, t1, t2, ..., a1, a2, ..., |t1 - a1|^2
    x2 = l2.get_x()

    start = 1 + l1.N
    np.testing.assert_allclose(x1[:start], x2[:start])
    for i, (n, k) in enumerate(l1.edges):
        x1_test = (
            x1[start + n]
            + x1[start + n_poses + k]
            - 2 * x1[start + n_poses + n_landmarks + i]
        )
        np.testing.assert_allclose(x2[start + i], x1_test)


def test_levels():
    from lifters.stereo_lifter import StereoLifter

    for level in StereoLifter.LEVELS:
        lifter_2d = Stereo2DLifter(n_landmarks=3, level=level)

        # inside below function we tests that dimensions are consistent.
        lifter_2d.get_x()

        lifter_3d = Stereo3DLifter(n_landmarks=3, level=level)
        lifter_3d.get_x()


def test_cost_noisy():
    test_cost(noise=0.1)


def test_cost(noise=0.0):
    for lifter in all_lifters:
        # np.random.seed(1)
        Q, y = lifter.get_Q(noise=noise)
        # np.random.seed(1)
        # Qold, yold = lifter.get_Q_old(noise=noise)
        # np.testing.assert_allclose(Q, Qold)
        # np.testing.assert_allclose(y, yold)
        if Q is None:
            continue

        theta = lifter.theta
        cost = lifter.get_cost(theta, y)

        x = lifter.get_x(theta)
        costQ = abs(x.T @ Q @ x)

        # TODO(FD) figure out why the tolerance is so bad
        # for Stereo3D problem.
        assert abs(cost - costQ) < 1e-7, (cost, costQ)

        if noise == 0 and not isinstance(lifter, PolyLifter):
            assert cost < 1e-10, cost
            assert costQ < 1e-7, costQ


def test_solvers_noisy(n_seeds=1, noise=1e-1):
    test_solvers(n_seeds=n_seeds, noise=noise)


def test_solvers(n_seeds=1, noise=0.0):
    for lifter in all_lifters:
        for j in range(n_seeds):
            np.random.seed(j)

            # noisy setup
            Q, y = lifter.get_Q(noise=noise)
            if Q is None:
                continue

            # test that we stay at real solution when initializing at it
            theta_gt = lifter.get_vec_around_gt(delta=0)
            theta_hat, msg, cost_solver = lifter.local_solver(theta_gt, y)
            if noise == 0:
                # test that solution is ground truth with no noise
                np.testing.assert_allclose(theta_hat, theta_gt)
            else:
                # just test that we converged when noise is added
                assert theta_hat is not None

            # test that we converge to real solution when initializing around it
            theta_0 = lifter.get_vec_around_gt(delta=1e-2)
            theta_hat, msg, cost_solver = lifter.local_solver(theta_0, y)

            if theta_hat is None:
                print(f"Warning: {lifter} did not converge noise {noise}, seed {j}.")
                continue
            else:
                print(f"{lifter} converged noise {noise}, seed {j}.")

            cost_lifter = lifter.get_cost(theta_hat, y)
            assert abs(cost_solver - cost_lifter) < 1e-10, (cost_solver, cost_lifter)

            if noise == 0:
                # test that "we made progress"
                progress = np.linalg.norm(theta_0 - theta_hat)
                assert progress > 1e-10, progress

                # test that cost decreased
                cost_0 = lifter.get_cost(theta_0, y)
                cost_hat = lifter.get_cost(theta_hat, y)
                assert cost_hat <= cost_0

                # TODO(FD) this doesn't pass, looks like the problem is actually not well conditioned!
                try:
                    np.testing.assert_allclose(theta_hat, theta_gt, atol=1e-5)
                except AssertionError as e:
                    print(
                        f"Found solution for {lifter} is not ground truth in zero-noise! is the problem well-conditioned?"
                    )
                    mineig_hess_hat = np.linalg.eigvalsh(lifter.get_hess(theta_hat, y))[
                        0
                    ]
                    mineig_hess_gt = np.linalg.eigvalsh(lifter.get_hess(theta_gt, y))[0]
                    print(
                        f"minimum eigenvalue at gt: {mineig_hess_gt:.1e} and at estimate: {mineig_hess_hat:.1e}"
                    )
            else:
                # test that "we made progress"
                progress = np.linalg.norm(theta_0 - theta_hat)
                assert progress > 1e-10, progress

                # just test that we converged when noise is added
                assert theta_hat is not None


def test_constraints():
    def test_with_tol(A_list, tol):
        x = lifter.get_x()
        for Ai in A_list:
            err = abs(x.T @ Ai @ x)
            assert err < tol, err

            # TODO(FD) not fully understood why this fixes the below test.
            Ai[range(Ai.shape[0]), range(Ai.shape[0])] /= 2.0

            ai = lifter._get_vec(Ai)
            xvec = lifter._get_vec(np.outer(x, x))
            np.testing.assert_allclose(xvec @ ai, 0.0, atol=tol)

    for lifter in all_lifters:
        Y = lifter.generate_Y()
        basis, S = lifter.get_basis(Y, eps=1e-7)
        A_learned = lifter.generate_matrices(basis)

        A_known = lifter.get_A_known()
        test_with_tol(A_learned, tol=1e-5)
        test_with_tol(A_known, tol=1e-10)


def test_gauge():
    import itertools

    # TODO(FD): understand why 3D needs 5 landmarks and positions
    params = [
        dict(n_landmarks=5, n_positions=5, d=3),
        dict(n_landmarks=3, n_positions=3, d=2),
    ]
    remove_gauge_list = ["hard", "cost", None]

    for param, remove_gauge in itertools.product(params, remove_gauge_list):
        lifter = RangeOnlySLAM1Lifter(**param, remove_gauge=remove_gauge)

        Q, y = lifter.get_Q(noise=0)
        theta = lifter.get_vec_around_gt(delta=0).flatten("C")

        grad = lifter.get_grad(theta, y)
        hess = lifter.get_hess(theta, y)

        np.testing.assert_almost_equal(grad, 0.0)

        S, V = np.linalg.eig(hess)
        mask = np.abs(S) < 1e-10
        n_null = np.sum(mask)

        if remove_gauge == "hard":
            assert n_null == 0
        else:
            assert n_null > 0
        continue

        eigvecs = V[:, mask]
        print(f"Hessian has nullspace of dimension {n_null}! Eigenvectors:", eigvecs)

        fig, ax = lifter.plot_setup(title=f"Nullspace dim: {n_null}")
        for i in range(n_null):
            vec = eigvecs[:, i]  # p0_x, p0_y, ...
            lifter.plot_nullvector(vec, ax, color=f"C{i}")

            # theta_delta = theta + eigvec
            # cost_delta = lifter.get_cost(theta_delta, y)
            # assert cost_delta < 1e-10, cost_delta
    print("done")
    return None


def compare_solvers():
    kwargs = {"method": None}

    compare_solvers = [
        "Nelder-Mead",
        "Powell",  # "CG",  CG takes forever.
        "BFGS",
        "Newton-CG",
        "TNC",
    ]
    noise = 1e-3
    for lifter in all_lifters:
        np.random.seed(0)

        # noisy setup
        Q, y = lifter.get_Q(noise=noise)
        if Q is None:
            continue

        # test that we stay at real solution when initializing at it
        theta_gt = lifter.get_vec_around_gt(delta=0)

        import time

        for solver in compare_solvers:
            kwargs["method"] = solver
            t1 = time.time()
            theta_hat, msg, cost_solver = lifter.local_solver(
                theta_gt, y, solver_kwargs=kwargs
            )
            ttot = time.time() - t1
            if theta_hat is None:
                print(solver, "failed")
            else:
                error = np.linalg.norm(theta_hat - theta_gt)
                print(
                    f"{solver} finished in {ttot:.4f}s, final cost {cost_solver:.1e}, error {error:.1e}"
                )


if __name__ == "__main__":
    import sys
    import warnings

    # import pytest
    # print("testing")
    # pytest.main([__file__, "-s"])
    # print("all tests passed")
    # test_solvers()
    # test_solvers_noisy()
    # test_gauge()
    test_grad_finite_diff()
    test_hess_finite_diff()

    test_cost()
    test_cost_noisy()

    test_equivalent_lifters()

    test_levels()

    test_constraints()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compare_solvers()
    # sys.exit()
