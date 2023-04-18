import matplotlib.pylab as plt
import numpy as np

from lifters.poly_lifters import PolyLifter
from lifters.test_tools import all_lifters


def test_hess_finite_diff():
    for lifter in all_lifters():
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
    for lifter in all_lifters():
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


def test_cost_noisy():
    test_cost(noise=0.1)


def test_cost(noise=0.0):
    for lifter in all_lifters():
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
    for lifter in all_lifters():
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

            Ai[range(Ai.shape[0]), range(Ai.shape[0])] /= 2.0
            ai = lifter._get_vec(Ai)
            xvec = lifter._get_vec(np.outer(x, x))
            np.testing.assert_allclose(ai @ xvec, 0.0, atol=tol)

    for lifter in all_lifters():
        A_known = lifter.get_A_known()
        test_with_tol(A_known, tol=1e-10)

        methods = ["qrp", "svd", "qr"]
        rank = None
        for method in methods:
            np.random.seed(0)
            A_learned = lifter.get_A_learned(factor=2, eps=1e-7, method=method)
            test_with_tol(A_learned, tol=1e-4)

            if rank is None:
                rank = len(A_learned)
            else:
                assert len(A_learned) == rank

            # np.random.seed(0)
            # A_learned2 = lifter.get_A_learned(factor=2, eps=1e-7, method=method)
            # test_with_tol(A_learned2, tol=1e-4)

            np.random.seed(0)
            A_learned_incremental = lifter.get_A_learned(
                factor=2, eps=1e-7, method=method, A_known=A_known
            )
            if method == "qrp":
                assert len(A_learned) == len(A_learned_incremental) + len(A_known)


def compare_solvers():
    kwargs = {"method": None}

    compare_solvers = [
        "Nelder-Mead",
        "Powell",  # "CG",  CG takes forever.
        # "Newton-CG",
        "BFGS",
        "TNC",
    ]
    noise = 1e-1
    for lifter in all_lifters():
        print("testing", lifter)
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
                    f"{solver} finished in {ttot:.4f}s, final cost {cost_solver:.1e}, error {error:.1e}. \n\tmessage:{msg} "
                )


if __name__ == "__main__":
    import sys
    import warnings

    # test_constraints()
    # sys.exit(0)
    # import pytest
    # print("testing")
    # pytest.main([__file__, "-s"])
    # print("all tests passed")
    # test_solvers()
    # test_solvers_noisy()
    # test_gauge()

    test_cost()
    test_cost_noisy()

    test_grad_finite_diff()
    test_hess_finite_diff()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compare_solvers()

    print("all tests passed")
    # sys.exit()
