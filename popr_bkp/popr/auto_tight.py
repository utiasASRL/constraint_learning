import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
from cert_tools.linalg_tools import find_dependent_columns, get_nullspace
from popr.utils.common import get_vec
from popr.utils.constraint import Constraint
from popr.utils.plotting_tools import plot_singular_values


class AutoTight(object):
    # consider singular value zero below this
    EPS_SVD = 1e-5

    # basis pursuit method, can be
    # - qr: qr decomposition
    # - qrp: qr decomposition with permutations (sparser), recommended
    # - svd: svd
    METHOD = "qrp"

    # normalize learned Ai or not
    NORMALIZE = False

    # how much to oversample (>= 1)
    FACTOR = 1.2

    # number of times we remove bad samples from data matrix
    N_CLEANING_STEPS = 1  # was 3

    # maximum number of iterations of local solver
    LOCAL_MAXITER = 100

    # find and remove linearly dependent constraints
    REDUCE_DEPENDENT = False

    def __init__(self):
        pass

    @staticmethod
    def clean_Y(basis_new, Y, s, plot=False):
        errors = np.abs(basis_new @ Y.T)  # Nb x n x n x Ns = Nb x Ns
        if np.all(errors < 1e-10):
            return []
        bad_bins = np.unique(np.argmax(errors, axis=1))
        if plot:
            fig, ax = plt.subplots()
            ax.semilogy(np.min(errors, axis=1))
            ax.semilogy(np.max(errors, axis=1))
            ax.semilogy(np.median(errors, axis=1))
            ax.semilogy(s)
        return bad_bins

    @staticmethod
    def test_S_cutoff(S, corank):
        if corank > 1:
            try:
                assert abs(S[-corank]) / AutoTight.EPS_SVD < 1e-1  # 1e-1  1e-10
                assert abs(S[-corank - 1]) / AutoTight.EPS_SVD > 10  # 1e-11 1e-10
            except AssertionError:
                print(
                    f"there might be a problem with the chosen threshold {AutoTight.EPS_SVD}:"
                )
                print(S[-corank], AutoTight.EPS_SVD, S[-corank - 1])

    @staticmethod
    def get_basis_sparse(lifter, var_list, param_list, A_known=[]):

        Y = AutoTight.generate_Y_sparse(
            lifter, var_subset=var_list, param_subset=param_list, factor=1.0
        )
        basis, S = AutoTight.get_basis(lifter, Y, A_known=A_known, var_subset=var_list)
        AutoTight.test_S_cutoff(S, corank=basis.shape[0])
        constraints = []
        for i, b in enumerate(basis):
            constraints.append(
                Constraint.init_from_b(
                    i,
                    b,
                    mat_var_dict=var_list,
                    mat_param_dict=param_list,
                    convert_to_polyrow=False,
                    known=False,
                )
            )
        return constraints

    @staticmethod
    def get_A_learned(
        lifter,  #:StateLifter,
        A_known=[],
        var_dict=None,
        method=METHOD,
        verbose=False,
    ) -> list:
        import time

        t1 = time.time()
        Y = AutoTight.generate_Y(lifter, var_subset=var_dict, factor=1.0)
        if verbose:
            print(f"generate Y ({Y.shape}): {time.time() - t1:4.4f}")
        t1 = time.time()
        basis, S = AutoTight.get_basis(
            lifter, Y, A_known=A_known, method=method, var_subset=var_dict
        )
        if verbose:
            print(f"get basis ({basis.shape})): {time.time() - t1:4.4f}")
        t1 = time.time()
        A_learned = AutoTight.generate_matrices(lifter, basis, var_dict=var_dict)
        if verbose:
            print(f"get matrices ({len(A_learned)}): {time.time() - t1:4.4f}")
        return A_learned

    @staticmethod
    def get_A_learned_simple(
        lifter,  #:StateLifter,
        A_known=[],
        var_dict=None,
        method=METHOD,
        verbose=False,
    ) -> list:
        import time

        t1 = time.time()
        Y = AutoTight.generate_Y_simple(lifter, var_subset=var_dict, factor=1.5)
        if verbose:
            print(f"generate Y ({Y.shape}): {time.time() - t1:4.4f}")
        t1 = time.time()
        if len(A_known):
            basis_known = np.vstack(
                [get_vec(Ai.get_matrix(var_dict)) for Ai in A_known]
            ).T
        else:
            basis_known = None
        basis, S = AutoTight.get_basis(
            lifter, Y, basis_known=basis_known, method=method, var_subset=var_dict
        )
        if verbose:
            print(f"get basis ({basis.shape})): {time.time() - t1:4.4f}")
        t1 = time.time()
        A_learned = AutoTight.generate_matrices_simple(lifter, basis, var_dict=var_dict)
        if verbose:
            print(f"get matrices ({len(A_learned)}): {time.time() - t1:4.4f}")
        return A_learned

    @staticmethod
    def generate_Y_simple(lifter, var_subset, factor):
        # need at least dim_Y different random setups
        dim_Y = lifter.get_dim_X(var_subset)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = lifter.sample_theta()
            x = lifter.get_x(theta=theta, parameters=None, var_subset=var_subset)
            X = np.outer(x, x)
            Y[seed, :] = get_vec(X)
        return Y

    @staticmethod
    def generate_Y_sparse(
        lifter, factor=FACTOR, ax=None, var_subset=None, param_subset=None
    ):
        assert (
            lifter.HOM in param_subset
        ), f"{lifter.HOM} must be included in param_subset."
        # need at least dim_Y different random setups
        dim_Y = lifter.get_dim_Y(var_subset, param_subset)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = lifter.sample_theta()
            parameters = lifter.sample_parameters(theta)

            if seed < 10 and ax is not None:
                if np.ndim(lifter.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = lifter.get_x(theta=theta, parameters=parameters, var_subset=var_subset)
            X = np.outer(x, x)

            # generates [1*x, a1*x, ..., aK*x]
            p = lifter.get_p(parameters=parameters, param_subset=param_subset)
            Y[seed, :] = np.kron(p, get_vec(X))
        return Y

    @staticmethod
    def generate_Y(lifter, factor=FACTOR, ax=None, var_subset=None, param_subset=None):
        # need at least dim_Y different random setups
        dim_Y = lifter.get_dim_Y(var_subset, param_subset)
        n_seeds = int(dim_Y * factor)
        Y = np.empty((n_seeds, dim_Y))
        for seed in range(n_seeds):
            np.random.seed(seed)

            theta = lifter.sample_theta()
            parameters = lifter.sample_parameters(theta)
            if seed < 10 and ax is not None:
                if np.ndim(lifter.theta) == 1:
                    ax.scatter(np.arange(len(theta)), theta)
                else:
                    ax.scatter(*theta[:, :2].T)

            x = lifter.get_x(theta=theta, parameters=parameters, var_subset=var_subset)
            X = np.outer(x, x)

            # generates [1*x, a1*x, ..., aK*x]
            p = lifter.get_p(parameters=parameters, param_subset=param_subset)
            assert p[0] == 1
            Y[seed, :] = np.kron(p, get_vec(X))
        return Y

    @staticmethod
    def get_basis(
        lifter,
        Y,
        A_known: list = [],
        basis_known: np.ndarray = None,
        var_subset: dict = None,
        method=METHOD,
    ):
        """Generate basis from lifted state matrix Y.

        :param A_known: if given, will generate basis that is orthogonal to these given constraints.

        :return: basis, S
        """

        # if there is a known list of constraints, add them to the Y so that resulting nullspace is orthogonal to them
        if basis_known is not None:
            if len(A_known):
                print(
                    "Warning: ignoring given A_known because basis_all is also given."
                )
            Y = np.vstack([Y, basis_known.T])
        elif len(A_known):
            A = np.vstack(
                [lifter.augment_using_zero_padding(get_vec(a)) for a in A_known]
            )
            Y = np.vstack([Y, A])

        basis, info = get_nullspace(Y, method=method, tolerance=AutoTight.EPS_SVD)

        basis[np.abs(basis) < lifter.EPS_SPARSE] = 0.0
        return basis, info["values"]

    @staticmethod
    def generate_matrices_simple(
        lifter,
        basis,
        normalize=NORMALIZE,
        sparse=True,
        trunc_tol=1e-10,
        var_dict=None,
    ):
        """
        Generate constraint matrices from the rows of the nullspace basis matrix.
        """
        try:
            n_basis = len(basis)
        except Exception:
            n_basis = basis.shape[0]

        if isinstance(var_dict, list):
            var_dict = lifter.get_var_dict(var_dict)

        from popr.lifters import StateLifter

        assert isinstance(lifter, StateLifter)

        A_list = []
        for i in range(n_basis):
            ai = basis[i]
            Ai = lifter.get_mat(ai, sparse=sparse, correct=True, var_dict=None)
            # Normalize the matrix
            if normalize and not sparse:
                # Ai /= np.max(np.abs(Ai))
                Ai /= np.linalg.norm(Ai, p=2)
            elif normalize and sparse:
                Ai /= sp.linalg.norm(Ai, ord="fro")
            # Sparsify and truncate
            if sparse:
                Ai.eliminate_zeros()
            else:
                Ai[np.abs(Ai) < trunc_tol] = 0.0
            # add to list
            A_list.append(Ai)
        return A_list

    @staticmethod
    def generate_matrices(
        lifter,
        basis,
        normalize=NORMALIZE,
        sparse=True,
        trunc_tol=1e-10,
        var_dict=None,
    ):
        """
        Generate constraint matrices from the rows of the nullspace basis matrix.
        """
        from popr.lifters import StateLifter

        assert isinstance(lifter, StateLifter)

        try:
            n_basis = len(basis)
        except Exception:
            n_basis = basis.shape[0]

        if isinstance(var_dict, list):
            var_dict = lifter.get_var_dict(var_dict)

        A_list = []
        basis_reduced = []
        for i in range(n_basis):
            ai = lifter.get_reduced_a(bi=basis[i], var_subset=var_dict, sparse=True)
            basis_reduced.append(ai)
        basis_reduced = sp.vstack(basis_reduced)

        if AutoTight.REDUCE_DEPENDENT:
            bad_idx = find_dependent_columns(basis_reduced.T, tolerance=1e-6)
        else:
            bad_idx = []

        for i in range(basis_reduced.shape[0]):
            if i in bad_idx:
                continue
            ai = basis_reduced[[i], :].toarray().flatten()
            Ai = lifter.get_mat(ai, sparse=sparse, correct=True, var_dict=None)
            # Normalize the matrix
            if normalize and not sparse:
                # Ai /= np.max(np.abs(Ai))
                Ai /= np.linalg.norm(Ai, p=2)
            elif normalize and sparse:
                Ai /= sp.linalg.norm(Ai, ord="fro")
            # Sparsify and truncate
            if sparse:
                Ai.eliminate_zeros()
            else:
                Ai[np.abs(Ai) < trunc_tol] = 0.0
            # add to list
            A_list.append(Ai)
        return A_list

    @staticmethod
    def get_basis_list(
        lifter,
        var_subset,
        A_known: list = [],
        plot: bool = False,
        method: str = METHOD,
    ):
        print("Warning: do not use get_basis_list anymore, it is not efficient.")
        """Generate list of PolyRow basis vectors"""
        basis_list = []

        Y = AutoTight.generate_Y(lifter, var_subset=var_subset)
        if len(A_known):
            A_known = lifter.extract_A_known(A_known, var_subset, output_type="dense")
            Y = np.r_[Y, A_known]
            for i in range(A_known.shape[0]):
                basis_list.append(lifter.convert_b_to_polyrow(A_known[i], var_subset))

        for i in range(lifter.N_CLEANING_STEPS + 1):
            # TODO(FD) can e enforce lin. independance to previously found
            # matrices at this point?
            basis_new, S = AutoTight.get_basis(lifter, Y, method=method)
            corank = basis_new.shape[0]

            if corank == 0:
                print(f"{var_subset}: no new learned matrices found")
                return basis_list
            print(f"{var_subset}: {corank} learned matrices found")
            AutoTight.test_S_cutoff(S, corank)
            bad_bins = AutoTight.clean_Y(basis_new, Y, S[-corank:], plot)
            if len(bad_bins) > 0:
                print(f"deleting {len(bad_bins)}")
                Y = np.delete(Y, bad_bins, axis=0)
            else:
                break

        if plot:
            plot_singular_values(S, eps=lifter.EPS_SVD)

        # find out which of the constraints are linearly dependant of the others.
        current_basis = None
        for i, bi_sub in enumerate(basis_new):
            bi_poly = lifter.convert_b_to_polyrow(
                bi_sub, var_subset, tol=lifter.EPS_SPARSE
            )
            ai = lifter.get_reduced_a(bi_sub, var_subset)
            basis_list.append(bi_poly)
            if current_basis is None:
                current_basis = ai[None, :]
            else:
                current_basis = np.r_[current_basis, ai[None, :]]
        return basis_list
