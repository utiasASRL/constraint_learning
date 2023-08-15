import cvxpy as cp
import numpy as np

from .common import adjust_Q, adjust_tol, solver_options

VERBOSE = False
SOLVER = "MOSEK"

# for computing the lambda parameter, we are adding all possible constraints
# and therefore we might run into numerical problems. Setting below to a high value
# was found to lead to less cases where the solver terminates with "UNKNOWN" status.
# see https://docs.mosek.com/latest/pythonapi/parameters.html#doc-all-parameter-list
LAMBDA_REL_GAP = 0.1


def bisection(function, inputs, left_num, right_num):
    """
    functions is cost tightness or rank tightness, which is of shape
              .-----
             |
    ---------'
    *==============*
           *=======*   middle not tight --> look in right half
           *===*       middle tight --> look in left half
    """
    A_list, df_data = inputs

    left_tight = function(A_list[: left_num + 1], df_data)
    right_tight = function(A_list[: right_num + 1], df_data)

    if left_tight and right_tight:
        print(
            "Warning: not a valid starting interval, both left and right already tight!"
        )
        return
    elif (not left_tight) and (not right_tight):
        print("Warning: problem is not tight on left or right.")
        return

    assert not left_tight
    assert right_tight
    # start at 0

    middle_num = (right_num + left_num) // 2
    middle_tight = function(A_list[: middle_num + 1], df_data)

    if middle_tight:  # look in the left half next
        right_num = middle_num
    else:
        left_num = middle_num
    if right_num == left_num + 1:
        return
    return bisection(function, inputs, left_num=left_num, right_num=right_num)


def brute_force(function, inputs, left, right):
    A_list, df_data = inputs
    tightness_counter = 0
    for idx in range(left, right + 1):
        is_tight = function(A_list[:idx], df_data)
        if is_tight:
            tightness_counter += 1
        if tightness_counter >= 10:
            return


def solve_lambda(
    Q,
    A_b_list,
    xhat,
    B_list=[],
    force_first=1,
    adjust=False,
    solver=SOLVER,
    opts=solver_options[SOLVER],
    primal=False,
    verbose=True,
    tol=None,
):
    """Determine lambda with an SDP.
    :param force_first: number of constraints on which we do not put a L1 cost, effectively encouraging the problem to use them.
    """

    adjust_tol([opts], tol) if tol else None
    Q_here, scale, offset = adjust_Q(Q) if adjust else (Q, 1.0, 0.0)

    if primal:
        raise NotImplementedError("primal form not implemented yet")
    else:  # Dual
        """
        max | y |
        s.t. H := Q + sum(Ai * yi for all i) >> 0
             H xhat == 0
        """
        m = len(A_b_list)
        y = cp.Variable(shape=(m,))
        epsilon = cp.Variable()

        k = len(B_list)
        if k > 0:
            u = cp.Variable(shape=(k,))

        As, b = zip(*A_b_list)
        H = Q_here + cp.sum(
            [y[i] * Ai for (i, Ai) in enumerate(As)]
            + [u[i] * Bi for (i, Bi) in enumerate(B_list)]
        )

        if k > 0:
            objective = cp.Minimize(cp.norm1(y[force_first:]) + cp.norm1(u) + epsilon)
        else:
            objective = cp.Minimize(cp.norm1(y[force_first:]) + epsilon)

        constraints = [H >> 0]  # >> 0 denotes positive SEMI-definite

        # constraints += [H @ xhat == 0]
        constraints += [H @ xhat <= epsilon]
        constraints += [H @ xhat >= -epsilon]
        if k > 0:
            constraints += [u >= 0]

        cprob = cp.Problem(objective, constraints)
        opts["verbose"] = verbose
        if solver == "MOSEK":
            opts["mosek_params"]["MSK_DPAR_INTPNT_CO_TOL_REL_GAP"] = LAMBDA_REL_GAP
        try:
            cprob.solve(
                solver=solver,
                **opts,
            )
        except:
            lamda = None
            X = None
        else:
            try:
                print("solve_lamda: epsilon is", epsilon.value)
            except:
                print("solve_lamda: epsilon is", epsilon)
            lamda = y.value
            X = constraints[0].dual_value

    # reverse Q adjustment
    if lamda is not None:
        lamda[0] *= scale
        lamda[0] += offset
    return X, y.value
