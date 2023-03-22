from copy import deepcopy
import numpy as np

from thesis.experiments.utils import StereoLocalizationProblem
from thesis.simulation.sim import Camera, World, generate_random_T
from thesis.solvers.local_solver import stereo_localization_gauss_newton

# from thesis.solvers.global_sdp import global_sdp_solution
# from thesis.solvers.iterative_sdp import iterative_sdp_solution

# make camera
cam = Camera(
    f_u=484.5,
    f_v=484.5,
    c_u=322,
    c_v=247,
    b=0.24,
    R=1 * np.eye(4),
    fov_phi_range=(-np.pi / 12, np.pi / 12),
    fov_depth_range=(0.5, 5),
)
M = cam.M()


def local_solver(p_w, y, W, T_init, log=False):
    from thesis.solvers.local_solver import _stereo_localization_gauss_newton

    solution = _stereo_localization_gauss_newton(
        T_init=T_init, y=y, p_w=p_w, W=W, M=M, log=log
    )
    return solution.solved, solution.T_cw, solution.cost


if __name__ == "__main__":
    num_landmarks = 3
    world = World(
        cam=cam,
        p_wc_extent=np.array([[3], [3], [0]]),
        num_landmarks=num_landmarks,
    )
    world.clear_sim_instance()
    world.make_random_sim_instance()

    # Generative camera model
    y = cam.take_picture(world.T_wc, world.p_w)

    # %% global minima
    W = np.eye(4)
    r0 = np.zeros((3, 1))
    gamma_r = 0

    global_min_cost = np.inf
    global_min_T = None

    problem = StereoLocalizationProblem(
        world.T_wc,
        world.p_w,
        cam.M(),
        W,
        y,
        r_0=r0,
        gamma_r=gamma_r,
        T_init=generate_random_T(world.p_wc_extent),
    )

    for i in range(20):
        T_init = generate_random_T(world.p_wc_extent)
        p_tmp = deepcopy(problem)
        p_tmp.T_init = T_init
        solution = stereo_localization_gauss_newton(p_tmp, log=False)
        T_op = solution.T_cw
        local_minima = solution.cost[0][0]
        if solution.solved and local_minima < global_min_cost:
            global_min_cost = local_minima
            global_min_T = T_op

    # %%
    num_tries = 1
    RECORD_HISTORY = False
    mosek_params = {}

    local_solution = stereo_localization_gauss_newton(
        problem,
        log=False,
        max_iters=100,
        num_tries=num_tries,
        record_history=RECORD_HISTORY,
    )
    print(f"Global minima: {global_min_cost}")
    print(f"Local solution cost: {local_solution.cost[0][0]}")

    # iter_sdp_soln = iterative_sdp_solution(
    #    problem,
    #    problem.T_init,
    #    max_iters=10,
    #    min_update_norm=1e-10,
    #    return_X=False,
    #    mosek_params=mosek_params,
    #    max_num_tries=num_tries,
    #    record_history=RECORD_HISTORY,
    #    refine=False,
    #    log=False,
    # )
    #
    # iter_sdp_soln_refine = iterative_sdp_solution(
    #    problem,
    #    problem.T_init,
    #    max_iters=20,
    #    min_update_norm=1e-5,
    #    return_X=False,
    #    mosek_params=mosek_params,
    #    max_num_tries=num_tries,
    #    record_history=RECORD_HISTORY,
    #    refine=True,
    #    log=False,
    # )
    # global_sdp_soln = global_sdp_solution(
    #    problem,
    #    return_X=False,
    #    mosek_params=mosek_params,
    #    record_history=RECORD_HISTORY,
    #    refine=False,
    # )
    # global_sdp_soln_refine = global_sdp_solution(
    #    problem,
    #    return_X=False,
    #    mosek_params=mosek_params,
    #    record_history=RECORD_HISTORY,
    #    refine=True,
    # )

    # add_coordinate_frame(np.linalg.inv(global_min_T), ax, "$\mathfrak{F}_{global_min}$")
    # add_coordinate_frame(np.linalg.inv(local_solution.T_cw), ax, "$\mathfrak{F}_{local}$")
    # print(f"Iter SDP cost: {iter_sdp_soln.cost}")
    # print(f"Refined Iter SDP refine: {iter_sdp_soln_refine.cost[0][0]}")
    # print(f"Global SDP cost: {global_sdp_soln.cost}")
    # print(f"Refined Global SDP cost: {global_sdp_soln_refine.cost[0][0]}")
