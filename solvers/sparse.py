import cvxpy as cp
import numpy as np


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

    left_tight = function(A_list[:left_num], df_data)
    right_tight = function(A_list[:right_num], df_data)

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


def brute_force(function, inputs, left_num, right_num):
    A_list, df_data = inputs
    tightness_counter = 0
    for idx in range(left_num, right_num + 1):
        is_tight = function(A_list[:idx], df_data)
        if is_tight:
            tightness_counter += 1
        if tightness_counter >= 100:
            return
