import numpy as np
import pandas as pd

# rounding for landmark and other known variables, to identify them in constraints.
N_ROUND = 4


def get_known_variables(lifter, n_round=N_ROUND, color=False):
    known_dict = {
        np.round(1 / np.sqrt(2), n_round): f"$\\sqrt{{2}}^{{-1}}$",
        np.round(np.sqrt(2), n_round): f"$\\sqrt{{2}}$",
    }
    try:
        names = {0: "x", 1: "y", 2: "z"}
        if color:
            known_dict.update(
                {
                    np.round(
                        lifter.landmarks[k, d], n_round
                    ): f"$\\textcolor{{red}}{{a_{k}^{names[d]}}}$"
                    for k in range(lifter.n_landmarks)
                    for d in range(lifter.d)
                }
            )
        else:
            known_dict.update(
                {
                    np.round(lifter.landmarks[k, d], n_round): f"$a_{k}^{names[d]}$"
                    for k in range(lifter.n_landmarks)
                    for d in range(lifter.d)
                }
            )
    except Exception as e:
        print(e)
    return known_dict


def convert_series_to_string(row, landmark_dict={}, precision=8, verbose=False):
    """
    Convert row to an interpretable string.
    """
    multiindex = pd.MultiIndex.from_arrays([np.zeros(len(row)), row.index])
    row.index = multiindex
    row_coo = row.astype(pd.SparseDtype(float, fill_value=0)).sparse.to_coo()[0]
    string = ""
    for idx in row_coo.col:
        key = row[0].index[idx]
        val = np.round(row[0][key], precision)
        # check if there is a landmark coordinate in this value.
        key_print = "$" + key.replace(".", "\\cdot ") + "$"
        val_key = float(np.round(val, N_ROUND))
        if val_key in landmark_dict:
            string += f" +{landmark_dict[val_key]} {key_print}"
        elif -val_key in landmark_dict:
            string += f" -{landmark_dict[-val_key]} {key_print}"
        elif abs(val) < 1e-4:
            continue
        elif np.mod(abs(val), 1) < 1e-8:
            # string += f" + {val_key:.2f} {key}"
            string += f" {val:+.0f} {key_print}"
        elif np.mod(abs(val) * 10, 1) < 1e-8:
            # string += f" + {val_key:.2f} {key}"
            string += f" {val:+.1f} {key_print}"
        else:
            if verbose:
                print("convert_series_to_string: unknown value", val)
            string += f" +$\\alpha$ {key_print}"
    return string[1:]
