import pandas as pd


def interpret_dataframe(lifter, A_b_list_all, order_dicts, fname_root):
    # create a new dataframe that is more readible than the expressions.
    from lifters.interpret import get_known_variables
    from poly_matrix.poly_matrix import PolyMatrix

    # expressions = []
    landmark_dict = get_known_variables(lifter)
    print("known values:", landmark_dict.keys())
    data_math = []
    for i, (A, b) in enumerate(A_b_list_all):
        # print(names[i])
        try:
            A_poly, var_dict = PolyMatrix.init_from_sparse(
                A, lifter.var_dict, unfold=True
            )
        except Exception as e:
            print(f"error at {i}:", e)
            continue
        sparse_series = A_poly.interpret(var_dict)
        data_math.append(sparse_series)

    df_math = pd.DataFrame(data=data_math, dtype="Sparse[object]")
    for name, values in order_dicts.items():
        df_math.loc[:, name] = values

    def sort_fun(series):
        return series.isna()

    df_math.dropna(axis=1, how="all", inplace=True)
    df_math.sort_values(
        key=sort_fun,
        by=list(df_math.columns),
        axis=0,
        na_position="last",
        inplace=True,
    )
    if fname_root != "":
        fname = fname_root + "_math.pkl"
        pd.to_pickle(df_math, fname)
        print("saved math as", fname)
        fname = fname_root + "_math.csv"
        df_math.to_csv(fname)
        print("saved math as", fname)
