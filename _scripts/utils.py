import os.path


def get_fname(name):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_results", f"{name}.pkl")
    )
