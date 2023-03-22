# Constraint Learning

This codebase allows for the learning of constraint matrices for arbitrary lifting functions.

To implement your own lifting function, inherit from the `StateLifter` class, following the examples in `custom_lifters.py`.

Visualizations of the example lifting functions and an explanation of the method can be found in `_notebooks/ConstraintsLearning.ipynb`. 

## Installation

This codebase relies on poly_matrix, which is added as a submodule. Run below commands for installation with pip.
```
pip install -e poly_matrix/
pip install -e .
```

## Author

Frederike Dümbgen
