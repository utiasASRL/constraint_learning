# Constraint Learning

This codebase allows for the learning of constraint matrices for arbitrary lifting functions.

## Usage

- To implement your own lifting function, inherit from the `StateLifter` class, following for example [lifters/stereo2d_lifter.py](lifters/stereo2d_lifter.py).
- Visualizations of the example lifting functions and an explanation of the methods can be found in [_notebooks/ConstraintsLearning.ipynb](_notebooks/ConstraintsLearning.ipynb). 
- To analyze tightness for different constraints and orderings, run `_scripts/constraints_study`. 
- To interpret constraints mathematically, run `_scripts/interpret`. 

## Installation

This codebase relies on poly_matrix, which is added as a submodule. Make sure to clone it recursively:
```
git clone --recurse-submodules git@github.com:utiasASRL/constraint_learning
```
(if you forgot, you can still run `git submodule init` and `git submodule update` after a normal clone)


Run below commands for installation with pip.
```
pip install -e poly_matrix/
pip install -e .
```
Run below commands for installation with conda.
```
conda env create -f environment.yml
```

## Author

Frederike Dümbgen
