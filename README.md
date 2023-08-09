# Constraint Learning

This codebase allows for the learning of constraint matrices for arbitrary lifting functions.

## Usage

- To regenerate all results from the paper, run `python _scripts/generate_all_results.sh`.

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
