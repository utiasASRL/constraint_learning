# Constraint Learning

This codebase allows for the learning of constraint matrices for arbitrary lifting functions.

## Installation

This codebase relies on several submodules. Make sure to clone it recursively:
```
git clone --recurse-submodules git@github.com:utiasASRL/constraint_learning
```
(if you forgot, you can still run `git submodule init` and `git submodule update` after a normal clone)

The below command creates an environment with all dependencies and installs this package locally.
```
conda env create -f environment.yml
```

## Usage

If you want to automatically tighten your own SDP, all you need to do is to create your own lifter class implementing the specific lifting functions you want to use, the cost and constraints matrices. Many example lifters can be found in the `lifters/` folder. To analyze your lifter, you can refer to the scripts `_scripts/run_<lifter>_study.py` for inspiration, and also `_scripts/run_autotemplate.py`.

## Reproducing results

To reproduce the results from our [paper](https://arxiv.org/abs/2308.05783) on automatic tightening of SDPs, run:
```
conda activate constraint_learning 
make results_generate
```

Alternatively, to inspect and plot existing results, run
```
make results_plot
```

## Contributors

- Frederike Dümbgen
- Connor Holmes
- Ben Agro
