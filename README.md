# Constraint Learning

This codebase allows for creating **tight** semidefinite relaxations of non-convex polynomial optimization problems in a semi-automatic way. The technique has been successfully applied to many state estimation problems in robotics and computer vision and is described in more detail in this [paper](http://arxiv.org/abs/2308.05783). 

In a nutshell, the codebase provides the *AUTOTIGHT* and *AUTOTEMPLATE* algorithms. We start with an optimization problem written in (QCQP) form:

$$ 
\begin{align} q^\star &\min_{x} x^\top Q x  \\ 
& \text{s.t. } (\forall i): x^\top A_i x = b_i
\end{align} 
$$

with cost matrix $Q$, known constraint matrices $A_i$, and $b_0=1, b_{i\geq 0}=0$. Many maximum-a-posteriori or maximum-likelihood estimation problems can be written as such, (for example range-only [localization](https://arxiv.org/abs/2209.04266) and range-aided [SLAM](https://arxiv.org/abs/2302.11614), ([matrix-weighted](https://arxiv.org/abs/2308.07275)) [SLAM](https://arxiv.org/abs/1612.07386), [outlier-robust estimation](https://ieeexplore.ieee.org/abstract/document/9785843)), and the same is true for many control and planning problems (for example [inverted pendulum](https://arxiv.org/abs/2406.05846) and other classical dynamical systems, and [slider-pusher planning systems](https://arxiv.org/abs/2402.10312)). 

*AUTOTIGHT* finds all possible additional constraints matrices $B_i$ which are also automatically satisfied by solutions of (QCQP), called **redundant constraints** and checks if the SDP (rank-)relaxation of the QCQP is cost and/or rank-tight after adding them. The rank relaxation is given by:

$$ 
\begin{align} p^\star &\min_{X} \langle Q X \rangle  \\ 
& \text{s.t. } (\forall i): \langle A_i X \rangle = b_i
\end{align} 
$$

Cost-tight means that strong duality holds ($p^\star = q^\star$) while rank-tight means that we even have $\text{rank}(X)=1$.
If successful, the output is a set of constraints that leads to a tight SDP relaxation of the original problem, which can be used to solve the problem to global optimality (if we have rank tightness) or certify given solutions (if we have cost tightness). 

*AUTOTEMPLATE* follows the same principle as *AUTOTIGHT*, but its output are templates rather than constraints. These templates can be seen as "parametrized" versions of the constraints matrices, and can be applied to new problem instances of any size without having to learn the constraints again from scratch. 

## Citation
If you use this codebase, please cite our [paper](http://arxiv.org/abs/2308.05783):

```bibtex
@article{dumbgen_toward_2024,
  title = {Toward Globally Optimal State Estimation Using Automatically Tightened Semidefinite Relaxations},
  author = {Dümbgen, Frederike and Holmes, Connor and Agro, Ben and Barfoot, Timothy D.},
  year = {2024},
  journal = {IEEE Transactions on Robotics (to appear)},
  publisher = {IEEE},
}
```

## Installation

Clone this codebase by running:
```
git clone --recurse-submodules git@github.com:utiasASRL/constraint_learning
```

The below command creates an environment with all dependencies and installs this package (and all required submodules) locally.
```
conda env create -f environment.yml
```

To test that the installation was successful, you can generate a representative set of example results by running
```
conda activate constraint_learning 
make results_test
```
The command should run in less than 4 minutes on a modern laptop, and the output can be found in the `_results_test` folder. The expected terminal output can be found in `_results/test/terminal_output.log`.

## Dependencies

Besides the automatically installed dependencies when using the above instructions, you need to also have a valid MOSEK license in order to use this repository. If you are an academic, you can get a license for free [here](https://www.mosek.com/license/request/?i=acp).

For plotting, we assume that LaTeX is installed.

## Usage

If you want to automatically tighten your own SDP, all you need to do is to create your own lifter class implementing the specific lifting functions you want to use (i.e. define $x$), the cost ($Q$) and constraints matrices ($A_i$). Many example lifters can be found in the `lifters/` folder. To analyze your lifter, you can refer to the scripts `_scripts/run_<lifter>_study.py` for inspiration, and also `_scripts/run_autotemplate.py`.

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
