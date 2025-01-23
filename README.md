# Wolfe-Gradient-Projection-QP
Efficient implementation of Wolfe's Gradient Projection Algorithm for Convex QP problems with simplex constraints. Features include exact line search, scalable partitioned framework, and simplex projections. Benchmarked on large, sparse matrices, outperforming traditional solvers like CVXOPT.

## Overview
This project implements the Wolfe projection method for solving quadratic optimization problems, with comparisons to the CVXOPT solver.

## Directory Structure
- utils.py: Utility functions(such as initialization,projections etc.).
- solver.py: Solver implementations.
- analysis.py: Comparison and analysis functions.
- visualization.py: Plotting utilities.
- matrix: Folder for .mat files.
- project_main: Jupyter Notebook  the workflow of the project.
- main.py: Main script to run the project with all the matrix.
-project_main: notebook for run any specific matrix


## Setup
Install dependencies:
in terminal 
pip install -r ../requirements.txt

