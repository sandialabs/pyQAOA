# pyQAOA: Simulation of Quantum Approximate Optimization Algorithms in Python

This package allows easy simulation of the expectation value of a Hamiltonian operator
(objective value) that is produced via a sequence of controlled unitary operations on 
an initial state. Using numerically efficient matrix-free representations to 
compute the action of Hermitian operators and their generated unitaries allows for 
rapid evaluation of the objective function. Adjoint methods and sensitivity equations
further provide access to objective gradients and Hessians or Hessian-vector products 
for use in gradient-based numerical optimization schemes. 

In general, we are interested in finding a collection of control angles that will minimize
an objective function (expectation value)

![Objective Function](/doc/equations/objective.pdf)


# Requirements
* Python 3
* NumPy
* Numba
* NetworkX
* SciPy

##### [LICENSE](https://github.com/gregvw/pyQAOA/blob/master/LICENSE)

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Under the terms of Contract DE-NA0003525 with NTESS,
the U.S. Government retains certain rights in this software.
