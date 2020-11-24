# pyQAOA: Simulation of Quantum Approximate Optimization Algorithms in Python

This package allows easy simulation of the expectation value of a Hamiltonian operator
(objective value) that is produced via a sequence of controlled unitary operations on 
an initial state. Using numerically efficient matrix-free representations to 
compute the action of Hermitian operators and their generated unitaries allows for 
rapid evaluation of the objective function. Adjoint methods and sensitivity equations
further provide access to objective gradients and Hessians or Hessian-vector products 
for use in gradient-based numerical optimization schemes. 

# Requirements
* Python 3
* NumPy
* Numba
* NetworkX
* SciPy


