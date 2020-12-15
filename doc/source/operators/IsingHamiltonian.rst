IsingHamiltonian
================

A :py:class:`qaoa.operators.DiagonalOperator` for representing Ising Model Hamiltonian operators

.. math::
    H = \sum\limits_{i=1}^n h_i Z_i + \sum\limits_{i,j\in E} J_{ij} Z_i Z_j

for an n-qubit system where the multi-qubit spin operators are defined as

.. math::
    Z_k &= I_{[2^{k}]} \otimes \sigma_z \otimes I_{[2^{n-k}]} \\
   \sigma_z &= \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}  \\
   I_{[k]} &\in \mathbb{R}^{k\times k},\quad [I_{[k]}]_{ij} = \delta_{ij}



.. autoclass:: qaoa.operators.IsingHamiltonian


