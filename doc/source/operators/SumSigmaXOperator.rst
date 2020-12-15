SumSigmaXOperator
=================

A HermitianOperator for representing the common QAOA mixing Hamiltonian/driver operator

.. math:: 
    D &= \bigoplus_{k=1}^n X_k \\
    X_k &= I_{[2^{k}]} \otimes \sigma_x \otimes I_{[2^{n-k}]} \\
    \sigma_x &= \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}  \\
    I_{[k]} &\in \mathbb{R}^{k\times k},\quad [I_{[k]}]_{ij} = \delta_{ij}

.. autoclass:: qaoa.operators.SumSigmaXOperator
    


