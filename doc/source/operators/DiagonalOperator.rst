DiagonalOperator
================


Diagonal operators have the matrix representation

.. math::
    D = \begin{pmatrix} d_{11} & 0      & \cdots & 0      \\
                        0      & d_{22} & \ddots & \vdots \\
                        \vdots & \ddots & \ddots & 0 &    \\
                        0      & \cdots & 0      & d_{NN} \end{pmatrix} 
      = \text{diag}(d)

Where :math:`d_{ii}\in\mathbb{R},\;1\leq i \leq N`. Matrix vector multiplication is
performed efficiently using the vector Hadamard product

.. math::
   y = D x = d \odot x


.. autoclass:: qaoa.operators.DiagonalOperator
    :members:
    :undoc-members:
    :show-inheritance:


