ProjectionOperator
==================

ProjectionOperator objects perform the action of an orthogonal projection onto a vector

.. math:: P = \frac{vv^\ast}{v^\ast v}

Matrix-vector multiplication is performed efficiently using the formula

.. math:: y = Px = \frac{v^\ast x}{v^\ast v} v

.. autoclass:: qaoa.operators.ProjectionOperator
    :members:
    :undoc-members:
    :show-inheritance:


