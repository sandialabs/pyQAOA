Derivative Calculation
======================

The pyQAOA package evaluates the analytical gradient and second directional derivatives
by formulating a constrained optimization problem using adjoint variables and
sensitities. 

Suppose we have a circuit with :math:`m` stages, where the output state of stage :math:`k` is
:math:`\psik{k}` 

Let the objective function be denoted as 

