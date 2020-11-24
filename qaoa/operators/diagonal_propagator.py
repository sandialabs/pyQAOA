from qaoa.operators import DiagonalOperator
from qaoa.operators import Propagator
from qaoa.util.math import cexp_hadamard_mult, cexp_hadamard_div
import numpy as np

class DiagonalPropagator(Propagator):

    def __init__(self,D,theta=0):

       assert(isinstance(D,DiagonalOperator))

       super().__init__(D,theta)

    def apply(self,v,u):
        cexp_hadamard_mult(self.get_operator().data,self.theta,v,u)
    
    def apply_adjoint(self,v,u):
        cexp_hadamard_div(self.get_operator().data,self.theta,v,u)

    
    
