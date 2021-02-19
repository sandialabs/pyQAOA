import numpy as np
from qaoa.operators import Propagator
from qaoa.operators import ProjectionOperator
from qaoa.util.math import projection_exp_1d   

class ProjectionPropagator(Propagator):

    def __init__(self,P,theta=0):
        
        assert( isinstance(P,ProjectionOperator) )
        super().__init__(P,theta)

    def set_control(self,theta):
        self.theta = theta

    def apply(self,x,y):
        projection_exp_1d(np.exp(1j*self.theta),self.get_operator().v,x,y)

    def apply_adjoint(self,x,y):
        projection_exp_1d(np.exp(-1j*self.theta),self.get_operator().v,x,y)
