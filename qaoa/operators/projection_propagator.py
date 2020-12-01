import numpy as np
from qaoa.operators import Propagator
from qaoa.operators import ProjectionOperator
from qaoa.util.math import projection_exp_1d   

class ProjectionPropagator(Propagator):

    def __init__(self,P,theta=0):
        
        assert( isinstance(P,ProjectionOperator) )
        super().__init__(P,theta)
        self.alpha = np.exp(1j*self.theta)

    def set_control(self,theta):
        self.theta = theta
        self.alpha = np.exp(1j*self.theta)

    def apply(self,x,y):
        projection_exp_1d(self.alpha,self.get_operator().v,x,y)
