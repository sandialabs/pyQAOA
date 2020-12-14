import numpy as np
from qaoa.operators import Kronecker, SumSigmaXOperator, Propagator

class SumSigmaXPropagator(Propagator):
    
    def __init__(self,D,theta=0):
        assert( isinstance(D,SumSigmaXOperator) )
        self.kronecker = Kronecker(np.eye(2),D.num_qubits(),dtype=complex)
        super().__init__(D,theta)

    def __str__(self):
        return "SumSigmaXPropagator"

    def set_control(self,theta):
        self.theta = theta
        c = np.cos(self.theta)
        s = 1j*np.sin(self.theta)
        self.kronecker.K = np.array(((c,s),(s,c)),dtype=complex)

    def apply(self,v,u):
        self.kronecker.apply(v,u)

    def as_matrix(self):
        return self.kronecker.as_matrix()
