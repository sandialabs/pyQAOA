from qaoa.operators import HermitianOperator
from qaoa.util.math import projection_1d
import numpy as np

class ProjectionOperator(HermitianOperator):

    def __init__(self,v=None,nq=None):
        self.v = v if v is not None else np.ones(1<<nq)/np.sqrt(1<<nq)
        nq = int(np.log2(len(self.v)))
        super().__init__(nq)
    
    def propagator(self,theta=0):
        from qaoa.operators import ProjectionPropagator
        return ProjectionPropagator(self,theta)

    def as_matrix(self):
        return np.outer(self.conj(v),v)
       
    def apply(self,x,Px):
        projection_1d(self.v,x,Px)

    def apply_inverse(self,x,Px):
        raise NotImplementedError("Projection operators have no inverse")
