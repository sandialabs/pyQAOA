from qaoa.operators import HermitianOperator
from qaoa.util.math import projection_1d

class ProjectionOperator(HermitianOperator):

    def __init__(self,v):
        import numpy as np
        self.v = v
        nq = int(numpy.log2(len(v)))
        super().__init__(nq)
 
    def as_matrix(self):
        return np.outer(self.conj(v),v)
       
    def apply(self,x,Px):
        projection_1d(self.v,x,Px)

    def apply_inverse(self,x,Px):
        raise NotImplementedError("Projection operators have no inverse")
