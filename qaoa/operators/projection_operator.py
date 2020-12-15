from qaoa.operators import HermitianOperator
from qaoa.util.math import projection_1d, inner_product, conj_inner_product
import numpy as np

class ProjectionOperator(HermitianOperator):

    def __init__(self,v=None,nq=None):
        """
        Create a projection operator for an nq-qubit cirquit
 
        Parameters
        ----------
        v : numpy.ndarray, optional
            A vector of length 2**nq. Default is a vector of 1's (normalized)
        nq : unsigned int, optional
            Number of qubits. Will create the plus-state projector if specified instead on nq

        Note
        ----
        Either or v or nq must be provided, but not both.
        """
        assert(v is None or nq is None)
        self.v = v if v is not None else np.ones(1<<nq)
        self.v /= np.linalg.norm(v)
        nq = int(np.log2(len(self.v)))
        super().__init__(nq)

    def inner_product(self,x,y):
        return inner_product(x,self.v) * inner_product(self.v,y)

    def conj_inner_product(self,x,y):
        return conj_inner_product(x,self.v) * conj_inner_product(self.v,y)

    def expectation(self,x):
        return np.abs(conj_inner_product(self.v,x))**2

    def propagator(self,theta=0):
        from qaoa.operators import ProjectionPropagator
        return ProjectionPropagator(self,theta)

    def as_matrix(self):
        return np.outer(self.conj(v),v)
       
    def apply(self,x,Px):
        projection_1d(self.v,x,Px)

    def apply_inverse(self,x,Px):
        raise NotImplementedError("Projection operators have no inverse")
