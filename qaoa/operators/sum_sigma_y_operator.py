from qaoa.operators import HermitianOperator
from qaoa.util.math import sum_sigma_y_mult, sum_sigma_y_inner_product, sum_sigma_y_conj_inner_product

class SumSigmaYOperator(HermitianOperator):

    def __init__(self,nq):
        super().__init__(nq)

    def __deepcopy__(self,memo):
        return SumSigmaYOperator(self.nq)

    def __str__(self):
        return "SumSigmaYOperator"

    def propagator(self,theta=0):
        from qaoa.operators import SumSigmaYPropagator
        return SumSigmaYPropagator(self,theta)

    def as_matrix(self):
        import numpy as np
        otimes = lambda A0,*A: np.kron(A0,otimes(*A)) if len(A) else A0
        sy = np.array(((0,-1j),(1j,0)))
        I = lambda k : np.eye(1<<k)
        D = sum( otimes(I(k),sy,I(self.num_qubits()-k-1)) for k in range (self.num_qubits()) )
        return D

    def apply(self,v,Dv):
        sum_sigma_y_mult(self.nq,v,Dv)


