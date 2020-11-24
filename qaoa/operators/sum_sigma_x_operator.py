from qaoa.operators import HermitianOperator
from qaoa.util.math import sum_sigma_x_mult 

class SumSigmaXOperator(HermitianOperator):

    def __init__(self,nq):
        super().__init__(nq)

    def __str__(self):
        return "SumSigmaXOperator"

    def as_matrix(self):
        import numpy as np
        otimes = lambda A0,*A: np.kron(A0,otimes(*A)) if len(A) else A0
        sx = np.array(((0,1),(1,0)))
        I = lambda k : np.eye(1<<k)
        D = sum( otimes(I(k),sx,I(self.num_qubits()-k-1)) for k in range(self.num_qubits()) )
        return D
    
    def apply(self,v,Dv):
        sum_sigma_x_mult(self.nq,v,Dv)          
