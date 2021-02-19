from qaoa.operators import HermitianOperator
from qaoa.util.math import sum_sigma_x_mult, sum_sigma_x_inner_product, sum_sigma_x_conj_inner_product

class SumSigmaXOperator(HermitianOperator):

    def __init__(self,nq):
        """
        Create a SumSigmaXOperator for nq qubits
        
        Parameters
        ----------
        nq : unsigned int
            Number of qubits
        """
        super().__init__(nq)

    def __str__(self):
        return "SumSigmaXOperator"

    def __deepcopy__(self,memo):
        return SumSigmaXOperator(self.nq) 

    def inner_product(self,u,v):
        return sum_sigma_x_inner_product(self.num_qubits(),u,v)

    def conj_inner_product(self,u,v):
        return sum_sigma_x_conj_inner_product(self.num_qubits(),u,v)

    def propagator(self,theta=0):
        from qaoa.operators import SumSigmaXPropagator
        return SumSigmaXPropagator(self,theta)

    def as_matrix(self):
        import numpy as np
        otimes = lambda A0,*A: np.kron(A0,otimes(*A)) if len(A) else A0
        sx = np.array(((0,1),(1,0)))
        I = lambda k : np.eye(1<<k)
        D = sum( otimes(I(k),sx,I(self.num_qubits()-k-1)) for k in range(self.num_qubits()) )
        return D
    
    def apply(self,v,Dv):
        sum_sigma_x_mult(self.num_qubits(),v,Dv)          
