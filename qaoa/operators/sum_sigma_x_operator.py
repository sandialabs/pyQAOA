from qaoa.operators import HermitianOperator
from numba import njit, prange

@njit(parallel=True)
def sum_sigma_x_mult(n,v,Dv):
    for j in prange(1<<n):
        Dv[j] = 0
        for k in prange(n):
            Dv[j] += v[j^(1<<k)]

@njit(parallel=True)
def sum_sigma_x_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for k in prange(n):
            lresult += v[j^(1<<k)]
        result += u[j]*lresult
    return result

@njit(parallel=True)
def sum_sigma_x_conj_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for k in prange(n):
            lresult += v[j^(1<<k)]
        result += np.conj(u[j])*lresult
    return result


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

    def inner_product(self,u,v):
        return sum_sigma_x_inner_product(self.nq,u,v)

    def conj_inner_product(self,u,v):
        return sum_sigma_x_conj_inner_product(self.nq,u,v)

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
        sum_sigma_x_mult(self.nq,v,Dv)          
