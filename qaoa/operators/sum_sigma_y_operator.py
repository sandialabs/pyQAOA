from qaoa.operators import HermitianOperator
from numba import njit, prange

@njit(parallel=True)
def sum_sigma_y_mult(n,v,Dv):
    for j in prange(1<<n):
        Dv[j] = 0
        for l in prange(n):
            k = j ^ (1<<l)
            s = 1j if j>k else -1j
            Dv[j] += s*v[k]

@njit(parallel=True)
def sum_sigma_y_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for l in prange(n):
            k = j ^ (1<<l)
            s = 1j if j>k else -1j
            lresult += s*v[j^(1<<k)]
        result += u[j]*lresult
    return result

@njit(parallel=True)
def sum_sigma_y_conj_inner_product(n,u,v):
    result = 0
    for j in prange(1<<n):
        lresult = 0
        for l in prange(n):
            k = j ^ (1<<l)
            s = 1j if j>k else -1j
            lresult += s*v[j^(1<<k)]
        result += np.conj(u[j])*lresult
    return result



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


