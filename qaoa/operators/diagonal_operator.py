from qaoa.operators import HermitianOperator
from qaoa.util.math import hadamard_mult, hadamard_div, hadamard_conj_mult, hadamard_conj_div
from numba import njit, prange

@njit(parallel=True)
def diag_inner_product(u,d,v):
    result = 0
    for k in prange(len(d)):
        result += u[k] * d[k] * v[k]
    return result

@njit(parallel=True)
def diag_conj_inner_product(u,d,v):
    result = 0
    for k in prange(len(u)):
        result += np.conj(u[k]) * d[k] * v[k]
    return result


class DiagonalOperator(HermitianOperator):
    """
    Implements a Hermitian operator that can be represented as a diagonal matrix
    """

    def __init__(self,d):

        import numpy

        assert(isinstance(d,numpy.ndarray) and d.ndim==1)
        assert(numpy.isreal(d.dtype))

        self.data = d
        self.length = len(d)
        self.nq = int(numpy.log2(self.length))
        self.dtype = d.dtype
#        super().__init__(int(numpy.log2(len(d))),dtype=d.dtype)
 
    def __str__(self):
        return "DiagonalOperator"

    def true_maximum(self):
        import numpy
        return numpy.max(self.data)

    def true_minimum(self):
        import numpy
        return numpy.min(self.data)

    def propagator(self,theta=0):
        from qaoa.operators import DiagonalPropagator
        return DiagonalPropagator(self,theta)

    def inner_product(self,u,v):
        return diag_inner_product(u,self.data,v)

    def conj_inner_product(self,u,v):
        return diag_conj_inner_product(u,self.data,v)

    def as_matrix(self):
        import numpy 
        return numpy.diag(self.data)
       
    def apply(self,v,Dv):
        hadamard_mult(self.data,v,Dv)   

    def apply_inverse(self,v,Dv):
        hadamard_div(self.data,v,Dv)   

    def apply_adjoint(self,v,Dv):
        hadamard_conj_mult(self.data,v,Dv)   

    def apply_adjoint_inverse(self,v,Dv):
        hadamard_conj_div(self.data,v,Dv)   
