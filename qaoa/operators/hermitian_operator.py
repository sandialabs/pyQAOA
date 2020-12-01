from qaoa.operators import LinearOperator
from qaoa.util.math import inner_product, conj_inner_product
import numpy as np

class HermitianOperator(LinearOperator):

    def __init__(self,nq,work=None):
        super().__init__(nq)
        self.work = work if work is not None else np.zeros(1<<nq,dtype=complex)
    
    def __str__(self):
        return "HermitianOperator"
 
    def true_minimum(self):
        raise NotImplementedError("Derived type does not implement true_minimum() method")

    def true_maximum(self):
        raise NotImplementedError("Derived type does not implement true_maximum() method")

    def inner_product(self,u,v):
        self.apply(v,self.work[:])
        return inner_product(u,self.work)

    def conj_inner_product(self,u,v):
        self.apply(v,self.work[:])
        return conj_inner_product(u,self.work)

    def expectation(self,v):
        return np.real(self.conj_inner_product(v,v))

    def apply_adjoint(self,v,Hv):
        self.apply(v,Hv)

    def apply_adjoint_inverse(self,v,Hv):
        self.apply_inverse(v,Hv)
