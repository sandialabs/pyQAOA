from qaoa.operators import LinearOperator
from qaoa.util.math import apply_kron2
import numpy as np

class Kronecker(LinearOperator):

    """
    Kronecker product of a 2x2 matrix with itself n times
    """
 
    def __init__(self,K,nq,dtype=float):
        super().__init__(nq)
        self.K = K
        self.dtype = dtype
        self.work = np.zeros(1<<nq,dtype=self.dtype)

    def compute(self,f,v,u):
        if self.nq % 2: # Odd number of stages
            f(0,v,u)
            for k in range(1,self.nq):
                if k % 2: # Odd stage
                    f(k,u,self.work)
                else:
                    f(k,self.work,u) # Even stage
        else: # Even number of stages       
            f(0,v,self.work)
            for k in range(1,self.nq):
                if k % 2: # Odd stage
                    f(k,self.work,u)
                else:
                    f(k,u,self.work)

    def apply(self,v,u):
        f = lambda k,x,y : apply_kron2(self.K,self.nq,k,x,y)
        self.compute(f,v,u)

    def apply_adjoint(self,v,u):
        f = lambda k,x,y : apply_kron2(np.conj(self.K.T),self.nq,k,x,y)
        self.compute(f,v,u)

    def apply_inverse(self,v,u):
        Ki = np.linalg.inv(K)
        f = lambda k,x,y : apply_kron2(Ki,self.nq,k,x,y)
        self.compute(f,v,u)

    def apply_adjoint_inverse(self,v,u):
        Ki = np.linalg.inv(K)
        f = lambda k,x,y : apply_kron2(np.conj(Ki.T),self.nq,k,x,y)
        self.compute(f,v,u)

    def as_matrix(self):
        otimes = lambda A0,*A : np.kron(A0,otimes(*A)) if len(A) else A0
        return otimes(*([self.K]*self.nq))
