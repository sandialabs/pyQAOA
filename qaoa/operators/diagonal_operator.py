from qaoa.operators import HermitianOperator
from qaoa.util.math import hadamard_mult, hadamard_div, hadamard_conj_mult, hadamard_conj_div

class DiagonalOperator(HermitianOperator):

    def __init__(self,d):
        import numpy
        assert(isinstance(d,numpy.ndarray) and len(d.shape)==1)
        self.data = d
        super().__init__(int(numpy.log2(len(d))))
 
    def __str__(self):
        return "DiagonalOperator"

    def true_maximum(self):
        import numpy
        return numpy.max(self.data)

    def true_minimum(self):
        import numpy
        return numpy.min(self.data)

    def as_matrix(self):
        """
        Returns a 2D NumPy array form of this operator. NOTE: This method should be used for 
        verification purposes only as it is computationally expensive.
        """
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
