from qaoa.operators import UnitaryOperator, HermitianOperator

class Propagator(UnitaryOperator):

    """
    Base class for unitary operators that are parameterized by a scalar control
    """

    def __init__(self,A,theta=0):
        
        """
        Constructor for generic propagator of the form exp(i*theta*A)

        Parameters
        ----------
        A : qaoa.operators.HermitianOperator
          The Hermitian operator that generates this propagator
        theta : float, optional
          The initial control angle
        """
        assert( isinstance(A,HermitianOperator) )
        self.A = A
        self.theta = theta
        super().__init__(A.nq)

    def __str__(self):
        return "Propagator"

    def set_control(self,theta):
        """
        Set the control angle
  
        Parameters
        ----------
        theta : float
          The control angle in the propagator exp(i*theta*A)
        """
        self.theta = theta

    def get_operator(self):
        """
        Get the Hermitian operator that generates this propagator
        
        Returns
        -------
        A : qaoa.operators.HermitianOperator
        """
        return self.A

    def as_matrix(self):
        """
        Compute the complex matrix exponential of the Hermitian matrix A
        using scipy.linalg.expm. Operators derived from HermitianOperator
        will often have a special form that makes evaluating this matrix
        by this method unnecessary.
         
        Returns 
        -------
        U : numpy.ndarray 
          A 2D complex-valued array containing exp(i*theta*A) 
        """
        from scipy.linalg import expm
        return expm(1j*self.theta*self.A.as_matrix())

    def check_apply(self,v=None,tol=1e-8):
        import numpy as np
        if v is None:
            v = np.random.randn(self.length) + \
                1j*np.random.randn(self.length) 
        else:
            assert isinstance(v.dtype,complex)
        Av = np.ndarray(self.length,dtype=complex)
        self.apply(v,Av)
        res = self.as_matrix() @ v - Av
        rnorm = np.linalg.norm(res)
        vnorm = np.linalg.norm(v)
        return rnorm < tol * vnorm 

#    def finite_difference_check(self,delta=1e-4,v=None):
#        import numpy as np
#        if v is None:
#            v = np.random.randn(self.length) + \
#                1j*np.random.randn(self.length) 
#        Uv = np.ndarray(self.length,dtype=complex)
#        Av = np.ndarray(self.length,dtype=complex)
#        self.A.apply(v,Av)
#        theta_old = np.copy(self.theta)
#        self.set_control(delta)
#        self.apply(v,Uv)
#        residual = ( Av-(Uv-v)/(1j*delta) )
#        self.set_control(theta_old)
#        return np.linalg.norm(residual)

