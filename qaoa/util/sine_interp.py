import numpy as np
import scipy
import itertools
import qaoa

class SineInterp(object):
    """ Class for interpolating objective functions using the Discrete Sine Transform 
    """

    def __init__(self,obj,ni):
        """ 
        Evaluate a qaoa.circuit.QuantumCircuit's objective function values on
        a tensor product grid with ni**d points where d is the number of circuit stages
        and ni is the number of interpolation points along eachin dimensional axis        

        Computes the multivariate sine series expansion coefficients of the objective
        on the grid upon construction. THe domain is [0,pi/2]^dim

        Attributes
        ----------
        ni : int
            The number of grid points along each dimensional axis
        dim : int 
            The number of dimensions
        theta : numpy.ndarray
            Uniform tensor product grid on which to sample the objective. Has shape (ni**dim,dim)
        F : numpy.ndarray
            The objective evaluated on the interpolation grid
        Fhat : numpy.ndarray
            The sine series expansion coefficients
  
        """

        assert isinstance(obj,qaoa.circuit.QuantumCircuit)        
        self.ni = ni
        self.dim = len(obj) # Get number of circuit stages (dimension of optimization space)
        self.theta = np.array([ (np.pi/2)*(np.array(k)+1)/(self.ni+1) for k in \
                                itertools.product(range(ni),repeat=self.dim) ])
        self.F = self.to_ncube(np.array([obj.value(theta) for theta in self.theta]))
        self.Fhat = scipy.fft.idstn(self.F,type=1)

    

    def to_ncube(self,v):
        """
        Internal utility function that reshapes a vector of length ni**dim to an ndarray of size (ni,)*dim
        
        Parameters
        ----------
        v : numpy.ndarray
            One dimensional array of length ni**dim 

        Returns
        -------
        numpy.ndarray
            Reshaped array with shape (ni,)**dim 

        """
        return np.reshape(v,(self.ni,)*self.dim)
        

    def interp_grid(self):
        """
        The interpolation point grid as a (dim)-tuple. Compare to numpy.meshgrid

        Returns
        -------
            tuple(numpy.ndarray)
        """
        return tuple(self.to_ncube(theta) for theta in self.theta.T)
        

    def __call__(self,theta):
        """
        Evaluate the interpreted objective at a particular angle configuration

        Parameters
        ----------
        theta : numpy.ndarray
            The location at which to evaluate the interpolated objective

        Returns
        -------
        value : float
            Value of the interpolated objective function      
        
        """
        value = sum( self.Fhat[k] * np.product(np.sin(theta*(np.array(k)+1))) for k in \
                     itertools.product(range(self.Fhat.shape[0]),repeat=self.Fhat.ndim ) )  
        return value
    
    def values(self,ne):
        """
        Evaluate the interpolant on a new uniform tensor product grid with n grid points 
        along each axis
        
        Parameters
        ----------
        ne : int
            Number of evaluation points per dimension of the tensor product grid

        Returns
        -------
        values : numpy.ndarray
            Array of shape (ni,)*dim containing the value of the interpolated objective
        on every point on the grid
        """
        Fh = np.zeros([ne]*self.dim)
        Fh[(slice(self.ni),)*self.dim] = self.Fhat
        values = scipy.fft.dstn(Fh,type=1)
        return values

    @staticmethod
    def nd_grid(d,n):
        import numpy
        return eval('numpy.mgrid['+','.join(['1:{0}'.format(n+1)]*d) + ']*numpy.pi/(2*{0})'.format(n+1),locals())

     






