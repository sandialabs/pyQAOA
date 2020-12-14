from qaoa.util.types import is_squarematrix, is_callable
import abc

class OperatorDocumentation(type):
    """
    Provides automatic inheritance of docstrings for methods not overridden by derived classes  
    """

    def __new__(metaclass, classname, bases, classdict):
        cls = super().__new__(metaclass, classname, bases, classdict)
        for k,v in classdict.items():
            if not getattr(v,'__doc__') and v in dir(super()):
                v.__doc__ = getattr(bases[-1],k).__doc__
        return cls

class LinearOperator(object,metaclass=OperatorDocumentation):
    """
    Defines the interface of generic linear operators that can be applied to vectors

    The current design assumes that the matrix representation of linear operators is square

    Attributes
    ----------
    
    nq : unsigned int
        The number of qubits this operator is defined for
    length : unsigned int
        The size of the domain and range spaces of this operator (2**nq)
    dtype : type
        Data type associated with the operator
    """

    def __init__(self,nq,dtype=float):
        """
        Constructor for an abstract (square) linear operator. 

        This class defines the general operator interface and is never actually instantiated. 
       
        Parameters
        ----------
        nq : unsigned int  
            The number of qubits on which to define this operator
        dtype : type (optional)
            Data type associated with the operator. Set to float if not provided
        """
 
        self.nq = nq
        self.length = 1 << nq 
  
    def __str__(self):
        return "LinearOperator"
   
    @abc.abstractmethod
    def __len__(self):
        """
        Dimension of the domain and range space

        Returns
        -------
        length : unsigned int
            The length of a vector that is an element of the domain or range space of this operator
        """
        return self.length

    @abc.abstractmethod
    def apply(self,v,Av):
        """
        Apply a linear operator to a vector

        Parameters
        ----------
        v : numpy.ndarray 
            Domain vector of length 2**nq
        Av : numpy.ndarray 
            Range vector of length 2**nq. Modified in-place

        """
        raise NotImplementedError("Derived class does not override apply() method")

    @abc.abstractmethod
    def apply_adjoint(self,v,Av):
        """
        Apply the adjoint of a linear operator to a vector

        Parameters
        ----------
        v : numpy.ndarray 
            Domain vector of length 2**nq
        Av : numpy.ndarray 
            Range vector of length 2**nq. Modified in-place

        """
        raise NotImplementedError("Derived class does not override apply_adjoint() method")

    @abc.abstractmethod
    def apply_inverse(self,v,Av):
        """
        Apply the inverse of a linear operator to a vector

        Parameters
        ----------
        v : numpy.ndarray 
            Domain vector of length 2**nq
        Av : numpy.ndarray 
            Range vector of length 2**nq. Modified in-place

        """
        raise NotImplementedError("Derived class does not override apply_inverse() method")

    @abc.abstractmethod
    def apply_adjoint_inverse(self,v,Av):
        """
        Apply the adjoint inverse of a linear operator to a vector

        Parameters
        ----------
        v : numpy.ndarray 
            Domain vector of length 2**nq
        Av : numpy.ndarray 
            Range vector of length 2**nq. Modified in-place

        """
        raise NotImplementedError("Derived class does not override apply_adjoint_inverse() method")

    @abc.abstractmethod
    def as_matrix(self):
        """
        Return a 2D NumPy array of the operator in matrix form. 

        Note
        ----
        This method is for testing and validation purposes only as the cost of forming the operator
        as a matrix is computationally expensive and unnecessary for simulation. 
        
        Returns
        -------
        M : numpy.ndarray
            Matrix of shape (2**nq,2**nq) that would be obtained by applying this operator to the 
        identity matrix.
        """
        raise NotImplementedError("Derived class does not override as_matrix() method")

    @abc.abstractmethod
    def check_apply(self,v=None,tol=1e-8):
        """
        Compare the action of the apply method against matrix multiplication 

        Parameters
        ----------
        v : numpy.ndarray (optional)
            Domain vector on which to check the apply method. If not supplied, a 

        Returns
        -------
        result : bool
            Check passes if norm(Av-A@v) < tol * norm(v), where Av is the range vector 
        produced by apply(v,Av) and A is the matrix returned by the method as_matrix()
                
        """

        import numpy as np
        
        if v is None:
            v = np.random.randn(self.length)
        Av = np.ndarray(self.length,dtype=v.dtype)
        self.apply(v,Av)
        res = self.as_matrix() @ v - Av
        rnorm = np.linalg.norm(res)
        vnorm = np.linalg.norm(v)
        return rnorm < tol * vnorm


    @abc.abstractmethod
    def num_qubits(self):
        """
        Number of qubits for which this operator is defined

        Returns
        -------
        nq : unsigned int
            The number of qubits for which this operator is defined

        """
        return self.nq

