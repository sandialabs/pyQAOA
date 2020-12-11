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

    Attributes
    ----------
    
    nq : unsigned int
        The number of qubits this operator is defined for
    length : unsigned int
        The size of the domain and range spaces of this operator (2**nq)

    """

    def __init__(self,nq):
        """
        Constructor for an abstract (square) linear operator. 

        This class defines the general operator interface and is never actually instantiated. 
       
        Parameters
        ----------
        nq : unsigned int  
            The number of qubits on which to define this operator
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
    def num_qubits(self):
        """
        Number of qubits for which this operator is defined

        Returns
        -------
        nq : unsigned int
            The number of qubits for which this operator is defined

        """
        return self.nq

    @abc.abstractmethod
    def test(self,method,A,v,Av):
        """
        Perform tests on this operator
        """
        assert( is_squarematrix(A) or is_callable(A) )

        import numpy
        dtype = type(v[0])
        eps = numpy.finfo(dtype).eps
        tol = numpy.sqrt(len(v))
        u = A.dot(v) if is_squarematrix(A) else A(v)
        method(v,Av)
        error = numpy.linalg.norm(Av-u)
        return error, tol        

    @abc.abstractmethod
    def test_apply(self,A,v,Av):
        """
        Perform tests on this operator
        """
        return self.test(self.apply,A,v,Av)


def test(op_type,cargs,exact,**kwargs):

    import numpy

    assert( isinstance(op_type,str) )
    assert( isinstance(cargs,str) )

    exec("from qaoa.operators import {0}".format(op_type),locals())
    get = lambda key, default : kwargs[key] if key in kwargs.keys() else default
         
    nqmin = get("nqmin",2)
    nqmax = get("nqmax",8)
    nqstep = get("nqstep",1)
    nqrange = get("nqrange",[nq for nq in range(nqmin,nqmax+1,nqstep)])
    rv = get("rv","numpy.zeros(1<<nq)")
    dv = get("dv","numpy.random.randn(1<<nq)")
    method = get("Method","apply")
    exprs  = get("exprs",list())
    exact = exact.replace("np","numpy")
    w1 = get("w1",16)
    w2 = get("w2",16)

    row = "{:" + str(w1) + "} | {:<" + str(w2) + "}"
    printrow = lambda x,y : print(row.format(x,y))

    exprs = [expr.replace("np","numpy") for expr in exprs]

    tolerance = dict()
    apply_error = dict()
    printrow("\n\nNumber of Qubits","Error")
    print("-"*w1 + "-+-" + "-"*w2)
    
    for nq in nqrange:

        setnq = lambda s : s.replace("nq",str(nq)).replace("np","numpy")
        init = setnq(cargs)
         
        for expr in exprs:
           exec(setnq(expr),locals())

        # Domain vector
        v = eval(setnq(dv),locals())

        # Range vector
        u = eval(setnq(dv),locals())

        construct = "{0}({1})".format(op_type,init)      
        A = eval(construct,locals())

        # Test apply()
        cmd = "A.test_{0}({1},v,u)".format(method,exact)
        error,tol = eval(cmd,locals())

        printrow(nq,error)
        tolerance[nq] = tol
        apply_error[nq] = error

    assert( all(apply_error[nq]<tol for nq,tol in tolerance.items()) )
