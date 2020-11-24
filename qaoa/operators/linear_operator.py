from qaoa.util.types import is_squarematrix, is_callable

class LinearOperator(object):
    """
    Defines the interface of generic linear operators that can be applied to vectors
    """

    def __init__(self,nq):
        self.nq = nq
        self.length = 1 << nq
 
    def __str__(self):
        return "LinearOperator"
   
    def __len__(self):
        return self.length

    def apply(self,v,Av):
        raise NotImplementedError("Derived class does not override apply() method")

    def apply_adjoint(self,v,Av):
        raise NotImplementedError("Derived class does not override apply_adjoint() method")

    def apply_inverse(self,v,Av):
        raise NotImplementedError("Derived class does not override apply_inverse() method")

    def apply_adjoint_inverse(self,v,Av):
        raise NotImplementedError("Derived class does not override apply_adjoint_inverse() method")

    def as_matrix(self):
        """
        Return a 2D NumPy array of the operator in matrix form. This is extremely wasteful and should
        only be used for validation purposes.
        """
        raise NotImplementedError("Derived class does not override as_matrix() method")

    def propagator(self,theta=0):
        from qaoa.operators import Propagator
        return Propagator.create(self,theta)

    def num_qubits(self):
        return self.nq

    def test(self,method,A,v,Av):
        assert( is_squarematrix(A) or is_callable(A) )

        import numpy
        dtype = type(v[0])
        eps = numpy.finfo(dtype).eps
        tol = numpy.sqrt(len(v))
        u = A.dot(v) if is_squarematrix(A) else A(v)
        method(v,Av)
        error = numpy.linalg.norm(Av-u)
        return error, tol        

    def test_apply(self,A,v,Av):
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
