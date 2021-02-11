import numpy
import networkx
import numbers

def is_nparray(x):
    return isinstance(x,numpy.ndarray)

def is_squarematrix(x):
    if is_nparray(x):
        return x.shape[0] == x.shape[1] if len(x.shape) == 2 else False
    else:
        return False

def is_callable(x):
    return hasattr(x,'__call__')

def is_container(x):
    return hasattr(x,"__getitem__")

def complex_type(rtype):
    return type(1j*rtype())

def real_type(ctype):
    return type(numpy.real(ctype()))

def is_float(x):
    xtype = type(x)
    return numbers.Real.__subclasscheck__(xtype) and not \
           numbers.Integral.__subclasscheck__(xtype)

def is_integer(x):
    return isinstance(x,numbers.Integral)

def is_graph(x):
    if hasattr(x,'edges'):
        if isinstance(x.edges,networkx.classes.reportviews.EdgeView):
            return True
    return False

def is_weightedgraph(x):
    return all( "weight" in x[e[0]][e[1]] for e in x.edges ) if is_graph(x) else False
      
def is_rowvalue(x):
    return all( is_integer(y[0]) for y in x ) if is_container(x) and len(x[0])==2 else False

def is_rowcolvalue(x):
    return all( is_integer(y[0]) for y in x ) if is_container(x) and len(x[0])==3 else False
