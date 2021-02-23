import numpy as np

def weights(N,p):
    """ Compute the weights for all (uniform) N-point
        finite difference approximations to the pth derivative """

    M = np.array([[ (i**j) for i in range(1-N,N) ] \
                           for j in range(N)])
    G = np.diag(np.cumprod([1]+[j for j in range(1,N)]))
    return np.flipud([np.linalg.solve(M[:,k:(k+N)],G[p]) for k in range(N)])


def stencil_weights( N, p=1, tol=1e-13 ):
    W = weights(N,p)
    d = int(N//2)
    h = [k-d for k in range(N)] 
    h,w = zip(*[ (h[k],w) for k,w in enumerate(W[d]) if abs(w)>tol ])
    return h,w

def check_derivative(fun,exact,x,v,delta,order=1):

    import numpy as np
    from numpy.linalg import norm

    d,w = stencil_weights(order+1)

    phi = lambda h : fun(x+h*v)
    df = lambda h : sum( fun(x+j*h*v)*w[k] for k,j in enumerate(d) )/h

    if isinstance(delta,np.ndarray):
        assert np.all(np.isreal(delta))
        return np.array([ norm(df(h)-exact) for h in delta ])

    elif hasattr(delta,'__getitem__') and hasattr(delta,'__len__'):
        assert np.all([np.isreal(delta[k]) for k in range(len(delta))])
        return [ norm(df(h)-exact) for h in delta ]

    elif np.isscalar(delta):
        assert np.isreal(delta)
        return norm(df(h)-exact)

    else:
        raise TypeError("Finite difference step delta must either be a real-valued scalar, an array-like container of real scalars, or an iterable that yields real scalars.")
