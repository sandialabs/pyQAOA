import qaoa
import numpy as np
from scipy.optimize import minimize
from random import sample

def projectors(p,d):
    n = [k for k in range(2*p)]
    s = sample(n[::2],d) + sample(n[1::2],d)
    c = list(set(n).difference(set(s)))
    I = np.eye(2*p)
    return I[:,s],I[:,c]


if __name__ == '__main__':

    np.set_printoptions(precision=4,linewidth=200)

    nq = 12 # Number of Qubits
    p = 10   # Number of Layers
    d = 3   # Layer Subspace size 

    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=p)
    true_min = obj.true_minimum()

    z = np.zeros(2*d)
    theta = np.random.rand(2*p)*np.pi
    opts = { 'gtol' : np.sqrt(np.finfo(float).eps) }

    for k in range(100):

        P1,P2 = projectors(p,d)

        # Restricted objective, gradient, and Hessian
        f = lambda x1,x2 : obj.value(P1 @ x1 + P2 @ x2)
        g = lambda x1,x2 : P1.T @ obj.gradient(P1 @ x1 + P2 @ x2)
        H = lambda x1,x2 : np.array([obj.hessVec(P1 @ x1 + P2 @ x2,e) for e in P1.T]) @ P1
    
        f0 = f(P1.T@theta,P2.T@theta)
        g0 = g(P1.T@theta,P2.T@theta)
        H0 = H(P1.T@theta,P2.T@theta)

#        m = lambda s : f0 + g0 @ s + s.dot(H0 @ s)/2

#        E,V = np.linalg.eig(H0)

#        print(f0)
#        print(g0)
#        print(H0)
#        print(E)
#        print(V)
#        print(V.T @ g0)
        result = minimize( f, P1.T @ theta, args=(P2.T @ theta,), \
                           method="trust-exact", jac=g, hess=H, options=opts )
        #print(result)
        if result['success']:
            x = result['x']
            print(k,result['fun']/true_min)
            theta = P1 @ x + P2 @ P2.T @ theta
        
#    v = np.pi*np.random.rand(2*p)  
#    h = np.array([10**(-1-j) for j in range(10)])
#    
#    print(obj.check_gradient(theta,h,v))
#    print(obj.check_hessVec(theta,h,v))

