import qaoa
import numpy as np
from scipy.optimize import minimize

if __name__ == '__main__':

    np.set_printoptions(precision=4,linewidth=200)

    nq = 8 # Number of Qubits
    p = 2  # Number of Layers

    # Create circuit objective using graph from library
    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=p)

    print("True Minimum: ", obj.true_minimum())



    L = np.sqrt(2)*p*np.pi
    opts = { 'gtol' : np.sqrt(np.finfo(float).eps) }

    print("SciPy result")

    for k in range(10):
        # Random initial guess
        theta = np.pi*np.random.rand(2*p)
        result = minimize( obj.value, theta, args=(), method="trust-exact", 
                           jac=obj.gradient, hess=obj.hessian, options=opts )
        print(result)                      
#    v = np.pi*np.random.rand(2*p)  
#    h = np.array([10**(-1-j) for j in range(10)])
#    
#    print(obj.check_gradient(theta,h,v))
#    print(obj.check_hessVec(theta,h,v))

