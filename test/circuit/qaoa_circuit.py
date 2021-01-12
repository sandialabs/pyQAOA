import qaoa
import numpy as np
from scipy.optimize import minimize

if __name__ == '__main__':

    np.set_printoptions(precision=4,linewidth=200)

    nq = 12 # Number of Qubits
    p = 3   # Number of Layers

    # Create a QAOA Max Cut objective from a previously generated graph
    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=p)
    true_min = obj.true_minimum()

    # Initial guess
    theta = np.random.rand(2*p)*np.pi

    print("Initial control angles:") 
    print(theta)

    # Solver options for scipy.optimize.minimize
    opts = { 'gtol' : np.sqrt(np.finfo(float).eps) }

    # Compute a local minimizer 
    result = minimize( obj.value, theta, method="trust-exact", 
                       jac=obj.gradient, hess=obj.hessian, options=opts )
    H = result["hess"]

    E,V = np.linalg.eig(H)
    
    print("Hessian eigenvalues at terminal vector:")
    print(E)
