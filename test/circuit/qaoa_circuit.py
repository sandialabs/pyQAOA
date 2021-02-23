import numpy as np
import qaoa
from qaoa.circuit.validation import *
from qaoa.util.number_format import spaced_decim

from scipy.optimize import minimize
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.set_printoptions(precision=4,linewidth=200)

    nq = 12 # Number of Qubits
    p = 3   # Number of Layers

    # Create a QAOA Max Cut objective from a previously generated graph
    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=p)

    # Initial guess
    theta  = np.random.rand(2*p)*np.pi
    dtheta = np.random.rand(2*p)*np.pi
    delta  = np.logspace(0,-13,14)

    res_g  = check_gradient(obj,theta,dtheta,delta,1)
    res_hv = check_hess_vec(obj,theta,dtheta,delta,1)

    print("\nValidating objective function\n")

    print('\n  Finite difference error as a function of step size: \n')
    print('  +---------+------------------------+------------------------+')
    print('  | h       | Directional Derivative | Hessian-Vector Product |')
    print('  +---------+------------------------+------------------------+')
    for k,h in enumerate(delta):
        print('  | {0}| {1}| {2}|'.format( spaced_decim(h,           1,  8 ), \
                                            spaced_decim(res_g[k],  16, 23 ), \
                                            spaced_decim(res_hv[k], 16, 23 )))
    print('  +---------+------------------------+------------------------+')

    true_min = obj.true_minimum()
    print("\nObjective function true minimum: ",true_min)

    # Initial guess
    theta = np.random.rand(2*p)*np.pi
    dtheta = np.random.rand(2*p)*np.pi


    # Solver options for scipy.optimize.minimize
    opts = { 'gtol' : np.sqrt(np.finfo(float).eps), 'disp':True }

    # Compute a local minimizer 
    result = minimize( obj.value, theta, method="trust-exact", 
                       jac=obj.gradient, hess=obj.hessian, options=opts )
    H = result["hess"]

    E,V = np.linalg.eig(H)
    
    print("Initial control angles:",theta) 
    print("Optimal control angles:",result['x']) 
    print("Hessian eigenvalues at terminal vector:")
    print(E)
