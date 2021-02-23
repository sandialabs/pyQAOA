from qaoa.util.finite_difference import check_derivative
import numpy

def check_gradient(obj,theta,dtheta,h,order=1):
    """
    Evaluate the discrepency of a circuit objective's first
    directional derivative using finite differences

    Using an m-point finite difference approximation with offsets d1,..,dm
    and weights w1,...,wm, the directional derivative is approximated by 

    .. math:

    f_{\\h\\theta}(\theta) = h^{1-m} \sum\limits_{k=1}^m w_k f(\\theta+d_k h\\h\\theta) + O(h^{m-1}) = \\bar f_{\\h\\theta}(\theta) + r(h)
    
    For the gradient to correctly correspond to the objective value, the
    residual norm must decrease at the appropriate power for at least some
    range of step sizes. The error will ultimately increase at small step sizes
    due to numerical round-off.

    Parameters
    ----------

    obj : qaoa.circuit.QuantumCircuit
        A quantum circuit objective function that provides value
        and gradient methods
    theta : numpy.ndarray
        A vector of control angles corresponding to each of the circuit
        stages
    dtheta : numpy.ndarray
        A vector of differential changes to control angles
    h : real scalar, array-like, or iterable 
        Finite difference step size(s) to use in approximating the
        first directional derivative
    order : unsigned int
        Finite difference approximation order.  

    Returns
    -------
    
    rnorm : same type as h
        Residual norm 

    Example
    -------

    >>> obj = qaoa.circuit.load_maxcut()
    >>> theta = numpy.random.randn(obj.num_stages)
    >>> dtheta = numpy.random.randn(obj.num_stages)
    >>> nsteps = 10
    >>> h = [(10)**(-k) for k in range(nsteps)]
    >>> res = check_gradient(obj,theta,dtheta,h)
    >>> for k,r in enumerate(res):
    ...     print(h[k],r)
    1 7.404420469731904
    0.1 0.487594551752089
    0.01 0.07499866545689926
    0.001 0.008702780633479179
    0.0001 0.0008822235938517053
    1e-05 8.834160148474268e-05
    1e-06 8.833390358731208e-06
    1e-07 8.714259376318978e-07
    1e-08 6.762446780328446e-08
    1e-09 1.2888697948909567e-06

    """
    
    theta_old  = numpy.copy(obj.get_control())
    dtheta_old = numpy.copy(obj.get_differential_control())
    rnorm =  check_derivative(obj.value,\
                                dtheta.dot(obj.gradient(theta)), \
                                theta,dtheta,h,order)
    obj.set_control(theta_old)
    obj.set_differential_control(dtheta_old)
    return rnorm

def check_hess_vec(obj,theta,dtheta,h,order=1):
    theta_old  = numpy.copy(obj.get_control())
    dtheta_old = numpy.copy(obj.get_differential_control())
    
    rnorm = check_derivative(obj.gradient,\
                               obj.hess_vec(theta,dtheta), \
                               theta,dtheta,h,order)
    obj.set_control(theta_old)
    obj.set_differential_control(dtheta_old)
    return rnorm

def check_dpsi(obj,theta,dtheta,h,order=1):
    theta_old  = numpy.copy(obj.get_control())
    dtheta_old = numpy.copy(obj.get_differential_control())

    obj.set_control(theta)
    obj.set_differential_control(dtheta)
    obj.stage[-2].dpsi()
    dpsi = numpy.copy(obj.dpsi)

    def psi(x):
        obj.set_control(x)
        obj.stage[-2].psi()
        return numpy.copy(obj.psi)

    rnorm = check_derivative(psi,dpsi,theta,dtheta,h,order)
    obj.set_control(theta_old)
    obj.set_differential_control(dtheta_old)
    return rnorm    

def check_dlam(obj,theta,dtheta,h,order=1):
    theta_old  = numpy.copy(obj.get_control())
    dtheta_old = numpy.copy(obj.get_differential_control())

    obj.set_control(theta)
    obj.set_differential_control(dtheta)
    obj.stage[1].dlam()
    dlam = numpy.copy(obj.dlam)

    def lam(x):
        obj.set_control(x)
        obj.stage[1].lam()
        return numpy.copy(obj.lam)

    rnorm = check_derivative(lam,dlam,theta,dtheta,h,order)
    obj.set_control(theta_old)
    obj.set_differential_control(dtheta_old)
    return rnorm    





