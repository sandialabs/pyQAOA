import qaoa
import numpy as np

#def test_DiagonalPropagator():
if __name__ == '__main__':
#    exprs = ["d = np.random.randn(1<<nq)",
#             "theta = np.pi*np.random.randn(1)"]
#    dv = "np.random.randn(1<<nq) + 1j * np.random.randn(1<<nq)"
#    rv = "np.zeros(1<<nq,dtype=complex)"
#
#    qaoa.operators.test(op_type="DiagonalPropagator",cargs="d,theta",
#                        exact="lambda v: np.exp(1j*theta*d)*v",
#                        dv=dv, rv=rv, exprs=exprs)
    nq = 4
    d = np.random.randn(1<<nq)
    v = np.random.randn(1<<nq) + 1j * np.random.rand(1<<nq)
    u = np.zeros(1<<nq,dtype=complex)

    theta = np.sqrt(2)
    D = qaoa.operators.DiagonalOperator(d)
    Ud = qaoa.operators.generate_propagator(D) #DiagonalPropagator(D,theta)
#
#    # Test apply()
    Ud.set_control(theta)
    Ud.apply(v,u)
    error = np.linalg.norm(u-np.exp(1j*theta*d)*v)
    print("apply error: ",error)

    # Test apply_adjoint()
    Ud.apply_adjoint(v,u)
    error = np.linalg.norm(u-np.exp(-1j*theta*d)*v)
    print("apply_adjoint error: ",error)



