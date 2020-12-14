import qaoa
import numpy as np


def check_DiagonalPropagator_apply(num_qubits,tol):
    m = 1 << num_qubits
    d = np.random.rand(m) 
    v = np.random.randn(m)*np.exp(2j*np.pi*np.random.rand(m)[0])
    D = qaoa.operators.DiagonalOperator(d)
    Ud = D.propagator(theta = np.random.rand(1)*np.pi/2)
    assert( Ud.check_apply(v,tol) )

def test_DiagonalPropagator():
    check_DiagonalPropagator_apply(8,1e-8)

if __name__ == '__main__':
    test_DiagonalPropagator()

