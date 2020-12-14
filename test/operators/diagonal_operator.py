import qaoa
import numpy as np


def check_DiagonalOperator_apply(num_qubits,tol):
    m = 1 << num_qubits
    d = np.random.rand(m) 
    v = np.random.randn(m)
    D = qaoa.operators.DiagonalOperator(d)
    assert( D.check_apply(v,tol) )

def test_DiagonalOperator():
    check_DiagonalOperator_apply(8,1e-8)

if __name__ == '__main__':
    test_DiagonalOperator()

