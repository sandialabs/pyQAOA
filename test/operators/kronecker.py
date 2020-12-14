import qaoa
import numpy as np

def check_Kronecker_apply(num_qubits,tol):
    m = 1 << num_qubits
    A = np.random.randn(2,2)
    K = qaoa.operators.Kronecker(A,num_qubits)
    v = np.random.randn(m)
    assert( K.check_apply(v,tol) )

def test_Kronecker():
    check_Kronecker_apply(8,1e-8)

if __name__ == '__main__':
    test_Kronecker()
