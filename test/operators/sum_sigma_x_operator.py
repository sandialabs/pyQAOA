import qaoa
import numpy as np

def test_SumSigmaXOperator_apply():
    nq = 8
    m = 1<<nq
    v = np.random.randn(m)
    Xv = np.random.randn(m)
    X = qaoa.operators.SumSigmaXOperator(m)
    X.apply(v,Xv)
    err = Xv - X.as_matrix() @ v
    assert(np.linalg.norm(err)<1e-8)      

if __name__ == '__main__':

    test_SumSigmaXOperator_apply()

