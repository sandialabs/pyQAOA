import qaoa
import numpy as np

def test_SumSigmaXOperator():
    
    exprs = ["otimes = lambda A0,*A: np.kron(A0,otimes(*A)) if len(A) else A0",
             "sx = np.array(((0,1),(1,0)))",
             "I = lambda k : np.eye(1<<k)",
             "X = sum( otimes(I(k),sx,I(nq-k-1)) for k in range(nq))"]

    qaoa.operators.test(op_type="SumSigmaXOperator",cargs="nq",exact="lambda v : X.dot(v)",exprs=exprs)


if __name__ == '__main__':

    test_SumSigmaXOperator()
