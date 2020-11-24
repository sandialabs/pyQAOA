import qaoa
import numpy as np

def test_DiagonalOperator_apply():
    exprs = ["d = np.random.randn(1<<nq)",]
    qaoa.operators.test(op_type="DiagonalOperator",cargs="d",exact="lambda v:d*v",exprs=exprs)

def test_DiagonalOperator_apply_adjoint():
    exprs = ["d = np.random.randn(1<<nq)",]
    qaoa.operators.test(op_type="DiagonalOperator",cargs="d",exact="lambda v:d*v",method="apply_adjoint",exprs=exprs)


if __name__ == '__main__':

    test_DiagonalOperator_apply()
    test_DiagonalOperator_apply_adjoint()

