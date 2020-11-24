import qaoa
import numpy as np

otimes = lambda A0,*A : np.kron(A0,otimes(*A)) if len(A) else A0

def test_Kronecker():

    nqmin = 2
    nqmax = 8
    
    w1 = 16
    w2 = 16
    row = "{:" + str(w1) + "} | {:<" + str(w2) + "}"
    printrow = lambda x,y : print(row.format(x,y))

    tolerance = dict()
    apply_error = dict()

    printrow("\n\nNumber of Qubits","Error")
    print("-"*w1 + "-+-" + "-"*w2)

    for nq in range(nqmin,nqmax+1):

        N = 1 << nq

        # Domain vector
        v = np.random.randn(N)
        u = np.zeros(N)

        # component matrix in Kronecker product
        Ae = np.random.randn(2,2)

        A = otimes(*([Ae,]*nq))

        K = qaoa.operators.Kronecker(Ae,nq)

        error, tol = K.test_apply(A,v,u)

        printrow(nq,error)
        tolerance[nq] = tol
        apply_error[nq] = error

    assert( all(apply_error[nq]<tol for nq,tol in tolerance.items()) )

if __name__ == '__main__':
    test_Kronecker()
