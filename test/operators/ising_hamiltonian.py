import qaoa
import numpy as np
import networkx as nx

def test_IsingHamiltonian():

    nq = 6
    h = np.random.randn(nq)
    G = nx.random_regular_graph(3,nq)
    C = qaoa.operators.IsingHamiltonian(graph=G)
    Uc = C.propagator()

    v = np.random.randn(1<<nq) + 1j * np.random.randn(1<<nq)
    u = np.zeros(1<<nq,dtype=complex)
    Uc.set_control(1)
    Uc.apply(v,u)
    exact = np.exp(1j*Uc.get_operator().data)*v
    tol = np.sqrt(1<<nq)*np.finfo(float).eps 

    from numpy.linalg import norm

    assert( np.abs(norm(u)-norm(v))<tol )
    assert( norm(u-exact) < tol )

if __name__ == '__main__':
    test_IsingHamiltonian()
