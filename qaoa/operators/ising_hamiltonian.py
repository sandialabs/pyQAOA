from qaoa.operators import DiagonalOperator
import numpy as np

class IsingHamiltonian(DiagonalOperator):

    def graph_terms(self,G):

        from qaoa.util.graph import is_graph, graph_edges
        from numba.typed import List
        from qaoa.util.math import ising_sparse_J

        if self.nq is not None:
            assert(self.nq==len(G))
        else:
            self.nq = len(G)
            self.c = np.zeros(1<<self.nq)

        J = graph_edges(G)
        JL = List()
        [JL.append(e) for e in J]        
        ising_sparse_J(JL,self.c)

    def J_terms(self,J):

        from qaoa.util.types import is_container, is_squarematrix

        if is_squarematrix(J):
            if self.nq is not None:
                assert(self.nq==J.shape[0])
            else:
                self.nq = J.shape[0]
                self.c = np.zeros(1<<self.nq)
            numba_ising_dense_J(J,self.c)

        elif is_container(J):
            assert(self.c is not None)
            from numba.typed import List
            JL = List()
            [JL.append(e) for e in J]
            if len(J[0]) == 2:
                numba_ising_sparse_J(JL,self.c)                      
            else: 
                numba_ising_sparse_weighted_J(JL,self.c)                      
        else:
            raise TypeError("Argument of type {0} is unsupported".format(type(J))) 

    def h_terms(self,h):

        from qaoa.util.types import is_container, is_nparray

        if is_nparray(h):
            print("="*100)
            if self.nq is not None:
                assert(self.nq==len(h))
            else:
                self.nq = len(h)
                self.c = np.zeros(1<<self.nq)

            numba_ising_dense_h(h,self.c)

        elif is_container(h):
            assert(self.c is not None)
            from numba.typed import List
            hL = List()
            [hL.append(e) for e in h]
            numba_ising_sparse(hL,self.c)
   
        else:
            raise TypeError("Argument of type {0} is unsupported".format(type(h))) 
 

    def __init__(self,nq=None,h=None,J=None,graph=None):

        from qaoa.util import types

        assert( (J is None) or (graph is None) )
         
        self.nq = None
        self.c = None

        if h is not None:
            self.h_terms(h)

        if J is not None:
            self.J_terms(J)

        if graph is not None:
            self.graph_terms(graph)

        super().__init__(self.c)

