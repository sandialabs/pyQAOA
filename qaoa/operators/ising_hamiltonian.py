from qaoa.operators import DiagonalOperator
import numpy as np

class IsingHamiltonian(DiagonalOperator):

    """
    A specialized DiagonalOperator for representing an Ising Model Hamiltonian
    """

    def __init__(self,nq=None,h=None,J=None,graph=None):
        """
        Create an IsingHamiltonian object from Ising Model expansion coefficients and/or a graph

        There are several ways that the IsingHamiltonian may be specified. h and J may be either
        dense or sparse. If both are sparse, then the number of qubits must be separately provided 
        with the parameter nq. Interaction coefficients may be specified either through the J container
        directly or via a (possibly weighted) graph object, but not both. 

        Parameters
        ----------
        nq : unsigned int, optional
            Number of qubits
        h : numpy.ndarray (1D) or list of (index,value) pairs, optional
            Container of external field coefficients
        J : numpy.ndarray (2D) or list of (row,column,value) tuples, optional
            Container of interaction term coefficients
        graph : networkx.Graph or compatible type, optional
            Object must have a member variable called edges that may be weighted

        Examples
        --------
   
        1. Dense representation of external field coefficients        

        .. math:: H = -Z_1 + 2 Z_2

        >>> h = np.array((-1,2))
        >>> C = IsingHamiltionian(h=h)
        >>> print(C.data)
        [1. -3. 3. 1.]

        2. Sparse representation of external field coefficients
   
        .. math:: H = Z_1 - Z_4

        >>> nq = 4
        >>> indices = [0,3]
        >>> values = [1,-1]
        >>> h = list(zip(indices,values))
        >>> C = IsingHamiltonian(nq=nq,h=h)
        >>> print(C.data)
        [ 0.  2.  0.  2.  0.  2.  0.  2. -2.  0. -2.  0. -2.  0. -2.  0.]

        3. Dense representation of interaction coefficients

        .. math:: H = 3 Z_1 Z_2 - Z_1 Z_3 - 2 Z_2 Z_3

        >>> J = np.array(((0,3,-1),(0,0,-2),(0,0,0)))
        >>> C = IsingHamiltonian(J=J)
        >>> print(C.data)
        [ 0.  6. -2. -4. -4. -2.  6.  0.]

        4. Sparse representation of interaction coefficients
 
        .. math:: H = \\frac{1}{4} Z_1 Z_2 - Z_1 Z_3 + \\frac{3}{4} Z_2 Z_3
       
        >>> nq = 3
        >>> rows = [0,0,1]
        >>> cols = [1,2,2]
        >>> vals = [1/4,-1,3/4]
        >>> J = list(zip(rows,cols,vals))
        >>> C = IsingHamiltonian(nq=nq,J=J)
        >>> print(C.data)
        [ 0.   0.5 -2.   1.5  1.5 -2.   0.5  0. ] 

        5. Construction from an unweighted graph
 
        .. math:: H = \\sum\\limits_{(i,j)\\in E} Z_i Z_j

        >>> import networkx as nx
        >>> G = nx.random_regular_graph(3,6,seed=6714)
        >>> E = set(G.edges)
        >>> print(E)
        {(1, 2), (0, 4), (1, 5), (4, 3), (0, 3), (2, 0), (4, 5), (2, 5), (1, 3)}
        >>> C = IsingHamiltonian(graph=G)
        >>> print(C.data)
        [ 9.  3.  3.  1.  3. -3.  1. -1.  3.  1. -3. -1. -3. -5. -5. -3.  3.  1.
         -3. -1.  1. -1. -1.  1.  1.  3. -5.  1. -1.  1. -3.  3.  3. -3.  1. -1.
          1. -5.  3.  1.  1. -1. -1.  1. -1. -3.  1.  3. -3. -5. -5. -3. -1. -3.
          1.  3. -1.  1. -3.  3.  1.  3.  3.  9.]

        This example works identically if weights would be assigned to the graph prior to 
        constructing C

        """

        from qaoa.util import types
        self.nq = None
        self.c = None
        assert( (J is None) or (graph is None) )

        if nq is not None: 
            self.nq = nq
            self.c =  np.zeros(1<<nq)

        if h is not None:
            self.h_terms(h)

        if J is not None:
            self.J_terms(J)

        if graph is not None:
            self.graph_terms(graph)

        super().__init__(self.c)
    def graph_terms(self,G):
        """
        Internally-used method used based on a graph edge set description 
        """
        from qaoa.util.graph import is_graph, graph_edges
        from numba.typed import List
        from qaoa.util.math import ising_sparse_J, ising_sparse_weighted_J, ising_dense_J
        from networkx import is_weighted

        if self.nq is not None:
            assert(self.nq==len(G))
        else:
            self.nq = len(G)
            self.c = np.zeros(1<<self.nq)

        J = graph_edges(G)
        JL = List()
        [JL.append(e) for e in J]
        if is_weighted(G):
            ising_sparse_weighted_J(JL, self.c) 
        else:
            ising_sparse_J(JL,self.c)

    def J_terms(self,J):
        """
        Internally-used method used to populate the interaction matrix J
        """

        from qaoa.util.types import is_container, is_squarematrix

        if is_squarematrix(J):
            if self.nq is not None:
                assert(self.nq==J.shape[0])
            else:
                self.nq = J.shape[0]
                self.c = np.zeros(1<<self.nq)

            from qaoa.util.math import ising_dense_J
            ising_dense_J(J,self.c)

        elif is_container(J):
            assert(self.c is not None)
            from numba.typed import List
            JL = List()
            [JL.append(e) for e in J]
            if len(J[0]) == 2:
                from qaoa.util.math import ising_sparse_J
                ising_sparse_J(JL,self.c)                      
            else: 
                from qaoa.util.math import ising_sparse_weighted_J
                ising_sparse_weighted_J(JL,self.c)                      
        else:
            raise TypeError("Argument of type {0} is unsupported".format(type(J))) 

    def h_terms(self,h):
        """
        Internally-used method used to populate the external field coefficient vector
        """

        from qaoa.util.math import ising_sparse_h, ising_dense_h
        from qaoa.util.types import is_container, is_nparray

        if is_nparray(h):
            if self.nq is not None:
                assert(self.nq==len(h))
            else:
                self.nq = len(h)
            if self.c is None:
                self.c = np.zeros(1<<self.nq)
            ising_dense_h(h,self.c)

        elif is_container(h):
            assert(self.c is not None)
            from numba.typed import List
            hL = List()
            [hL.append(e) for e in h]
            ising_sparse_h(hL,self.c)
   
        else:
            raise TypeError("Argument of type {0} is unsupported".format(type(h))) 
 

  
