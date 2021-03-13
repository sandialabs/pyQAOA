def load_max_cut_hamiltonian(degree=3,nvert=8,graph_num=0):
    import qaoa
    G = qaoa.util.graph.load(degree,nvert,graph_num)
    C = qaoa.operators.IsingHamiltonian(graph=G)
    return C
