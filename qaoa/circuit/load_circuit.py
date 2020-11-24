def load_maxcut(degree=3,nvert=8,graph_num=0,nlayers=1,driver="SumSigmaX",*args):

    from qaoa.circuit import QAOACircuit
    from qaoa.util.graph import load
    from qaoa.operators import IsingHamiltonian

    G = load(degree,nvert,graph_num)
    C = IsingHamiltonian(graph=G)

    if driver == "SumSigmaX":
        from qaoa.operators import SumSigmaXOperator
        D = SumSigmaXOperator(nvert)
    else:
        D = driver(*args)

    obj = QAOACircuit(nlayers,C,D=D)
    return obj
