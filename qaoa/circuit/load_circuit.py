def load_maxcut(degree=3,nvert=8,graph_num=0,nlayers=1,driver="SumSigmaX",driver_args=None):
    import qaoa
    from qaoa.circuit import QAOACircuit
    from qaoa.util.graph import load
    from qaoa.operators import IsingHamiltonian

    G = load(degree,nvert,graph_num)
    C = IsingHamiltonian(graph=G)

    if driver_args == None:
        driver_args = [nvert]

    if isinstance(driver,str):
        D = eval("qaoa.operators.{0}Operator(*{1})".format(driver,driver_args),locals())
    elif isinstance(driver,type):
        D = driver(*driver_args) 
    else:
        D = driver
#    if driver == "SumSigmaX":
#        from qaoa.operators import SumSigmaXOperator
#        D = SumSigmaXOperator(nvert)
#    else:
#        D = driver(*driver_args)

    obj = QAOACircuit(nlayers,C,D=D)
    return obj
