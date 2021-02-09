import qaoa
import os
import numpy as np
from continued_objective import ContinuedObjective

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(e)

if __name__ == '__main__':

    np.set_printoptions(precision=8,linewidth=400)

    opt_method   = 'trust-exact'
    graph_degree = 3
    graph_num    = 0
    num_qubits   = 12
    num_layers   = 6


    cwd = os.getcwd()
    path = os.path.join(cwd,'results')
    mkdir(path)

    filename = 'd{0}_g{1}_n{2}_p{3}_m{4}.csv'.format(graph_degree,graph_num,num_qubits,\
                                                 num_layers,opt_method)
    obj = qaoa.circuit.load_maxcut(degree=graph_degree,nvert=num_qubits,graph_num=graph_num,nlayers=num_layers)
    cobj = ContinuedObjective(obj)

    results = cobj.minimize()

    pathfile = os.path.join(path,filename)

    results.to_csv(pathfile)

