import qaoa
from qaoa.sampling import sample_QAOA
import multiprocessing as mp
import numpy as np

if __name__ == '__main__':

    nvert = 12
    ngraph = 0
    nlayers = 3
    ndegree = 3
    nsamples = 500
    njobs = mp.cpu_count()
    buffersize = 20

    # Result output file
    resfile = 'output.csv'

    # Quantities to evaluate
    quantities = ["Theta", "Value", "Gradient"]

    # Function that returns a new random control
    ctrlgen = lambda : np.random.rand(2*nlayers)*np.pi/2

    # Create a Max Cut Ising Hamiltonian from a graph in the library
    C = qaoa.operators.load_max_cut_hamiltonian(ndegree,nvert,ngraph)
    sample_QAOA(resfile,quantities,nsamples,njobs,buffersize,ctrlgen,C)

    
  
