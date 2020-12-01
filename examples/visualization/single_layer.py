import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import qaoa
from qaoa.circuit import load_maxcut
from qaoa.util.finterp1 import interpolation_matrix
import itertools 


class InterpolatedObjective(object):

    def __init__(self,ni,ne):

        self.ni = ni
        t = 2*np.pi*np.arange(ne)/(ne)
        self.x,self.y = np.meshgrid(t,t)

        self.Fi = np.zeros((self.ni,self.ni))
        self.L = interpolation_matrix(self.ni,ne)

    def __call__(self,obj):

        for k in itertools.product(range(self.ni),repeat=2):
            theta = 2*np.pi*np.array(k)/(ni)
            self.Fi[k] = obj.value(theta)
    
        return self.L @ self.Fi @ self.L.T





if __name__ == '__main__':

    ni = 40  # Number of interpolation points per dimension
    ne = 100 # Number of evaluation points per dimension

    plt.rcParams.update({"text.usetex":True})
    fig, (ax0, ax1) = plt.subplots(1,2)

    F = InterpolatedObjective(ni,ne)

    nq = 16
    
    Dx = qaoa.operators.SumSigmaXOperator(nq)
    Dp = qaoa.operators.ProjectionOperator(nq=nq)

    # Create a QAOA objective for MAX CUT using a graph from the library
    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=1,driver=Dx)
    c0 = ax0.contourf(F.x,F.y,F(obj),cmap=plt.cm.viridis)
    plt.colorbar(c0,ax=ax0)
    

    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=1,driver=Dp)
    c1 = ax1.contourf(F.x,F.y,F(obj),cmap=plt.cm.viridis)
    plt.colorbar(c1,ax=ax1)

    plt.show()
    

    
