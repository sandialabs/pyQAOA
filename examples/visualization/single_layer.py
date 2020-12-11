import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import qaoa


def plot_landscape(obj,ax,ni=20,ne=100):

    f = qaoa.util.SineInterp(obj,ni)
    gamma,beta = f.nd_grid(2,ne)   

    fvals = f.values(ne)
    fmin = np.min(fvals)
    fmax = np.max(fvals)
    fnorm = (fvals-fmin)/(fmax-fmin)
    colors = plt.cm.viridis(fnorm)

    levels = np.arange(int(fmin),int(fmax)+1,int(fmax-fmin)//8)

    labels = ['$0$',r'$\frac{\pi}{8}$',r'$\frac{\pi}{4}$',r'$\frac{3\pi}{8}$',r'$\frac{\pi}{2}$']
    ticks = np.pi*np.arange(5)/8

    title = "{0}-Qubits, True Min = {1}".format(nq,obj.true_minimum())
    ax.set_title(title,fontsize=10)

    ax.imshow(colors,interpolation='lanczos',extent=(0,np.pi/2,0,np.pi/2),origin='lower')
    lclrs = [ 'w' if l<0 else 'k' for l in levels  ]
    C = ax.contour(beta,gamma,fvals,levels,colors=lclrs,linestyles=':')
    ax.set_xlim(0,np.pi/2)
    ax.set_xlabel(r'$\beta$',fontsize=12)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels,fontsize=12)

    ax.set_ylim(0,np.pi/2)
    ax.set_ylabel(r'$\gamma$',fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels,fontsize=12)

    ax.clabel(C,levels,inline=1,fontsize=10,fmt="%d")
    ax.set_aspect('equal','box')
    ax.grid(True,linestyle='--')



if __name__ == '__main__':

    ni = 15 # Number of interpolation points per dimension
    ne = 60 # Number of evaluation points per dimension

    fig, axes = plt.subplots(2,2)


    for k, ax in enumerate(axes.flatten()):

        # Number of Qubits
        nq = 10 + 2*k

        # Create a QAOA objective for MAX CUT using a graph from the library
        obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=1)
        
        plot_landscape(obj,ax)
        plt.suptitle("QAOA Max Cut Objective Landscape",fontsize=16)

    plt.show()
    

    
