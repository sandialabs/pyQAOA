import qaoa
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 



if __name__ == '__main__':
    np.set_printoptions(precision=8,linewidth=400)

    nq = 12
    p = 12
    twop = p*2
    I = np.eye(twop)
    P = lambda k : np.vstack((I[:k],I[-k:])).T
    obj = qaoa.circuit.load_maxcut(nvert=nq,nlayers=p)

    f = lambda x,k : obj.value(P(k)@x)
    g = lambda x,k : P(k).T @ obj.gradient(P(k)@x)
    H = lambda x,k : P(k).T @ np.array([ obj.hessVec(P(k)@x,p) for p in P(k).T ]).T
    Hp = lambda x,v,k : P(k).T @ obj.hessVec(P(k)@x,P(k)@v) 
   
    
    t = np.pi*np.random.rand(2)
    solve = lambda theta: minimize(f,theta,args=(len(t)//2,),method='trust-exact',jac=g,hess=H)
#    solve = lambda theta: minimize(f,theta,args=(len(t)//2,),method='trust-krylov',jac=g,hessp=Hp)

    for k in range(1,p-1):
        result = solve(t)
        print(result['fun'])
        t = P(k+1).T@P(k)@result['x'] + np.random.randn(2*(k+1))*1e-3
    result = solve(t)
    print(result['fun'])
    
    print("True Minimum = ",obj.true_minimum())
