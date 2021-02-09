import numpy as np
from scipy.optimize import minimize 
import pandas as pd

class ContinuedObjective(object):

    """
    Set all layer angles after the first to zero,
    compute a minimizer, and restart with an additional layer
    """

    uses_jac = { "Nelder-Mead"  : False, 
                 "Powell"       : False,
                 "CG"           : True,
                 "BFGS"         : True,
                 "Newton-CG"    : True,
                 "L-BFGS-B"     : True,
                 "TNC"          : True,
                 "COBYLA"       : False,
                 "SLSQP"        : True,
                 "dogleg"       : True,
                 "trust-ncg"    : True,
                 "trust-krylov" : True,
                 "trust-exact"  : True }

    uses_hessp = { "Nelder-Mead"  : False, 
                   "Powell"       : False,
                   "CG"           : False,
                   "BFGS"         : False,
                   "Newton-CG"    : True,
                   "L-BFGS-B"     : False,
                   "TNC"          : False,
                   "COBYLA"       : False,
                   "SLSQP"        : False,
                   "dogleg"       : False,
                   "trust-ncg"    : True,
                   "trust-krylov" : True,
                   "trust-exact"  : False }

    uses_hess = { "Nelder-Mead"  : False, 
                  "Powell"       : False,
                  "CG"           : False,
                  "BFGS"         : False,
                  "Newton-CG"    : True,
                  "L-BFGS-B"     : False,
                  "TNC"          : False,
                  "COBYLA"       : False,
                  "SLSQP"        : False,
                  "dogleg"       : True,
                  "trust-ncg"    : True,
                  "trust-krylov" : True,
                  "trust-exact"  : True }

    # Golden Ratio
    phi = (1+np.sqrt(5))/2

    def __init__(self,obj):
        
        self.nl = obj.num_stages // 2
        self.obj = obj

    def minimize(self,theta=None,method="trust-exact"):

        assert method in self.uses_jac.keys()

        f = lambda x,P : self.obj.value(P@x)
        g = lambda x,P : P.T @ self.obj.gradient(P@x)
        H = lambda x,P : P.T @ np.array([ self.obj.hessVec(P@x,p) for p in P.T ]).T
        Hp = lambda x,v,P : P.T @ self.obj.hessVec(P@x,P@v) 
 
        kwargs = { 'method' : method }

        # Add function handles if the method supports them
        if self.uses_jac[method]:
            kwargs['jac'] = g

        if self.uses_hessp[method]:
            kwargs['hessp'] = Hp  

        if self.uses_hess[method]:
            kwargs['hess'] = H  

        if theta is None:
            theta = np.random.rand(2)*np.pi/2

        results = list()
        I = np.eye(2*self.nl)
        P = I[:,:2]

        for k in range(self.nl+1):
            
            r = minimize(f,theta,args=(P,),**kwargs)
            theta = r['x']
            x = P @ theta
            E = np.linalg.eig(self.obj.hessian(x))[0]
            fval = r['fun']

            results.append({ 'theta' : x, 
                             'fval'  : fval,
                             'gnorm' : np.linalg.norm(r['jac']),
                             'heig'  : E, 
                             'iter'  : r['nit'],
                             'nfval' : r['nfev'],
                             'ngrad' : r['njev'],
                             'success' : r['success'] })
 
            P = I[:,:2*(k+1)]

            theta0 = P.T @ x if k<self.nl else x
            
            alpha = 1e-4
            found_pt = False

            # Try to find a nearby restart point
            while not found_pt:
                dtheta = np.random.randn(len(theta0))
                theta = theta0 + alpha * dtheta
                fnew = f(theta,P)
                gnew = g(theta,P)
                found_pt = (fnew < fval) or (np.linalg.norm(gnew)>1e-5)
                alpha *= self.phi
            df = pd.DataFrame(results)           
        return df

