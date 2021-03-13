lass ObjectiveSampler(object):

    @staticmethod
    def run(objlist,k,x,*opts):
        from numpy.linalg import norm
        obj = objlist[k]
        results = dict()

        if "Objective Value" in opts:
            results["fval"] = obj.value(x)
        if "Approximation Ratio "in opts:
            results["APR"] = obj.value(x)/obj.true_minimum() if  \
                             "value" not in results.keys() else results["value"]/obj.true_minimum()

        if "Gradient" in opts:
            results["grad"] = obj.gradient(x)
        if "Gradient Norm" in opts:
            results["gnorm"] = norm(obj.gradient(x)) if "grad" not in results.keys() else \
                               norm(results["grad"])

        if "Hessian Eigenvalues" in opts:
            results["heig"] = obj.hess_eig(x)
        
        return results

    def __init__(self,obj,num_threads=None):
        from multiprocessing import cpu_count
        from copy import deepcopy
        import numpy as np
        self.obj = [obj,]
        self.num_threads = max(1,cpu_count()//4) if num_threads is None else num_threads
        self.num_stages = len(obj)
        self.default_sample_dist = lambda k : np.random.rand(k)*np.pi/2
        [self.obj.append(deepcopy(obj)) for thread in range(1,self.num_threads)]

    def __call__(self,num_samples,*options,sample_dist=None):
        from pandas import DataFrame
        from multiprocessing import Pool
        pool = Pool(self.num_threads)
        if not len(options):
            options = ["Objective Value",]
        sample = self.default_sample_dist if sample_dist is None else sample_dist
        num_threads = self.num_threads
        num_stages = self.num_stages
        
        results = [ pool.apply_async( self.run, (self.obj,k%self.num_stages,sample(num_stages),*options) ) for k in range(num_samples) ]
        

        return DataFrame([ res.get() for res in results ])
       


