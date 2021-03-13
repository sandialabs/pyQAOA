import multiprocessing as mp
import queue
from copy import deepcopy

class QAOAProcess(mp.Process):

    def __init__(self,job,qctrl,qresult,quantities,nlayers,C,D=None,psi0=None):
        from qaoa.circuit import QAOACircuit
        super().__init__()
        self.qctrl = qctrl
        self.qresult = qresult
        self.quantities = quantities
        
        print("Starting job {0}".format(job))
        self.obj = QAOACircuit(nlayers,C,D,psi0)

    def run(self):
        while not self.qctrl.empty():
            try:

                result = list()
                theta = self.qctrl.get(block=False)

                if "Theta" in self.quantities:
                    result.append(theta)

                if "Value" in self.quantities:
                    fval = self.obj.value(theta)
                    result.append(fval)

                if "Gradient" in self.quantities:
                    grad = self.obj.gradient(theta)
                    result.append(grad)

                if "Hessian Eigenvalues" in self.quantities:
                    heig = self.obj.hess_eig(theta)
                    result.append(hess_eig)

                self.qresult.put(result)
                self.qctrl.task_done()
            except queue.Empty:
                break


def sample_QAOA(filename,quantities,nsamples,njobs,buffersize,ctrlgen,C,D=None,psi0=None):
    from qaoa.util import QueueLogger

    print("\nSetting up QAOA processes") 

    log = QueueLogger(filename,quantities,buffersize,nsamples)

    qctrl = mp.JoinableQueue()
    qresult = mp.Queue()

    [ qctrl.put(ctrlgen()) for k in range(nsamples) ]
    
    nlayers = len(ctrlgen())//2

    qaoa_procs = [ QAOAProcess(job,qctrl,qresult,quantities,nlayers,C,D,psi0) for job in range(njobs) ]

    log_proc = mp.Process(target=log.read,args=(qresult,))
    log_proc.start()

    [ qp.start() for qp in qaoa_procs ]

    qctrl.join()    

    log_proc.terminate()

    [ qp.join() for qp in qaoa_procs ]
 
    print("\nResults collected for {0} control angles and written to file {1}".format(nsamples,filename))
               
