from qaoa.circuit import CircuitStage

class InitialStage(CircuitStage):

    def __init__(self,nq=None,psi0=None):
        import numpy as np
        assert( not ((nq is None) and (psi0 is None)) )
        if psi0 is None:
            N = 1 << nq 
            self._psi = np.ones(N,dtype=complex)/np.sqrt(N)
        else:
            self._psi = psi0
            nq = int(np.log2(len(self._psi)))
        super().__init__(nq)

    def get_operator(self):
        return None

    def get_stage_number(self):
        return 0

    def is_initial(self):
        return True

    def psi(self):
        return self._psi

