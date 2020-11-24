from qaoa.circuit import CircuitStage
from qaoa.util.math import negate_real
import numpy as np

class TargetStage(CircuitStage):
    def __init__(self,A,lam=None,dlam=None):
        self.A = A
        nq = A.num_qubits()
        N = 1 << nq
        self._lam = np.zeros(N,dtype=complex) if (lam is None) else lam
        self._dlam = np.zeros(N,dtype=complex) if (dlam is None) else dlam
        super().__init__(nq)

    def get_operator(self):
        return self.A    

    def lam(self):
        if self.need_compute_lam:
            self.A.apply(self.prev.psi(),self._lam[:]) 
            negate_real(self._lam)
            self.prev.notify_compute_lam()
            self.need_compute_lam = False
        return self._lam

    def dlam(self):
        if self.need_compute_dlam:
            self.A.apply(self.prev.dpsi(),self._dlam[:])
            negate_real(self._dlam)
            self.prev.notify_compute_dlam()
            self.need_compute_dlam = False
        return self._dlam

