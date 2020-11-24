from qaoa.circuit import CircuitStage
from qaoa.util.math import scale, additive_assign, aypx

class UnitaryStage(CircuitStage):

    def __init__(self,A,psi=None,lam=None,dpsi=None,dlam=None):

        from qaoa.operators import HermitianOperator

        assert( isinstance(A,HermitianOperator) )        
        self.A = A
        self.U = self.A.propagator() 
        nq = self.A.num_qubits()
        N = 1 << nq

        import numpy as np
        self._psi  = np.zeros(N,dtype=complex) if (psi  is None) else psi
        self._lam  = np.zeros(N,dtype=complex) if (lam  is None) else lam
        self._dpsi = np.zeros(N,dtype=complex) if (dpsi is None) else dpsi
        self._dlam = np.zeros(N,dtype=complex) if (dlam is None) else dlam
        self.theta = 0
        self.dtheta = 0
        super().__init__(nq)

    def get_operator(self):
        return self.A

    def set_control(self,theta):
        if theta != self.theta:
            self.theta = theta
            self.U.set_control(self.theta)
            self.notify_compute_psi()

    def set_differential_control(self,dtheta):
        if dtheta != self.dtheta:
            self.dtheta = dtheta
            self.notify_compute_dpsi()

    def psi(self):
        if self.need_compute_psi:
            self.U.apply(self.prev.psi(),self._psi)
            self.need_compute_psi = False
            if not self.next.is_target():
               self.next.notify_compute_psi()
        return self._psi

    def lam(self):
        if self.need_compute_lam:
            self.U.apply(self.next.lam(),self._lam)
            self.need_compute_lam = False
            self.prev.notify_compute_lam()
        return self._lam

    def dpsi(self):
        work = self._dpsi
        if self.need_compute_dpsi:
            self.A.apply(self.prev.psi(),work)
            scale(1j*self.dtheta,work)
            if not self.prev.is_initial():
                additive_assign(self.prev.dpsi(),work )
            self.U.apply(work,self._dpsi)
            self.need_compute_dpsi = False
            self.next.notify_compute_dpsi()
        return self._dpsi

    def dlam(self):
        if self.need_compute_dlam:
            work = self._dlam
            self.A.apply(self.next.lam(),work)
            scale(1j*self.dtheta,work)
            additive_assign(self.next.dlam(),work)
            self.U.apply(work,self._dlam)
            self.need_compute_dlam = False
            self.prev.notify_compute_dlam()
        return self._dlam

    def deriv_1(self):
        psi = self.prev.psi()
        lam = self.lam()
        return 2 * self.A.inner_product(lam,psi).imag

    def deriv_2(self):
        psi = self.prev.psi()
        lam = self.lam()
        dlam = self.dlam()
        result = self.A.inner_product(dlam,psi)
        if not self.prev.is_initial():
            dpsi = self.prev.dpsi()
            result += self.A.inner_product(lam,dpsi)
        return 2 * result.imag


