from qaoa.circuit import CircuitStage
from qaoa.util.math import scale, additive_assign, axpy

class UnitaryStage(CircuitStage):

    def __init__(self,A,psi=None,lam=None,dpsi=None,dlam=None):

        from qaoa.operators import HermitianOperator
        from numpy import zeros

        assert( isinstance(A,HermitianOperator) )        
        self.A = A
        self.U = self.A.propagator() 
        nq = self.A.num_qubits()
        N = 1 << nq

        self._psi  = zeros(N,dtype=complex) if (psi  is None) else psi
        self._lam  = zeros(N,dtype=complex) if (lam  is None) else lam
        self._dpsi = zeros(N,dtype=complex) if (dpsi is None) else dpsi
        self._dlam = zeros(N,dtype=complex) if (dlam is None) else dlam
        self.work  = zeros(N,dtype=complex) 
        self.theta = 0
        self.dtheta = 0
        super().__init__(nq)

    def get_operator(self):
        return self.A

    def get_unitary(self):
        return self.U

    def set_control(self,theta):
        if theta != self.theta:
            self.theta = theta
            self.U.set_control(self.theta)
            self.notify_compute_psi()
            self.notify_compute_lam()

    def set_differential_control(self,dtheta):
        if dtheta != self.dtheta:
            self.dtheta = dtheta
            self.notify_compute_dpsi()
            self.notify_compute_dlam()

    def psi(self):
        if self.need_compute_psi:
            self.U.apply(self.prev.psi(),self._psi)
            self.need_compute_psi = False
            if not self.next.is_target():
               self.next.notify_compute_psi()
        return self._psi

    def lam(self):
        if self.need_compute_lam:
            if self.next.is_target():
                self.next.get_operator().apply(self.psi(),self._lam)
            else:
                self.next.U.apply_adjoint(self.next.lam(),self._lam)
            self.prev.notify_compute_lam()
            self.need_compute_lam = False
        return self._lam

    def dpsi(self):
        if self.need_compute_dpsi:
            self.A.apply(self.prev.psi(),self.work)
            scale(self.num_qubits(),1j*self.dtheta,self.work)
            if not self.prev.is_initial():
                additive_assign(self.num_qubits(),self.prev.dpsi(),self.work )
            self.U.apply(self.work,self._dpsi)
            self.need_compute_dpsi = False
            self.next.notify_compute_dpsi()
        return self._dpsi

    def dlam(self):
        if self.need_compute_dlam:
            if self.next.is_target():
                self.next.get_operator().apply(self.dpsi(),self._dlam)
            else:
                self.next.U.apply_adjoint(self.next.dlam(),self._dlam)
                self.next.A.apply(self.lam(),self.work)
                axpy(self.num_qubits(),-1j*self.next.dtheta,self.work,self._dlam)
                
            self.need_compute_dlam = False
            self.prev.notify_compute_dlam()
        return self._dlam

    def deriv_1(self):
        return -2 * self.A.conj_inner_product(self.lam(),self.psi()).imag

    def deriv_2(self):
                return -2 * (self.A.conj_inner_product(self.dlam(),self.psi()) + \
                             self.A.conj_inner_product(self.lam(),self.dpsi()) ).imag


