import qaoa

class CircuitStage(object):

    def __init__(self,nq):
        self.nq = nq
        self.prev = None
        self.next = None
        self.need_compute_psi  = True
        self.need_compute_lam  = True
        self.need_compute_dpsi = True
        self.need_compute_dlam = True

    def get_operator(self):
        raise NotImplementedError("get_operator method not implemented")

    def psi(self):
        raise NotImplementedError("psi method not implemented")

    def lam(self):
        raise NotImplementedError("lam method not implemented")

    def dpsi(self):
        raise NotImplementedError("dpsi method not implemented")

    def dlam(self):
        raise NotImplementedError("dlam method not implemented")

    def num_qubits(self):
        return nq

    def is_initial(self):
        return self.prev is None

    def is_target(self):
        return self.next is None

    def notify_compute_psi(self):
        self.need_compute_psi  = True
        self.need_compute_lam  = True
        self.need_compute_dpsi = True
        self.need_compute_dlam = True
        if isinstance(self.next,CircuitStage):
            self.next.notify_compute_psi()

    def notify_compute_lam(self):
        self.need_compute_lam  = True
        self.need_compute_dlam = True
        if isinstance(self.prev,CircuitStage):
            self.prev.notify_compute_lam()

    def notify_compute_dpsi(self):
        self.need_compute_dpsi = True
        self.need_compute_dlam = True
        if isinstance(self.next,CircuitStage):
            self.next.notify_compute_dpsi()

    def notify_compute_dlam(self):
        self.need_compute_dlam = True
        if isinstance(self.prev,CircuitStage):
            self.prev.notify_compute_dlam()

    def get_stage_number(self):
        return 1+self.prev.get_stage_number() if self.prev is not None else 0

def link_stages(s1,s2,*srest):
    assert(s1.nq == s2.nq)
    s1.next = s2
    s2.prev = s1
    if len(srest):
        link_stages(s2,*srest)

