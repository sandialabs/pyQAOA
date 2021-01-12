from qaoa.util import DocumentationInheritance

class CircuitStage(object,metaclass=DocumentationInheritance):

    """
    Interface class for a stage in a quantum circuit

    Attributes
    ----------
    nq : unsigned int
        Number of qubits
    prev : CircuitStage
        The circuit stage before this one (if any)
    next : CircuitStage
        The circuit stage after this one (if any)
    need_compute_psi : bool 
        Flag that indicates that the state variable must be recomputed
    need_compute_lam : bool 
        Flag that indicates that the adjoint variable must be recomputed
    need_compute_dpsi : bool 
        Flag that indicates that the state sensitivity must be recomputed
    need_compute_dlam : bool 
        Flag that indicates that the adjoint sensitivity must be recomputed
    """

    import abc

    def __init__(self,nq):
        self.nq = nq
        self.prev = None
        self.next = None
        self.need_compute_psi  = True
        self.need_compute_lam  = True
        self.need_compute_dpsi = True
        self.need_compute_dlam = True

    @abc.abstractmethod
    def get_operator(self):
        """
        Return the Hermitian operator that generates the Unitary for this stage, if it exists
        """
        raise NotImplementedError("get_operator method not implemented")

    @abc.abstractmethod
    def psi(self):
        """
        Return the state vector. Recompute it only if its dependencies have changed. 
        
        Returns
        -------
        psi : numpy.ndarray
            The state vector with complex-valued elements. Has length 2**nq, where nq is the number of qubits
        """
        raise NotImplementedError("psi method not implemented")

    @abc.abstractmethod
    def lam(self):
        """
        Return the adjoint vector. Recompute it only if its dependencies have changed. 
        
        Returns
        -------
        lam : numpy.ndarray
            The adjoint vector with complex-valued elements. Has length 2**nq, where nq is the number of qubits
        """
        raise NotImplementedError("lam method not implemented")

    @abc.abstractmethod
    def dpsi(self):
        """
        Return the state sensitivity vector. Recompute it only if its dependencies have changed. 
        
        Returns
        -------
        dpsi : numpy.ndarray
            The state sensitivity vector with complex-valued elements. Has length 2**nq, where nq is the number of qubits
        """
 
        raise NotImplementedError("dpsi method not implemented")

    @abc.abstractmethod
    def dlam(self):
        """
        Return the adjoint sensitivity vector. Recompute it only if its dependencies have changed. 
        
        Returns
        -------
        dlam : numpy.ndarray
            The adjoint sensitivity vector with complex-valued elements. Has length 2**nq, where nq is the number of qubits
        """
        raise NotImplementedError("dlam method not implemented")

    @abc.abstractmethod
    def num_qubits(self):
        """
        Return the number of qubits in this stage
  
        Returns
        -------
        nq : unsigned int
            The number of qubits in this stage
        """
        return self.nq

    @abc.abstractmethod
    def is_initial(self):
        """
        Indicates whether there is a previous CircuitStage

        Returns
        -------
        flag : bool
            True if there is no previous CircuitStage, False otherwise
        """
        return self.prev is None

    @abc.abstractmethod
    def is_target(self):
        """
        Indicates whether there is a next CircuitStage

        Returns
        -------
        flag : bool
            True if there is no next CircuitStage, False otherwise
        """        
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

    @staticmethod
    def link(s1,s2,*rest):
        assert(s1.num_qubits() == s2.num_qubits())
        s1.next = s2
        s2.prev = s1
        if len(rest):
            CircuitStage.link(s2,*rest)


