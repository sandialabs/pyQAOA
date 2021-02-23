import numpy as np

class QuantumCircuit(object):
    """
    Simulates a quantum circuit formed by a sequence of controlled unitary operators U1,...Un, which are
    generated from Hermitian operators H1,...,Hn. Evalutes the expectation value of the given Hamiltonian
    using control-parameterized states generated by the circuit. 

    Attributes
    ----------

    A : list 
        A list of Hermitian operators. One for every UnitaryStage in the circuit and an 
        additional for the TargetStage.
    num_stages : unsigned int
        The number of UnitaryStages 
    num_qubits : unsigned int
        The number of qubits in a stage. Every stage must have the same number of qubits. 
       
    """
    def __init__(self,ops,H,psi0=None):

        from qaoa.operators import HermitianOperator  
        from qaoa.circuit import CircuitStage, InitialStage, UnitaryStage, TargetStage

        self.count = { 'value'    : 0, \
                       'gradient' : 0, \
                       'hess_vec' : 0, \
                       'hessian'  : 0 }

        self.num_stages = len(ops)
        self.A = [A for A in ops]
        self.A.append(H)

        # Require all operators to be Hermitian
        assert all( isinstance(A,HermitianOperator) for A in self.A ) 

        # Ensure each stage has the same number of qubits
        nqs = [A.num_qubits() for A in self.A]
        self.num_qubits = nqs[0]
 
        assert nqs.count(self.num_qubits) == self.num_stages+1 

        # Allocate workspace vectors
        #  
        # 1 vector for psi0
        # 4 vectors per UnitaryStage (psi,lam,dpsi,dlam)

        N = 1 << self.num_qubits
        L = 4*self.num_stages
        self.work = np.zeros((N,L+1),dtype=complex)

        self.psi0 = self.work[:,-1]

        self.psi0[:] = np.ones(N,dtype=complex)/np.sqrt(N) if psi0 is None else psi0

        self.psi  = self.work[:,0:L:4]
        self.lam  = self.work[:,1:L+1:4]
        self.dpsi = self.work[:,2:L+2:4]
        self.dlam = self.work[:,3:L+3:4]

        self.stage = [InitialStage(psi0=self.psi0)]

        [self.stage.append(UnitaryStage(A,  psi=self.psi[:,k],   lam=self.lam[:,k],   \
                                           dpsi=self.dpsi[:,k], dlam=self.dlam[:,k])) \
        for k,A in enumerate(self.A[:-1])]

        self.stage.append(TargetStage(self.A[-1]))
        CircuitStage.link(*self.stage)


    def __len__(self):
        return self.num_stages

    def __deepcopy__(self,memo):
        from copy import deepcopy
        import numpy
        Acopy = list()
        shallow = [ np.where([y is x for x in self.A])[0] for y in self.A ]        
        for k, A in enumerate(self.A):
            k0 = shallow[k][0]
            if k0 == k:
                Acopy.append(deepcopy(A,memo))
            else:
                Acopy.append(Acopy[k0])  
        qc_copy = QuantumCircuit(Acopy[:-1],Acopy[-1],numpy.copy(self.psi0))
        qc_copy.set_control(self.get_control())
        qc_copy.set_differential_control(self.get_differential_control())
        return qc_copy


    def reset_count(self):
        for key in self.count.keys():
            self.count[key] = 0

    def true_minimum(self):
        """
        Return the true minimum expectation value of this circuit's target operator if possible
        """
        return self.A[-1].true_minimum()

    def true_maximum(self):
        """
        Return the true maximum expectation value of this circuit's target operator if possible
        """
        return self.A[-1].true_maximum()

    def set_control(self,theta):
        """
        Assign control angles to each stage
        """
        assert(len(theta)==self.num_stages)
        [ self.stage[k+1].set_control(theta[k]) for k in range(self.num_stages) ]

    def set_differential_control(self,dtheta):
        """
        Assign differential control angles to each stage
        """
        assert(len(dtheta)==self.num_stages)
        [ self.stage[k+1].set_differential_control(dtheta[k]) \
          for k in range(self.num_stages) ]

    def get_control(self):
        return np.array([ stage.theta for stage in self.stage[1:-1]])

    def get_differential_control(self):
        return np.array([ stage.dtheta for stage in self.stage[1:-1]])

    def final_state(self,theta):
        return self.stages[-1].psi()

    def value(self,theta):
        """
        Compute the objective function at a point theta
        """
        self.count["value"] += 1
        self.set_control(theta)
        return self.A[-1].expectation(self.stage[-2].psi())

    def gradient(self,theta):
        """
        Compute the gradient of the objective function at a point theta
        """
        self.count["gradient"] += 1
        self.set_control(theta)
        return np.array([self.stage[k+1].deriv_1() \
                         for k in range(self.num_stages)])

    def gradient_norm(self,theta):
        """
        Compute the norm of the gradient of the objective function at a point theta
        """
        return np.linalg.norm(self.gradient(theta))

    def hess_vec(self,theta,dtheta):
        """
        Compute the action of the Hessian matrix evaluated at 
        a point theta on a direction vector dtheta
        """
        self.count["hess_vec"] += 1
        self.set_control(theta)
        self.set_differential_control(dtheta)
        return np.array([self.stage[k+1].deriv_2() \
                         for k in range(self.num_stages)])
 
    def hessian(self,theta):
        """
        Evaluate the Hessian matri at a point theta
        """
        self.count["hessian"] += 1
        self.set_control(theta)
        I = np.eye(self.num_stages)
        return np.array([self.hess_vec(theta,e) for e in I])

    def hess_eig(self,theta):
        """
        Compute the eigenvalues of the Hessian at the point theta
        """
        return np.linalg.eig(self.hessian(theta))[0]
