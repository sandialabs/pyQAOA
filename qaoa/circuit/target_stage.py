from qaoa.circuit import CircuitStage

class TargetStage(CircuitStage):
    def __init__(self,A):
        self.A = A
        super().__init__(self.A.num_qubits())
 
    def is_target(self):
        return True

    def get_operator(self):
        return self.A    


