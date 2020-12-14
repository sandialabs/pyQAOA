from qaoa.operators import UnitaryOperator

class Propagator(UnitaryOperator):

    def __init__(self,A,theta=0):
        self.A = A
        self.theta = self.A.dtype.type(theta)
        super().__init__(A.nq)

    def __str__(self):
        return "Propagator"

    def set_control(self,theta):
        self.theta = self.A.dtype.type(theta)

    def get_operator(self):
        return self.A

    def as_matrix(self):
        from scipy.linalg import expm
        return expm(1j*self.theta*self.A.as_matrix())

    @staticmethod
    def create(A,theta=0):
        import qaoa.operators as ops
        prop_types = [ op.split('.')[-1] for op in dir(ops) if 'Propagator' in op ]
        prop = str(A).replace("Oper","Propag")
        if prop in prop_types:
            return eval("ops.{0}(A,theta)".format(prop),locals())
        else:
           raise TypeError("Operator type {0} does not have a corresponding Propagator".format(str(A)))

