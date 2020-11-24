from qaoa.operators import LinearOperator

class UnitaryOperator(LinearOperator):

    def __str__(self):
        return "UnitaryOperator"

    def apply_inverse(self,v,Uv):
        return self.apply_adjoint(v,Uv)

    def apply_adjoint_inverse(self,v,Uv):
        return self.apply(v,Uv)

