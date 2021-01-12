from qaoa.operators import LinearOperator

class UnitaryOperator(LinearOperator):

    """
    Defines the interface of generic unitary operators and provides infrastructure used
    by derived types.
    """

    def __str__(self):
        return "UnitaryOperator"

    def apply_inverse(self,v,Uv):
        """
        Delegates to apply_adjoint() 
        """
        return self.apply_adjoint(v,Uv)

    def apply_adjoint_inverse(self,v,Uv):
        """
        Delegates to apply() 
        """
        return self.apply(v,Uv)

