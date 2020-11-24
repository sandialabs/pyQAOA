import numpy as np
from qaoa.circuit import QuantumCircuit

class QAOACircuit(QuantumCircuit):


    def __init__(self,p,C,D=None,psi0=None):
        """
        Simulates a Quantum Approximate Optimization Algorithm circuit with p layers
        using a Hamiltonian C and driver Hamiltonian/mixing operator D

        Evaluates the objective function :math:`f(\theta) = \langle \psi_{2p}|C|\psi_2p\rangle` where

        .. math::

            \psi_k = U_k(\theta_k) \psi_{k-1} \\
            U_k(\theta_k) = exp(i \theta_k A_k)\\
            A_{2k} = D, \quad A_{2k+1} = C

        Parameters
        ----------
        p - (positive integer) Number of QAOA circuit layers. NOTE: Each layer has two stages
        C - (qaoa.operators.HermitianOperator) The provided Hamiltonian operator
        D - (qaoa.operators.HermitianOperator) The driver Hamiltonian aka "mixing operator" 
            If not provided, the circuit will use the default 

        .. math::

            D = \sum_{k=1}^n X_j \\
            X_k = I_{2^k} \otimes \sigma_x \otimes I_{2^{n-k-1}} \\
            \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
        """

        if D is None:
            from qaoa.operators import SumSigmaXOperator
            D = SumSigmaXOperator(C.num_qubits())
        else:
            from qaoa.operators import HermitianOperator
            assert( isinstance(D,HermitianOperator) )

        super().__init__([C,D]*p,C,psi0)
