from torch.autograd import Variable
from variational_circuit import VariationalCircuit
import torch
import pennylane as qml

class ChainedQPUs:
    """ Class to encapsulate the internals of a chained list of qpus
    """
    
    def __init__(self, variational_circuits):
        for var_circuit in variational_circuits:
            assert(type(var_circuit) == VariationalCircuit)
        self._n = len(variational_circuits)
        self._variational_circuits = variational_circuits

    def forward(self, variational_weights, starting_input):
        """ Computes one forward pass of the chained circuit, with the initial state
        being starting_input. Returns a list of outputs at each stage of the chain
        Note: starting_input is a classical value
        variational_weights is s list of all weights for the chain
        """
        intermediate_outputs = []
        inp = starting_input
        
        for i in range(self._n):
            out = self._variational_circuits[i].forward(variational_weights[i], inp)
            intermediate_outputs.append(out)
            inp = out

        self._variational_weights = variational_weights
        self._outputs = intermediate_outputs
        return self._outputs

def unit_test():
    # Lets have a qnode, with 3 qubit
    num_qubits = 2
    dev1 = qml.device("default.qubit", wires=num_qubits, shots=1)
    import numpy as np
    def layer(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.CNOT(wires=[0, 1])
   
    def embedding_layer(inp):
        qml.templates.embeddings.AmplitudeEmbedding(inp, [0, 1], pad=True,
                                                    normalize=True)

    def measurement():
        return qml.expval(qml.PauliZ(0))
        
    # Let's have 6 identical layers
    layers = [layer for i in range(6)]
    embeddings = embedding_layer

    v_circuit1 = VariationalCircuit(layers, num_qubits, embeddings,
            dev1, measurement)
    c_qpus = ChainedQPUs([v_circuit1])
    weights = torch.randn(6, num_qubits, 3, requires_grad=True)
    r_inp = [0.1, 0.4, 0.3, 0.2]
    val = c_qpus.forward([weights], r_inp)

if __name__ == "__main__":
    unit_test()




