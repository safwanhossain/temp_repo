import pennylane as qml


class VariationalCircuit:
    """ Class to encapsulate the internals of a variational circuit as
        used in a chained list of qpus.
    """

    # Internal variables storing info the variational_circuit
    def _circuit(self, weights, inp=None):
        """ High level topology of the variational circuit
        """
        print("I be here", inp)
        self._embedding_layer(inp, self._num_qubits)
        for i, W in enumerate(weights):
            self._layers[i](W)
        return self._measurement(self._num_qubits)
    
    def __init__(self, layers, num_qubits, embedding_layer,
                 device, measurement):
        """ Layers are a list of functions, implementing the variational layers; 
            Weights are a list of weight tensors for each layer
            embedding_layers are a list of functions embedding a classical vector to quantum
            Device is a quantum device
            measurement is a mesaurement fucntion (ex: qml.expval(qml.PauliZ(0))
        """
        self._num_qubits = num_qubits
        self._num_layers = len(layers)
        self._layers = layers
        self._embedding_layer = embedding_layer
        self._qnode = qml.QNode(self._circuit, device).to_torch()
        self._measurement = measurement

    def forward(self, weights, inp):
        """ Executes a forward pass of the variational circuit
        """
        print("Weights: ", weights)
        print("Input: ", inp)
        return self._qnode(weights, inp=inp)

