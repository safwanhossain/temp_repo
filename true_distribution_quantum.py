import pennylane as qml
import numpy as np
from measurements import get_sample

class TrueDistributionQuantum:
    """ Class to generate a distribution from a complicated quantum circuit. We will use the 
    samples generated from this distribution as a training dataset on which to learn using
    generative modelling. In that since, this represents the true distribution """

    def __init__(self, num_qubits, num_layers=10):
        """ num_qubits will indicate the size of the system. The size of the distribution support 
            will be of size 2^num_qubits 
        """
        self._num_qubits = num_qubits
        self._num_layers = num_layers 
        self._device = qml.device("default.qubit", wires=self._num_qubits, shots=1)
        self._qnode = qml.QNode(self.complex_circuit, self._device).to_torch()
    
    def entangle_all(self, wires):
        """ Wires is list of qubits to entangle
        """
        for i in range(len(wires)):
            for j in range(i+1, len(wires)):
                qml.CNOT(wires=[wires[i], wires[j]])

    def rotate_all_qubits(self, wires):
        """ rotate each qubit according to some arbitrary weight
        """
        W = np.random.uniform(0, 2*np.pi, (len(wires),3))
        for i in range(len(wires)):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=wires[i])

    def complex_circuit(self):
        wires = [i for i in range(self._num_qubits)]
        for layer in range(self._num_layers):
            self.rotate_all_qubits(wires)
            self.entangle_all(wires)
        return get_sample(self._num_qubits)

    def get_samples(self, n):
        """ Get n samples from the quantum device constructed
        """
        samples = []
        for i in range(n):
            samples.append(int(self._qnode()))
        return samples

    ## accessor methods
    def get_num_qubits(self):
        return self._num_qubits
    
    def get_num_layers(self):
        return self._num_layers
    
    def get_device(self):
        return self._device

def get_one_hot(samples, num_qubits):
    """ To sample, we use a Hermitian operator. For 2^k possible states (for a k qubit system)
    it returns an integer between 0 and 2^k-1. This function converts them to equivalent bit strings
    """
    num_states = np.power(2, num_qubits)
    one_hot_matrix = np.zeros((len(samples), num_states))
    for i, sample in enumerate(samples):
        one_hot_matrix[i, sample] = 1
    return one_hot_matrix

def get_bit_string(samples, num_qubits):
    samples = samples.view(-1, 1)
    num_states = np.power(2, num_qubits)
    bit_strings = np.zeros((len(samples), num_qubits))
    
    for i, sample in enumerate(samples):
        total = num_states - 1
        bit = 0
        while total >= 1:
            if sample == 0:
                total = total // 2
                continue
            if total // sample > 1:
                bit_strings[i,bit] = 0
            else:
                bit_strings[i,bit] = 1
                sample = sample - ((total+1)//2)
            bit += 1
            total = total // 2
                
    return bit_strings

######### Unit Testing #############
def convert_bit_string_test():
    print("----------- RUNNING UNIT TEST 1 --------------")
    bit_strings = get_bit_string([i for i in range(2**3)], 4)
    print(bit_strings)

def true_distribution_test():
    print("----------- RUNNING UNIT TEST 2 --------------")
    test_distribution = TrueDistributionQuantum(2, 5) 
    samples = test_distribution.get_samples(10)
    print(samples)
    one_hot = get_one_hot(samples, 2)
    print(one_hot)

if __name__ == "__main__":
    convert_bit_string_test()
    true_distribution_test()


