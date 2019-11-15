import pennylane as qml
import numpy as np
# This is a helper file to store various measurements that might be useful

def get_sample(num_qubits):
    return qml.sample(qml.Hermitian(np.diag(range(2**num_qubits)), wires=range(num_qubits)))

