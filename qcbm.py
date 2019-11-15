import pennylane as qml
from pennylane import numpy as np
from variational_circuit import VariationalCircuit
from chained_qpu import ChainedQPUs
from circuit_architectures.liu_wang import liu_wang_layers
import torch

class QCBM(VariationalCircuit):
    def __init__(self, layers, num_qubits, embedding_layer,
                 device, measurement):
        VariationalCircuit.__init__(self, layers, num_qubits, embedding_layer,
                                    device, measurement)
    def cost(self):
        pass


class MultiQCBM(ChainedQPUs):
    pass


if __name__ == "__main__":
    n_wires = 2
    dev1 = qml.device("default.qubit", wires=n_wires)

    n_layers = 5
    layers = liu_wang_layers(n_layers)

    def embedding_layer(inp):
        angles = np.pi*(inp + 1)
        qml.templates.embeddings.AngleEmbeddings(features=angles, wires=range(n_wires), rotation='X')


    def measurement():
        return [qml.expval(qml.PauliZ(k)) for k in range(n_wires)]


    embeddings = embedding_layer

    n_qnodes = 1
    QCBMs = []
    for _ in range(n_qnodes):
        machine = QCBM(layers, n_wires, embeddings,
                    dev1, measurement)
        QCBMs.append(machine)
    chained_qcbms = MultiQCBM(QCBMs)
    weights = torch.randn(6, n_wires, 3)


    initial_input = 2*np.pi*torch.randn(1, n_wires)
    chained_qcbms.forward([weights], initial_input)


