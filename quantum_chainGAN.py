import pennylane as qml
import torch.optim as optim
from variational_circuit import VariationalCircuit
from basic_parametric import *
from true_distribution_quantum import *
from bars_and_stripes import BarsAndStripes

dev1 = qml.device("default.qubit", wires=4, shots=1)
dev2 = qml.device("default.qubit", wires=4, shots=1)
    
def pauli_z_measurement(num_qubits):
    return qml.expval(qml.PauliZ(0))

def sample_measurement(num_qubits):
    return qml.sample(qml.Hermitian(np.diag(range(2**num_qubits)), wires=range(num_qubits)))

def basis_embedding(inp, num_qubits):
    print("here 2:", inp)
    assert(num_qubits == len(inp))
    qml.templates.embeddings.BasisEmbedding(np.array(inp), wires=[0,1,2,3])

def normalize_to_prob(inp):
    """ Convert something in [-1,1] to [0,1]
    """
    return (inp + 1)/2

def zero_state(num_qubits):
    zero_vec = [0 for i in range(2**num_qubits)]
    basis_embedding(zero_vec, num_qubits)
    
class Quantum_ChainGAN:
    """ A serialized generator architecture, where each component lives on an individual qpu
    """
    def __init__(self, num_qubits=4):
        self.num_discrim_layers = 5
        self.num_gen_layers = 5
        self.num_qubits = num_qubits
        self.num_qpu = 4

        dis_layers = [basic_4_qubit_layer_3 for i in range(self.num_discrim_layers)]
        self.discriminator = VariationalCircuit(dis_layers, self.num_qubits, basis_embedding, \
                dev1, pauli_z_measurement) 
        
        gen_layers = [basic_4_qubit_layer_3 for i in range(self.num_gen_layers)]
        self.generator = VariationalCircuit(gen_layers, self.num_qubits, basis_embedding, \
                dev2, sample_measurement) 
        
        self.data_generator = BarsAndStripes()

    def train(self, num_epochs):
        dis_weights = [torch.randn(self.num_discrim_layers, self.num_qubits, 3, requires_grad=True)]
        gen_weights = [torch.randn(self.num_discrim_layers, self.num_qubits, 3, requires_grad=True)]
        
        dis_optimizer = optim.Adam(dis_weights, lr=0.01)
        gen_optimizer = optim.Adam(gen_weights, lr=0.01)
        batch_size = 16

        # Discriminator pre train
        def train_discriminator(true_batch, fake_batch):
            print("Dis training")
            fake_scores, real_scores = 0, 0
            for i in range(batch_size):
                fake_scores = torch.log(1 - normalize_to_prob(self.discriminator.forward(dis_weights, \
                        fake_batch[i])))
                real_scores = torch.log(normalize_to_prob(self.discriminator.forward(dis_weights, \
                        true_batch[i])))
                
            dis_optimizer.zero_grad()
            discrim_loss = fake_scores - real_scores
            discrim_loss.backward()
            dis_optimizer.step()
            return loss.item()

        def pre_train(gen_weights, dis_weights, gen_optimizer, dis_optimizer):
            pre_train_iter = 100
            for i in range(pre_train_iter):
                true_batch = self.data_generator.generate_samples(self.num_qubits, batch_size//2)
                true_batch = torch.from_numpy(true_batch)
                true_batch = true_batch.reshape(batch_size, self.num_qubits)

                fake_batch = torch.randn(batch_size, self.num_qubits)
                for i in range(batch_size):
                    gen_weights = gen_weights*torch.randn(self.num_gen_layers, self.num_qubits, 3)
                    fake_batch[i] = torch.from_numpy(get_bit_string(self.generator.forward(gen_weights, \
                            torch.tensor([0 for i in range(self.num_qubits)])), self.num_qubits))
               
                loss = train_discriminator(true_batch, fake_batch)
                print("Pre train iter:", i, "Loss:", loss)

        pre_train(gen_weights[0], dis_weights[0], gen_optimizer, dis_optimizer)


qgan = Quantum_ChainGAN()
qgan.train(100)


