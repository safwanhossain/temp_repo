import numpy as np
import math
import matplotlib.pyplot as plt

class BarsAndStripes:
    """ Dataset for the bars and stripes"""

    def generate_samples(self, num_qubits, num_samples):
        assert(math.sqrt(num_qubits) == int(math.sqrt(num_qubits)))
        side_len = int(math.sqrt(num_qubits))
        dataset = np.zeros((num_samples*2, side_len, side_len))

        for k in range(num_samples):
            init = np.zeros((side_len, side_len))
            nr_of_stripes = np.random.randint(low = 1, high=side_len, size = 1)
            columns = np.random.randint(side_len, size=nr_of_stripes)

            for i in columns:
                init[:, i] = 1.0
            
            dataset[2*k] = init
            dataset[2*k+1] = init.T
        
        return dataset
        
    def plot_data(self, bar_stripe_sample):
        plt.imshow(bar_stripe_sample)
        plt.show()


def unit_test():
    bs_data = BarsAndStripes()
    dataset = bs_data.generate_samples(16, 10)
    assert(dataset.shape[0] == 2*10)
    print(dataset[0])

if __name__ == "__main__":
    unit_test()


