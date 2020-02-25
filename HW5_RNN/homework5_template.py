import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        # TODO: IMPLEMENT ME

    def backward (self, y):
        # TODO: IMPLEMENT ME
        pass

    def forward (self, x):
        # TODO: IMPLEMENT ME
        pass

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    print xs
    print ys
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    # TODO: IMPLEMENT ME
