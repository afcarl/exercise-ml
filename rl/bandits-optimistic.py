import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Bandit:

    def __init__(self, m, upper_limit):
        self.m = m # the actual mean
        self.mean = upper_limit # guesstimated mean
        self.N = 1

    def pull(self):
        # get some random value with unit variance
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        # update the mean
        self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N*x


def run_experiment(m1, m2, m3, N, upper_limit=10):
    bandits = [Bandit(m1, upper_limit), Bandit(m2, upper_limit), Bandit(m3, upper_limit)]
    data = np.empty(N)

    for i in range(N):
        # optimistic initial values
        j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    return cumulative_average

if __name__ == '__main__':
    optimistic = run_experiment(1.0, 2.0, 3.0, 100000)

    plt.plot(c_10, label="optimistic")
    plt.legend()
    plt.xscale("log")
    plt.show()
