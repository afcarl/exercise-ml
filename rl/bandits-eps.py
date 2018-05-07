import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Bandit:

    def __init__(self, m):
        self.m = m # the actual mean
        self.mean = 0 # guesstimated mean
        self.N = 0

    def pull(self):
        # get some random value with unit variance
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        # update the mean
        self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N*x


def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    data = np.empty(N)

    for i in range(N):
        # epsilon greedy
        p = np.random.random()

        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)

        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    # plt.plot(cumulative_average)
    # plt.plot(np.ones(N)*m1)
    # plt.plot(np.ones(N)*m2)
    # plt.plot(np.ones(N)*m3)
    # plt.xscale("log")
    # plt.show()

    # for b in bandits:
    #     print(b.mean)

    return cumulative_average

if __name__ == '__main__':
    c_10 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)

    plt.plot(c_10, label="epsilon = 0.1")
    plt.plot(c_05, label="epsilon = 0.05")
    plt.plot(c_01, label="epsilon = 0.05")
    plt.legend()
    plt.xscale("log")
    plt.show()

    # plt.plot(c_10, label="epsilon = 0.1")
    # plt.plot(c_05, label="epsilon = 0.05")
    # plt.plot(c_01, label="epsilon = 0.05")
    # plt.legend()
    # plt.show()